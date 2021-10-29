"""A module for interfacing RDS data with external models.
Currently contains only LFADSInterface, which handles breaking up
continuous or trialized RDS data into smaller sections known as chops,
as well as reassembling these chops back into continuous or trialized
data after they are processed by LFADS.
"""

import logging
import os
from collections import defaultdict
from os import path

import h5py
import numpy as np
import pandas as pd

# Standard data names expected by LFADS
DATA_NAME = "data"
EXT_INPUT_NAME = "ext_input"
INDEX_NAME = "inds"
# Mapping from `signal_type`s to HDF5 data labels
DEFAULT_CHOP_MAP = {"spikes": DATA_NAME}
# Mapping from LFADSOutput fields to `signal_type`s
DEFAULT_MERGE_MAP = {
    "rates": "lfads_rates",
    "factors": "lfads_factors",
    "gen_inputs": "lfads_gen_inputs",
}
# NOTE: deEMG NOT IMPLEMENTED IN lfads_tf2
# Mapping for EMG chopping
DEFAULT_EMG_CHOP_MAP = {"emg_notch": DATA_NAME}
# Mapping from LFADSOutput fields to `signal_type`s
DEFAULT_EMG_MERGE_MAP = {
    "rates": "deEMG_means",
    "factors": "deEMG_factors",
    "gen_inputs": "deEMG_gen_inputs",
}

# Standard per-module logging system
logger = logging.getLogger(__name__)


class SegmentRecord:
    """Stores information needed to reconstruct a segment from chops."""

    def __init__(self, seg_id, clock_time, offset, n_chops, overlap):
        """Stores the information needed to reconstruct a segment.
        Parameters
        ----------
        seg_id : int
            The ID of this segment.
        clock_time : pd.Series
            The TimeDeltaIndex of the original data from this segment.
        offset : int
            The offset of the chops from the start of the segment.
        n_chops : int
            The number of chops that make up this segment
        overlap : int
            The number of bins of overlap between adjacent chops.
        """
        self.seg_id = seg_id
        self.clock_time = clock_time
        self.offset = offset
        self.n_chops = n_chops
        self.overlap = overlap

    def rebuild_segment(self, chops, smooth_pwr=2):
        """Reassembles a segment from its chops.
        Parameters
        ----------
        chops : np.ndarray
            A 3D numpy array of shape n_chops x seg_len x data_dim that
            holds the data from all of the chops in this segment.
        smooth_pwr : float, optional
            The power to use for smoothing. See `merge_chops`
            function for more details, by default 2
        Returns
        -------
        pd.DataFrame
            A DataFrame of reconstructed segment data, indexed by the
            clock_time of the original segment.
        """

        # Merge the chops for this segment
        merged_array = merge_chops(
            chops,
            overlap=self.overlap,
            orig_len=len(self.clock_time) - self.offset,
            smooth_pwr=smooth_pwr,
        )
        # Add NaNs for points that were not modeled due to offset
        data_dim = merged_array.shape[1]
        offset_nans = np.full((self.offset, data_dim), np.nan)
        merged_array = np.concatenate([offset_nans, merged_array])
        # Recreate segment DataFrame with the appropriate `clock_time`s
        segment_df = pd.DataFrame(merged_array, index=self.clock_time)
        return segment_df


class LFADSInterface:
    def __init__(
        self,
        window,
        overlap,
        max_offset=0,
        chop_margins=0,
        random_seed=None,
        chop_fields_map=DEFAULT_CHOP_MAP,
        merge_fields_map=DEFAULT_MERGE_MAP,
    ):
        """Initializes an LFADSInterface.
        Parameters
        ----------
        window : int
            The length of chopped segments in ms
        overlap : int
            The overlap between chopped segments in ms.
        max_offset : int, optional
            The maximum offset of the first chop from the beginning of
            each segment in ms. The actual offset will be chose
            randomly. By default, 0 adds no offset.
        chop_margins : int, optional
            The size of extra margins to add to either end of each chop
            in bins, designed for use with the temporal_shift operation
            in LFADS, by default 0.
        random_seed : int, optional
            The random seed for generating the dataset, by default None
            does not use a random seed.
        chop_fields_map : dict, optional
            A dictionary mapping the column groups of the neural
            DataFrame to names in the LFADS input file. By default,
            maps 'spikes' to 'data'.
        merge_fields_map : dict, optional
            A dictionary mapping elements of the LFADSOutput tuple to
            names in the full DataFrame. By default, maps `rates` to
            `lfads_rates`, `factors` to `lfads_factors`, and
            `gen_inputs` to `lfads_gen_inputs`.
        """

        def to_timedelta(ms):
            return pd.to_timedelta(ms, unit="ms")

        self.window = to_timedelta(window)
        self.overlap = to_timedelta(overlap)
        self.max_offset = to_timedelta(max_offset)
        self.chop_margins = chop_margins
        self.random_seed = random_seed
        self.chop_fields_map = chop_fields_map
        self.merge_fields_map = merge_fields_map

    def chop(self, neural_df):
        """Chops a trialized or continuous RDS DataFrame.
        Parameters
        ----------
        neural_df : pd.DataFrame
            A continuous or trialized DataFrame from RDS.
        Returns
        -------
        dict of np.array
            A data_dict of the chopped data. Consists of a dictionary
            with data tags mapping to 3D numpy arrays with dimensions
            corresponding to samples x time x features.
        """

        # Set the random seed for the offset
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Get info about the column groups to be chopped
        fields_map = self.chop_fields_map
        data_fields = sorted(fields_map.keys())

        def get_field_dim(field):
            return len(getattr(neural_df, field).columns)

        data_dims = [get_field_dim(f) for f in data_fields]
        data_splits = data_dims[:-1]
        input_fields = [fields_map[f] for f in data_fields]
        # Report information about the fields that are being chopped
        logger.info(
            f"Mapping data field(s) {data_fields} to LFADS input "
            f"field(s) {input_fields} with dimension(s) {data_dims}."
        )

        # Calculate bin widths and set up segments for chopping
        if "trial_id" in neural_df:
            # Trialized data
            bin_width = neural_df.clock_time.iloc[1] - neural_df.clock_time.iloc[0]
            segments = neural_df.groupby("trial_id")
        else:
            # Continuous data
            bin_width = neural_df.index[1] - neural_df.index[0]
            segments = {1: neural_df.reset_index()}.items()

        # Calculate the number of bins to use for chopping parameters
        window = int(self.window / bin_width)
        overlap = int(self.overlap / bin_width)
        chop_margins_td = pd.to_timedelta(self.chop_margins * bin_width, unit="ms")

        # Get correct offset based on data type
        if "trial_id" in neural_df:
            # Trialized data
            max_offset = int(self.max_offset / bin_width)
            max_offset_td = self.max_offset

            def get_offset():
                return np.random.randint(max_offset + 1)

        else:
            # Continuous data
            max_offset = 0
            max_offset_td = pd.to_timedelta(max_offset)

            def get_offset():
                return 0

            if self.max_offset > pd.to_timedelta(0):
                # Doesn't make sense to use offset on continuous data
                logger.info("Ignoring offset for continuous data.")

        def to_ms(timedelta):
            return int(timedelta.total_seconds() * 1000)

        # Log information about the chopping to be performed
        chop_message = " - ".join(
            [
                "Chopping data for LFADS",
                f"Window: {window} bins, {to_ms(self.window)} ms",
                f"Overlap: {overlap} bins, {to_ms(self.overlap)} ms",
                f"Max offset: {max_offset} bins, {to_ms(max_offset_td)} ms",
                f"Chop margins: {self.chop_margins} bins, {to_ms(chop_margins_td)} ms",
            ]
        )
        logger.info(chop_message)

        # Iterate through segments, which can be trials or continuous data
        data_dict = defaultdict(list)
        segment_records = []
        for segment_id, segment_df in segments:
            # Get the data from all of the column groups to extract
            data_arrays = [getattr(segment_df, f).values for f in data_fields]
            # Concatenate all data types into a single segment array
            segment_array = np.concatenate(data_arrays, axis=1)
            if self.chop_margins > 0:
                # Add padding to segment if we are using chop margins
                seg_dim = segment_array.shape[1]
                pad = np.full((self.chop_margins, seg_dim), 0.0001)
                segment_array = np.concatenate([pad, segment_array, pad])
            # Sample an offset for this segment
            offset = get_offset()
            # Chop all of the data in this segment
            chops = chop_data(
                segment_array,
                overlap + 2 * self.chop_margins,
                window + 2 * self.chop_margins,
                offset,
            )
            # Split the chops back up into the original fields
            data_chops = np.split(chops, data_splits, axis=2)
            # Create the data_dict with LFADS input names
            for field, data_chop in zip(input_fields, data_chops):
                data_dict[field].append(data_chop)
            # Keep a record to represent each original segment
            seg_rec = SegmentRecord(
                segment_id, segment_df.clock_time, offset, len(chops), overlap
            )
            segment_records.append(seg_rec)
        # Store the information for reassembling segments
        self.segment_records = segment_records
        # Consolidate data from all segments into a single array
        data_dict = {name: np.concatenate(c) for name, c in data_dict.items()}
        # Report diagnostic info
        dict_key = list(data_dict.keys())[0]
        n_chops = len(data_dict[dict_key])
        n_segments = len(segment_records)
        logger.info(f"Created {n_chops} chops from {n_segments} segment(s).")

        return data_dict

    def chop_and_save(
        self,
        neural_df,
        fname,
        valid_ratio=0.2,
        valid_block=1,
        heldin_ratio=1.0,
        overwrite=False,
    ):
        """Chops the data from an RDS DataFrame and saves it to an HDF5
        file compatible with LFADS.
        Parameters
        ----------
        neural_df : pd.DataFrame
            A continuous or trialized DataFrame from RDS.
        fname : str
            The path to the file where the data should be saved.
        valid_ratio : float, optional
            The ratio of validation data, by default 0.2
        valid_block : int, optional
            The number of consecutive samples in validation sample
            blocks, consistent with a "blocked" validation strategy.
            Ensures minimal overlap between training and validation
            sets, by default 1.
        heldin_ratio : float, optional
            The ratio of samples to use for a heldin subset on which
            to train LFADS. This option creates an additional LFADS data
            file with the added prefix 'heldin_' which contains a
            fraction of the total training and validation data. The idea
            is to train LFADS on the heldin subset and perform posterior
            sampling on the full dataset. This argument is useful for
            large datasets.
        overwrite : bool, optional
            Whether to overwrite any existing file , by default False
        """

        # Check if output file exists
        if not overwrite and path.isfile(fname):
            raise AssertionError(
                f"File {fname} already exists. " "Set `overwrite`=True to overwrite."
            )
        # Create any directories if necessary
        data_dir = path.dirname(fname)
        os.makedirs(data_dir, exist_ok=True)
        # Chop the data and save segment records internally
        data_dict = self.chop(neural_df)
        # Get the total number of available samples
        dict_key = list(data_dict.keys())[0]
        n_samples = len(data_dict[dict_key])
        # Compute whether sample blocks belong in train or valid
        n_blocks = np.ceil(n_samples / valid_block).astype(int)
        block_nums = np.repeat(np.arange(n_blocks), valid_block)
        in_valid = block_nums % np.round(1 / valid_ratio) == 0
        # Trim extra samples from incomplete blocks
        in_valid = in_valid[:n_samples]
        # Get the training and validation samples
        (valid_inds,) = np.where(in_valid)
        (train_inds,) = np.where(~in_valid)

        # Add function for saving data to the H5 file for LFADS
        def save_data(fname, train_inds, valid_inds):
            with h5py.File(fname, "w") as h5file:
                for ind_name, inds in zip(
                    ["train_", "valid_"], [train_inds, valid_inds]
                ):
                    # Save the index
                    h5file.create_dataset(ind_name + INDEX_NAME, data=inds)
                    for data_tag, samples in data_dict.items():
                        # Save each data type in the data_dict
                        h5file.create_dataset(ind_name + data_tag, data=samples[inds])
            # Report an informative message
            logger.info(
                f"Successfully wrote {len(train_inds)} train and "
                f"{len(valid_inds)} valid samples to {fname}."
            )

        # Write all data for the DataFrame
        save_data(fname, train_inds, valid_inds)
        # Save a heldin subset if specified
        if heldin_ratio < 1.0:
            # Add the 'heldin_' prefix to the file name
            dirname, basename = path.dirname(fname), path.basename(fname)
            heldin_fname = path.join(dirname, "heldin_" + basename)
            # Compute heldin sample indices
            n_train, n_valid = len(train_inds), len(valid_inds)
            in_heldin_train = np.arange(n_train) % np.round(1 / heldin_ratio) == 0
            in_heldin_valid = np.arange(n_valid) % np.round(1 / heldin_ratio) == 0
            heldin_train_inds = train_inds[np.where(in_heldin_train)[0]]
            heldin_valid_inds = valid_inds[np.where(in_heldin_valid)[0]]
            # Save the heldin subset file
            save_data(heldin_fname, heldin_train_inds, heldin_valid_inds)

    def merge(self, data_dict, smooth_pwr=2):
        """Merges the chops to reconstruct the original input
        sequence.
        Parameters
        ----------
        data_dict : dict of np.array
            A dictionary of data to merge, where the first dimension is
            samples, the second dimension is time, and the third
            dimension is variable.
        smooth_pwr : float, optional
            The power to use for smoothing. See `merge_chops`
            function for more details, by default 2
        Returns
        -------
        pd.DataFrame
            A merged DataFrame indexed by the clock time of the original
            chops. Columns are multiindexed using `fields_map`.
            Unmodeled data is indicated by NaNs.
        """

        # Get the desired arrays from the output
        fields_map = self.merge_fields_map
        output_fields = sorted(fields_map.keys())
        output_arrays = [data_dict[f] for f in output_fields]
        # Keep track of boundaries between the different signals
        output_dims = [a.shape[-1] for a in output_arrays]
        # Concatenate the output arrays for more efficient merging
        output_full = np.concatenate(output_arrays, axis=-1)
        # Get info for separating the chops related to each segment
        seg_splits = np.cumsum([s.n_chops for s in self.segment_records])[:-1]
        # Separate out the chops for each segment
        seg_chops = np.split(output_full, seg_splits, axis=0)
        # Reconstruct the segment DataFrames
        segment_dfs = [
            record.rebuild_segment(chops, smooth_pwr)
            for record, chops in zip(self.segment_records, seg_chops)
        ]
        # Concatenate the segments with clock_time indices
        merged_df = pd.concat(segment_dfs)
        # Add mulitindexed columns
        signal_types = [fields_map[f] for f in output_fields]
        midx_tuples = [
            (sig, f"{i:04}")
            for sig, dim in zip(signal_types, output_dims)
            for i in range(dim)
        ]
        merged_df.columns = pd.MultiIndex.from_tuples(midx_tuples)

        return merged_df

    def load_and_merge(self, fname, orig_df, smooth_pwr=2):
        """Loads the posterior averages from an LFADS output file and places
        them in the appropriate locations in the original DataFrame.
        Parameters
        ----------
        fname : str
            The path to the LFADS output file.
        orig_df : pd.DataFrame
            The continuous or trial_data DataFrame that was used to
            generate the LFADS input.
        smooth_pwr : float, optional
            The power to use for smoothing. See `merge_chops`
            function for more details, by default 2
        Returns
        -------
        pd.DataFrame
            The original DataFrame with the merged data columns added.
        """

        # Load the model output
        with h5py.File(fname, "r") as hf:
            data_dict = {k: np.array(v) for k, v in hf.items()}
        # Merge training and validation sets
        merge_data = {}
        for field in self.merge_fields_map.keys():
            train_data = data_dict["train_" + field]
            valid_data = data_dict["valid_" + field]
            data = np.full_like(np.concatenate([train_data, valid_data]), np.nan)
            data[data_dict["train_inds"].astype(int)] = train_data
            data[data_dict["valid_inds"].astype(int)] = valid_data
            merge_data[field] = data
        # Merge the LFADS output using self.segment_records
        merged_df = self.merge(merge_data, smooth_pwr=smooth_pwr)
        # Concatenate the merged data to the original data using clock_time
        if "trial_id" in orig_df:
            # Temporarily use clock_time as the index
            concat_df = orig_df.set_index("clock_time")
            full_df = pd.concat([concat_df, merged_df], axis=1)
            full_df = full_df.reset_index()
        else:
            full_df = pd.concat([orig_df, merged_df], axis=1)
        # TODO: Make sure spike column names are inherited by rates

        return full_df


class deEMGInterface(LFADSInterface):
    def __init__(
        self,
        *args,
        clip_quantile=0.99,
        scale_quantile=0.95,
        scale_factor=1,
        log_transform=False,
        chop_fields_map=DEFAULT_EMG_CHOP_MAP,
        merge_fields_map=DEFAULT_EMG_MERGE_MAP,
        **kwargs,
    ):
        """Implements interface functionality specific to EMG data.
        Also allows args and kwargs from `LFADSInterface.__init__`.
        Parameters
        ----------
        clip_quantile : float, optional
            The quantile above which to clip EMG values, by default 0.99.
            Note 1.0 is 100th quantile and doesn't perform clipping.
        scale_quantile : float, optional
            The quantile to use for range scaling of EMG values, by
            default 0.95. Note 1.0 is the 100th percentile and will use
            the max for scaling, but None will perform no scaling.
        scale_factor : int, optional
            The multiplier for rescaling EMG , by default 1
        log_transform : bool, optional
            Whether to log-transform the EMG, by default False
        """
        # Record transformation parameters
        self.clip_quantile = clip_quantile
        self.scale_quantile = scale_quantile
        self.scale_factor = None if scale_quantile is None else scale_factor
        self.log_transform = log_transform
        self.scale_quants = None
        # Perform initialization as defined in the base class
        super().__init__(
            *args,
            chop_fields_map=chop_fields_map,
            merge_fields_map=merge_fields_map,
            **kwargs,
        )

    def chop(self, neural_df, *args, **kwargs):
        """Chops a trialized or continuous RDS DataFrame.
        Also allows args and kwargs from `LFADSInterface.chop`.
        Parameters
        ----------
        neural_df : pd.DataFrame
            A continuous or trialized DataFrame from RDS.
        """

        # Get the data to be transformed
        fields_map = self.chop_fields_map
        data_fields = sorted(fields_map.keys())
        trans_df = neural_df[data_fields]
        # Ensure that the EMG data is positive
        trans_df = trans_df.abs()
        # Clip the data by quantile to remove outliers
        clip_quants = trans_df.quantile(self.clip_quantile)
        trans_df = trans_df.clip(upper=clip_quants, axis=1)
        # Rescale the data to the desired range
        if self.scale_quantile is not None:
            self.scale_quants = trans_df.quantile(self.scale_quantile)
            trans_df = trans_df / self.scale_quants * self.scale_factor
        # Log-transform the data
        if self.log_transform:
            trans_df = np.log(trans_df)
        # Bring transformed values back into the original dataframe
        neural_df.update(trans_df)
        # Report transformation of EMG data
        logger.info(
            f"Transforming EMG data - "
            f"Clip quantile: {self.clip_quantile}, "
            f"Scale quantile: {self.scale_quantile}, "
            f"Scale factor: {self.scale_factor}, "
            f"Log transform: {self.log_transform}"
        )
        # Perform chopping and saving as defined in the base class
        data_dict = super().chop(neural_df, *args, **kwargs)
        return data_dict


# ========== STATELESS CHOPPING AND MERGING FUNCTIONS ==========


def chop_data(data, overlap, window, offset=0):
    """Rearranges an array of continuous data into overlapping segments.

    This low-level function takes a 2-D array of features measured
    continuously through time and breaks it up into a 3-D array of
    partially overlapping time segments.
    Parameters
    ----------
    data : np.ndarray
        A TxN numpy array of N features measured across T time points.
    overlap : int
        The number of points to overlap between subsequent segments.
    window : int
        The number of time points in each segment.
    Returns
    -------
    np.ndarray
        An SxTxN numpy array of S overlapping segments spanning
        T time points with N features.

    See Also
    --------
    lfads_tf2.utils.merge_chops : Performs the opposite of this operation.

    """

    # Random offset breaks temporal connection between trials and chops
    offset_data = data[offset:]
    shape = (
        int((offset_data.shape[0] - offset - overlap) / (window - overlap)),
        window,
        offset_data.shape[-1],
    )
    strides = (
        offset_data.strides[0] * (window - overlap),
        offset_data.strides[0],
        offset_data.strides[1],
    )
    chopped = (
        np.lib.stride_tricks.as_strided(offset_data, shape=shape, strides=strides)
        .copy()
        .astype("f")
    )
    return chopped


def merge_chops(data, overlap, orig_len=None, smooth_pwr=2):
    """Merges an array of overlapping segments back into continuous data.
    This low-level function takes a 3-D array of partially overlapping
    time segments and merges it back into a 2-D array of features measured
    continuously through time.
    Parameters
    ----------
    data : np.ndarray
        An SxTxN numpy array of S overlapping segments spanning
        T time points with N features.
    overlap : int
        The number of overlapping points between subsequent segments.
    orig_len : int, optional
        The original length of the continuous data, by default None
        will cause the length to depend on the input data.
    smooth_pwr : float, optional
        The power of smoothing. To keep only the ends of chops and
        discard the beginnings, use np.inf. To linearly blend the
        chops, use 1. Raising above 1 will increasingly prefer the
        ends of chops and lowering towards 0 will increasingly
        prefer the beginnings of chops (not recommended). To use
        only the beginnings of chops, use 0 (not recommended). By
        default, 2 slightly prefers the ends of segments.
    Returns
    -------
    np.ndarray
        A TxN numpy array of N features measured across T time points.

    See Also
    --------
    lfads_tf2.utils.chop_data : Performs the opposite of this operation.
    """

    if smooth_pwr < 1:
        logger.warning(
            "Using `smooth_pwr` < 1 for merging " "chops is not recommended."
        )

    merged = []
    full_weight_len = data.shape[1] - 2 * overlap
    if overlap > 0:
        # Create x-values for the ramp
        x = np.linspace(1 / overlap, 1 - 1 / overlap, overlap)
        # Compute a power-function ramp to transition
        ramp = 1 - x ** smooth_pwr
    else:
        # Use a placeholder ramp
        ramp = np.full(0, np.nan)
    ramp = np.expand_dims(ramp, axis=-1)
    # Compute the indices to split up each chop
    split_ixs = np.cumsum([overlap, full_weight_len])
    for i in range(len(data)):
        # Split the chop into overlapping and non-overlapping
        first, middle, last = np.split(data[i], split_ixs)
        # Ramp each chop and combine it with the previous chop
        if i == 0:
            last = last * ramp
        elif i == len(data) - 1:
            first = first * (1 - ramp) + merged.pop(-1)
        else:
            first = first * (1 - ramp) + merged.pop(-1)
            last = last * ramp
        # Track all of the chops in a list
        merged.extend([first, middle, last])
    # Cover the case where there first dimension of chops is zero
    if len(merged) < 1:
        n_samples, _, data_dim = data.shape
        merged = [np.empty((n_samples, data_dim))]
    merged = np.concatenate(merged)
    # Indicate unmodeled data with NaNs
    if orig_len is not None and len(merged) < orig_len:
        nans = np.full((orig_len - len(merged), merged.shape[1]), np.nan)
        merged = np.concatenate([merged, nans])

    return merged


if __name__ == "__main__":
    """A demonstration of the use of LFADSInterface."""
    # Configure the loggers
    logging.basicConfig(level=logging.INFO)

    # Set parameters for data generation
    BIN_SIZE = 20  # in ms
    WINDOW = 1000  # in ms
    OVERLAP = 300  # in ms
    MAX_OFFSET = 100  # in ms
    # NOTE: We don't use chop_margins here because it relies on LFADS
    # dropping margins in its posterior average arrays, which has not
    # yet been implemented in lfads_t2.
    CHOP_MARGINS = 0  # in bins
    RANDOM_SEED = 0

    # =========== DEMONSTRATE EMG CHOPPING ===========
    from snel_toolkit.datasets.area2 import EXAMPLE_FILE, Area2Dataset

    TEST_DIR = path.expanduser("~/tmp/area2")
    DATA_FILE = path.join(TEST_DIR, "lfads_input/lfads_data.h5")
    MODEL_DIR = path.join(TEST_DIR, "lfads_output")
    PS_FILENAME = "posterior_samples.h5"
    PS_PATH = path.join(MODEL_DIR, PS_FILENAME)

    # Load, resample, and trialize the dataset
    dataset = Area2Dataset(path.basename(EXAMPLE_FILE))
    dataset.load(EXAMPLE_FILE)
    dataset.resample(BIN_SIZE / 1000)
    trial_data = dataset.make_trial_data()

    # Initialize the interface
    interface = LFADSInterface(
        WINDOW,
        OVERLAP,
        max_offset=MAX_OFFSET,
        chop_margins=CHOP_MARGINS,
        random_seed=RANDOM_SEED,
        # NOTE: We wouldn't typically use force as an external input
        chop_fields_map={"spikes": "data", "force": "ext_input"},
    )

    # Chop the data and save an input file for LFADS
    interface.chop_and_save(trial_data, DATA_FILE, overwrite=True)

    # If the posterior average file does not exist, train and sample LFADS
    if not path.isfile(PS_PATH):
        # Only load TF / LFADS if required
        from lfads_tf2.defaults import get_cfg_defaults
        from lfads_tf2.models import LFADS

        # Get the default configuration
        cfg_node = get_cfg_defaults()
        # Make some updates to the configuration
        cfg_node.MODEL.DATA_DIM = len(trial_data.spikes.columns)
        cfg_node.MODEL.SEQ_LEN = int(WINDOW / BIN_SIZE)
        cfg_node.MODEL.EXT_INPUT_DIM = len(trial_data.force.columns)
        cfg_node.MODEL.CD_RATE = 0.3
        cfg_node.TRAIN.BATCH_SIZE = 512
        cfg_node.TRAIN.L2.GEN_SCALE = 1e-4
        cfg_node.TRAIN.L2.CON_SCALE = 1e-4
        cfg_node.TRAIN.KL.IC_WEIGHT = 1e-4
        cfg_node.TRAIN.KL.CO_WEIGHT = 1e-4
        cfg_node.TRAIN.DATA.DIR = path.dirname(DATA_FILE)
        cfg_node.TRAIN.DATA.PREFIX = path.basename(DATA_FILE)
        cfg_node.TRAIN.MODEL_DIR = MODEL_DIR
        cfg_node.TRAIN.OVERWRITE = True
        model = LFADS(cfg_node=cfg_node)
        N_EPOCHS = 10
        # Train for a few epochs and sample the model
        for epoch in range(N_EPOCHS):
            model.train_epoch()
        # NOTE: This model is not intended to be a good fit
        model.sample_and_average(ps_filename=PS_FILENAME)

    # Load the LFADS outputs and merge them back to the continuous DF
    trial_data = interface.load_and_merge(PS_PATH, trial_data)
    logger.info("Chopping and merging demonstration complete.")
    print(trial_data)

    # =========== DEMONSTRATE EMG CHOPPING ===========
    CLIP_QUANTILE = 0.95
    SCALE_QUANTILE = 0.9
    SCALE_FACTOR = 10
    LOG_TRANSFORM = True

    from snel_toolkit.datasets.xds import EXAMPLE_FILE, XDSDataset

    TEST_DIR = path.expanduser("~/tmp/xds")
    DATA_FILE = path.join(TEST_DIR, "lfads_input/lfads_data.h5")
    MODEL_DIR = path.join(TEST_DIR, "lfads_output")
    PS_FILENAME = "posterior_samples.h5"
    PS_PATH = path.join(MODEL_DIR, PS_FILENAME)

    dataset = XDSDataset(path.basename(EXAMPLE_FILE))
    dataset.load(EXAMPLE_FILE)
    dataset.resample(BIN_SIZE / 1000)
    trial_data = dataset.make_trial_data()
    interface = deEMGInterface(
        WINDOW,
        OVERLAP,
        max_offset=MAX_OFFSET,
        chop_margins=CHOP_MARGINS,
        random_seed=RANDOM_SEED,
        clip_quantile=CLIP_QUANTILE,
        scale_quantile=SCALE_QUANTILE,
        scale_factor=SCALE_FACTOR,
        log_transform=LOG_TRANSFORM,
    )
    interface.chop_and_save(trial_data, DATA_FILE, overwrite=True)
