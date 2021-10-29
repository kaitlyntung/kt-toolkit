import logging

import h5py
import numpy as np
import pandas as pd
from lfads_tf2 import compat

# TODO: Remove dependence on lfads_tf2
from lfads_tf2.utils import load_posterior_averages, merge_chops
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from ..utils import rgetattr
from .base import BaseDataset

logger = logging.getLogger(__name__)


class RTTDataset(BaseDataset):
    def load(self, fname, multiunits=False, m1_only=True, freq=0.001):
        """Method for loading data from O'Doherty's random target task.
        Parameters
        ----------
        fname : str
            The path to the `.mat` file of interest.
        multiunits : bool, optional
            Whether to use multiunit channels or spike-sorted channels,
            by default False uses multiunit channels.
        m1_only : bool, optional
            Whether to use only channels from M1 or also include S1
            channels, by default True only uses M1.
        freq : float, optional
            The desired frequency of the data in seconds, by default 0.001
        """

        logger.info(f"Loading RTT dataset from {fname}")
        with h5py.File(fname, "r") as h5file:
            h5dict = {key: val[()] for key, val in h5file.items() if key != "#refs#"}
            # Pull integers representing channel name characters
            int_arrays = [h5file[ref][()] for ref in h5dict["chan_names"][0]]

            # Convert integers to corresponding characters
            def make_chan_name(array):
                return "".join([chr(num) for num in array])

            chan_names = [make_chan_name(array) for array in int_arrays]
            if multiunits:
                # Pull unsorted spike times
                spike_refs = h5dict["spikes"][0]
                spike_times = []
                for chan_name, ref in zip(chan_names, spike_refs):
                    if m1_only and "M1" not in chan_name:
                        continue
                    spike_times.append(h5file[ref][()])
            else:
                # Pull sorted spike times, ignoring M1 if specified
                spike_refs = h5dict["spikes"][1:].T
                spike_times = []
                for chan_name, chan_spike_refs in zip(chan_names, spike_refs):
                    # Iterate through channels and extract sorted times
                    if m1_only and "M1" not in chan_name:
                        continue
                    # Get times for each of the units
                    chan_times = [h5file[ref][()] for ref in chan_spike_refs]
                    # Exclude the empty blocks (dypes are uint64)
                    chan_times = [
                        times for times in chan_times if times.dtype == np.float
                    ]
                    spike_times.extend(chan_times)
        # Upsample the position data to 1 ms bins
        orig_time_stamps = np.squeeze(h5dict["t"])
        orig_freq = np.around(np.median(np.diff(orig_time_stamps)), decimals=3)
        new_time_stamps = np.around(
            np.arange(orig_time_stamps.min(), orig_time_stamps.max() + orig_freq, freq),
            decimals=3,
        )

        def upsample_matrix(data, kind="cubic"):
            """Upsample the matrix along the first dimension."""
            interp_fn = interp1d(
                orig_time_stamps, data, kind=kind, axis=0, fill_value="extrapolate"
            )
            return interp_fn(new_time_stamps)

        # Upsample the behavioral variables
        finger_pos = upsample_matrix(h5dict["finger_pos"].T)
        cursor_pos = upsample_matrix(h5dict["cursor_pos"].T)
        target_pos = upsample_matrix(h5dict["target_pos"].T, kind="previous")

        # Bin the spikes
        extra_point = np.around([new_time_stamps[-1] + freq], decimals=3)
        hist_bins = np.concatenate([new_time_stamps, extra_point])
        spikes = [np.histogram(st, bins=hist_bins)[0] for st in spike_times]
        spikes = np.stack(spikes, axis=-1)

        # Add the data to the dataframe
        data_dict = {
            "spikes": spikes,
            "finger_pos": finger_pos,
            "cursor_pos": cursor_pos,
            "target_pos": target_pos,
        }
        name_dict = {
            "cursor_pos": ["x", "y"],
            "target_pos": ["x", "y"],
        }
        self.init_data_from_dict(data_dict, freq, name_dict=name_dict)
        self.trial_info = pd.DataFrame()

    def _make_midx(self, signal_type, chan_names=None, num_channels=None):
        """Creates a pd.MultiIndex for a given signal_type."""

        if chan_names is not None:
            # If using custom channel names, make sure they match data
            assert len(chan_names) == num_channels
        elif "rates" in signal_type:
            # If merging rates, use the same names as the spikes
            chan_names = self.data.spikes.columns
        else:
            # Otherwise, generate names for the channels
            chan_names = [f"{i:04d}" for i in range(num_channels)]
        # Create the MultiIndex for this data
        midx = pd.MultiIndex.from_product(
            [[signal_type], chan_names], names=("signal_type", "channel")
        )
        return midx

    def add_continuous_data(self, cts_data, signal_type, chan_names=None):
        """Adds a continuous data field to the main DataFrame
        Parameters
        ----------
        cts_data : np.ndarray
            A numpy array whose first dimension matches the DataFrame
            at self.data.
        signal_name : str
            The label for this group of signals.
        chan_names : list of str, optional
            The channel names for this data.
        """

        logger.info(f"Adding continuous {signal_type} to the main DataFrame.")
        # Make MultiIndex columns
        midx = self._make_midx(signal_type, chan_names, cts_data.shape[1])
        # Build the DataFrame and attach it to the current dataframe
        new_data = pd.DataFrame(cts_data, index=self.data.index, columns=midx)
        self.data = pd.concat([self.data, new_data], axis=1)

    def add_trial_data(self, trial_data, signal_type, chan_names=None):
        """Adds a trialized data field to the main DataFrame."""

        logger.info(f"Adding trialized {signal_type} to the main DataFrame")
        new_data = trial_data[["clock_time", signal_type]].set_index("clock_time")
        self.data = pd.concat([self.data, new_data], axis=1)

    def load_lfads_output(self, lfads_dir, overlap, smooth_pwr=2, lf2=False):
        """Loads the LFADS output into the continuous DataFrame.
        Parameters
        ----------
        lfads_dir : str
            The path to an lfadslite model that has been posterior-sampled.
        overlap : int
            The number of overlapping bins for the LFADS rates.
        smooth_pwr : bool, optional
            Whether to linearly blend the overlapping portions of the rates,
            by default True.
        """

        # Decide whether to use lfads_tf2 or lfadslite implementation
        load_func = load_posterior_averages if lf2 else compat.load_posterior_averages
        impl = "lfads_tf2" if lf2 else "lfadslite"
        # Load the LFADS output from posterior sample files
        logger.info(f"Loading `{impl}` output from {lfads_dir}")
        lfads_output = load_func(lfads_dir, merge_tv=True)

        # Define merging and adding data back to the DataFrame
        def merge_and_add(data, name):
            cts_data = merge_chops(data, overlap, smooth_pwr=smooth_pwr)
            self.add_continuous_data(cts_data, name)

        # Merge the data and add it to the DataFrame
        merge_and_add(lfads_output.rates, "lfads_rates")
        merge_and_add(lfads_output.factors, "lfads_factors")
        merge_and_add(lfads_output.gen_inputs, "lfads_gen_inputs")

    def calculate_onset(
        self,
        field_name,
        onset_threshold,
        peak_prominence=0.1,
        peak_distance_s=0.1,
        multipeak_threshold=0.2,
    ):
        """Calculates onset for a given field by finding
        peaks and threshold crossings. Tested on speed.
        Parameters
        ----------
        field_name : str
            The field to use for onset calculation, used
            with recursive getattr on self.data.
        onset_threshold : float
            The threshold for onset as a percentage of the
            peak height.
        peak_prominence : float, optional
            Minimum prominence of peaks. Passed to
            `scipy.signal.find_peaks`, by default 0.1
        peak_distance_s : float, optional
            Minimum distance between peaks. Passed to
            `scipy.signal.find_peaks`, by default 0.1
        multipeak_threshold : float, optional
            Subsequent peaks within a trial must be no
            larger than this percentage of the first peak,
            otherwise the onset calculation fails, by default 0.2
        Returns
        -------
        pd.Series
            The times of identified peaks.
        """

        logger.info(f"Calculating {field_name} onset.")
        signal = rgetattr(self.data, field_name)
        # Find peaks - parameters from the MATLAB script
        peaks, properties = find_peaks(
            signal,
            prominence=peak_prominence,
            distance=peak_distance_s / self.bin_width,
        )
        peak_times = pd.Series(self.data.index[peaks])

        # Find the onset for each trial
        onset, onset_index = [], []
        for index, row in self.trial_info.iterrows():
            trial_start, trial_end = row["start_time"], row["end_time"]
            # Find the peaks within the trial boundaries
            trial_peaks = peak_times[
                (trial_start < peak_times) & (peak_times < trial_end)
            ]
            peak_signals = signal.loc[trial_peaks]
            # Remove trials with multiple larger peaks
            if multipeak_threshold is not None and len(trial_peaks) > 1:
                # Make sure the first peak is relatively large
                if peak_signals[0] * multipeak_threshold < peak_signals[1:].max():
                    continue
            elif len(trial_peaks) == 0:
                # If no peaks are found for this trial, skip it
                continue
            # Find the point just before speed crosses the threshold
            signal_threshold = onset_threshold * peak_signals[0]
            under_threshold = (
                signal[trial_start : trial_peaks.iloc[0]] < signal_threshold
            )
            if under_threshold.sum() > 0:
                onset.append(under_threshold[::-1].idxmax())
                onset_index.append(index)
        # Add the movement onset for each trial to the DataFrame
        onset_name = field_name.split(".")[-1] + "_onset"
        logger.info(f"`{onset_name}` field created in trial_info.")
        self.trial_info[onset_name] = pd.Series(onset, index=onset_index)

        return peak_times

    def make_trial_data(
        self,
        align_field=None,
        align_range=(None, None),
        ignored_trials=None,
        allow_overlap=False,
    ):
        """Makes a DataFrame of trialized data based on
        an alignment field.
        Parameters
        ----------
        align_field : str, optional
            The field in `trial_info` to use for alignment,
            by default None uses `trial_start` and `trial_end`.
        align_range : tuple of int, optional
            The offsets to add to the alignment field to
            calculate the alignment window, by default (None, None)
            uses `trial_start` and `trial_end`.
        ignored_trials : pd.Series, optional
            A boolean pd.Series of the same length as trial_info
            with True for the trials to ignore, by default None
            ignores no trials. This is useful for rejecting trials
            outside of the alignment process.
        allow_overlap : bool, optional
            Whether to allow overlap between trials, by default False
            truncates each trial at the end of the previous trial and
            the start of the subsequent trial.
        Returns
        -------
        pd.DataFrame
            A DataFrame containing trialized data. It has the same
            fields as the continuous `self.data` DataFrame, but
            adds `trial_id`, `trial_time`, and `align_time`. It also
            resets the index so `clock_time` is a column rather than
            an index. This DataFrame can be pivoted to plot its
            various fields across trials, aligned relative to
            `align_time`, `trial_time`, or `clock_time`.
        """

        # Allow rejection of trials by passing a boolean series
        if ignored_trials is not None:
            trial_info = self.trial_info.loc[~ignored_trials]
        else:
            trial_info = self.trial_info

        # Use the trial id's as the index
        ti = trial_info.set_index("trial_id", drop=True)

        # Find alignment points
        trial_start = ti["start_time"]
        trial_end = ti["end_time"]
        if align_field is not None:
            align_points = ti[align_field]
            align_left = align_right = align_points
        else:
            align_field = "start and end"
            align_left = trial_start
            align_right = trial_end

        # Find start and end points based on the alignment range
        start_offset, end_offset = pd.to_timedelta(align_range, unit="ms")
        if pd.isnull(start_offset):
            align_start = trial_start
        else:
            align_start = align_left + start_offset
        if pd.isnull(end_offset):
            align_end = trial_end
        else:
            align_end = align_right + end_offset

        # Store the alignment data in a dataframe
        align_data = pd.DataFrame(
            {
                "align_start": align_start,
                "align_end": align_end,
                "trial_start": trial_start,
                "align_left": align_left,
            }
        ).dropna()
        # Bound the end by the next trial / alignment start
        align_data["end_bound"] = (
            pd.concat([trial_start, align_start], axis=1).min(axis=1).shift(-1)
        )
        trial_dfs = []
        num_overlap_trials = 0
        for trial_id, row in align_data.iterrows():
            # Handle overlap with the start of the next trial
            endpoint = row.align_end
            if not pd.isnull(row.end_bound) and row.align_end > row.end_bound:

                num_overlap_trials += 1
                if not allow_overlap:
                    endpoint = row.end_bound
            # Take a slice of the continuous data
            trial_df = (
                self.data.loc[row.align_start : endpoint]
                .iloc[:-1]  # Omit the last entry
                .reset_index()  # Move clock_time to column
            )
            # Add trial identifiers
            trial_df["trial_id"] = trial_id
            # Add times to orient the trials
            clock_time = trial_df["clock_time"]
            trial_df["trial_time"] = clock_time - row.trial_start
            trial_df["align_time"] = clock_time - row.align_left
            trial_dfs.append(trial_df)
        # Summarize alignment
        logger.info(
            f"Aligned {len(trial_dfs)} trials to "
            f"{align_field} with offset of {align_range} ms."
        )
        # Report any overlapping trials to the user.
        if num_overlap_trials > 0:
            if allow_overlap:
                logger.warning(f"Allowed {num_overlap_trials} overlapping trials.")
            else:
                logger.warning(
                    f"Shortened {num_overlap_trials} trials to prevent overlap."
                )
        # Combine all trials into one DataFrame
        trial_data = pd.concat(trial_dfs).reset_index(drop=True)
        # Sanity check to make sure there are no duplicated `clock_time`'s
        if not allow_overlap:
            assert (
                trial_data.clock_time.duplicated().sum() == 0
            ), "Duplicated points still found. Double-check overlap code."
        # Make sure NaN's caused by adding trialized data to self.data are ignored
        nans_found = trial_data.isnull().sum().max()
        if nans_found > 0:
            pct_nan = (nans_found / len(trial_data)) * 100
            logger.warning(
                f"NaNs found in `self.data`. Dropping {pct_nan:.2f}% "
                "of points to remove NaNs from `trial_data`."
            )
            trial_data = trial_data.dropna()

        return trial_data
