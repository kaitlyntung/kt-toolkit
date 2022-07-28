# A generalized dataset for working with data in NWB format. Adapted from
# https://github.com/neurallatents/nlb_tools/blob/main/nlb_tools/nwb_interface.py
# Original author: Felix Pei
# and
# https://github.com/snel-repo/snel-toolkit/blob/nlb_tools/snel_toolkit/datasets/nlb.py
# Original author: Andrew Sedler


import logging
import os
from glob import glob

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO, ProcessingModule, TimeSeries
from pynwb.core import MultiContainerInterface

from .base import BaseDataset

logger = logging.getLogger(__name__)

DATA_PATHS = {
    "mc_maze": "/snel/share/data/dandi/000128/sub-Jenkins",
    "mc_rtt": "/snel/share/data/dandi/000129/sub-Indy",
    "area2_bump": "/snel/share/data/dandi/000127/sub-Han",
    "dmfc_rsg": "/snel/share/data/dandi/000130/sub-Haydn",
    "mc_maze_large": "/snel/share/data/dandi/000138/sub-Jenkins",
    "mc_maze_medium": "/snel/share/data/dandi/000139/sub-Jenkins",
    "mc_maze_small": "/snel/share/data/dandi/000140/sub-Jenkins",
}


class BRANDDataset(BaseDataset):
    """A class for loading/preprocessing data from NWB files. Can also be used for
    loading the NLB competition datasets.
    """

    def __init__(self, fpath, prefix="", split_heldout=False, skip_fields=[]):
        """Initializes an NWBDataset, loading data from
        the indicated file(s)
        Parameters
        ----------
        fpath : str
            Either the path to an NWB file or to a directory
            containing NWB files
        prefix : str, optional
            A pattern used to filter the NWB files in directory
            by name. By default, prefix='' loads all .nwb files in
            the directory. Please refer to documentation for
            the `glob` module for more details:
            https://docs.python.org/3/library/glob.html
        split_heldout : bool, optional
            (specific to NLB datasets) Whether to load heldin units
            and heldout units to separate fields or not, by default
            False
        skip_fields : list, optional
            List of field names to skip during loading,
            which may be useful if memory is an issue.
            Field names must match the names automatically
            assigned in the loading process. Spiking data
            can not be skipped. Field names in the list
            that are not found in the dataset are
            ignored
        """
        fpath = os.path.expanduser(fpath)
        self.fpath = fpath
        self.prefix = prefix
        # Check if file/directory exists
        if not os.path.exists(fpath):
            raise FileNotFoundError("Specified file or directory not found")
        # If directory, look for files with matching prefix
        if os.path.isdir(fpath):
            filenames = sorted(glob(os.path.join(fpath, prefix + "*.nwb")))
        else:
            filenames = [fpath]
        # If no files found
        if len(filenames) == 0:
            raise FileNotFoundError(
                f"No matching files with prefix {prefix} found in directory {fpath}"
            )
        # If multiple files found
        elif len(filenames) > 1:
            loaded = [
                self.load(fname, split_heldout=split_heldout, skip_fields=skip_fields)
                for fname in filenames
            ]
            datas, trial_infos, descriptions, bin_widths = [
                list(out) for out in zip(*loaded)
            ]
            assert np.all(
                np.array(bin_widths) == bin_widths[0]
            ), "Bin widths of loaded datasets must be the same"

            # Shift loaded files to stack them into continuous array
            def trial_shift(x, shift_ms, trial_offset):
                if x.name.endswith("_time"):
                    return x + pd.to_timedelta(shift_ms, unit="ms")
                elif x.name == "trial_id":
                    return x + trial_offset
                else:
                    return x

            # Loop through files, shifting continuous data
            past_end = datas[0].index[-1].total_seconds() + round(50 * bin_widths[0], 4)
            descriptions_full = descriptions[0]
            tcount = len(trial_infos[0])
            for i in range(1, len(datas)):
                block_start_ms = np.ceil(past_end * 10) * 100
                datas[i] = datas[i].shift(block_start_ms, freq="ms")
                trial_infos[i] = trial_infos[i].apply(
                    trial_shift, shift_ms=block_start_ms, trial_offset=tcount
                )
                descriptions_full.update(descriptions[i])
                past_end = datas[i].index[-1].total_seconds() + round(
                    50 * bin_widths[i], 4
                )
                tcount += len(trial_infos[i])
            # Stack data and reindex to continuous
            self.data = pd.concat(datas, axis=0, join="outer")
            self.trial_info = pd.concat(trial_infos, axis=0, join="outer").reset_index(
                drop=True
            )
            self.descriptions = descriptions_full
            self.bin_width = bin_widths[0]
            new_index = pd.to_timedelta(
                (
                    np.arange(
                        round(self.data.index[-1].total_seconds() * 1 / self.bin_width)
                        + 1
                    )
                    * (self.bin_width * 1000)
                ).round(4),
                unit="s",
            )
            self.data = self.data.reindex(new_index)
            self.data.index.name = "clock_time"
        # If single file found
        else:
            data, trial_info, descriptions, bin_width = self.load(
                filenames[0], split_heldout=split_heldout
            )
            self.data = data
            self.trial_info = trial_info
            self.descriptions = descriptions
            self.bin_width = bin_width

    def load(self, fpath, split_heldout=False, skip_fields=[]):
        """Loads data from an NWB file into two dataframes,
        one for trial info and one for time-varying data

        Parameters
        ----------
        fpath : str
            Path to the NWB file
        split_heldout : bool, optional
            Whether to load heldin units and heldout units
            to separate fields or not, by default True
        skip_fields : list, optional
            List of field names to skip during loading,
            which may be useful if memory is an issue.
            Field names must match the names automatically
            assigned in the loading process. Spiking data
            can not be skipped. Field names in the list
            that are not found in the dataset are
            ignored

        Returns
        -------
        tuple
            Tuple containing a pd.DataFrame of continuous loaded
            data, a pd.DataFrame with trial metadata, a dict
            with descriptions of fields in the DataFrames, and
            the bin width of the loaded data in ms
        """
        logger.info(f"Loading {fpath}")

        # Open NWB file
        io = NWBHDF5IO(fpath, "r")
        nwbfile = io.read()
        # Load trial info and units
        trial_info = (
            nwbfile.trials.to_dataframe()
            .reset_index()
            .rename({"id": "trial_id", "stop_time": "end_time"}, axis=1)
        )

        if nwbfile.units is not None:
            has_units = True
            units = nwbfile.units.to_dataframe()
            unit_info = nwbfile.units.electrodes.to_dataframe()
            unit_info = unit_info.drop(
                columns=unit_info.columns.difference(["group_name", "location"])
            )
            self.unit_info = unit_info
        else:
            has_units = False
            acq_key = list(nwbfile.acquisition.keys())[0]
            name_key = list(nwbfile.acquisition[acq_key].time_series.keys())[0]
            dt_s = np.diff(nwbfile.acquisition[acq_key][name_key].timestamps).round(4)
            print("line 203")
            bin_width = np.unique(dt_s)[0]

        # Load descriptions of trial info fields
        descriptions = {}
        for name, info in zip(nwbfile.trials.colnames, nwbfile.trials.columns):
            descriptions[name] = info.description

        # Find all timeseries
        def make_df(ts):
            """Converts TimeSeries into pandas DataFrame"""
            if ts.timestamps is not None:
                index = ts.timestamps[()]
            else:
                index = np.arange(ts.data.shape[0]) / ts.rate + ts.starting_time
            # MR: Don't like that it assumes comments to have the column names
            columns = (
                ts.comments.split("[")[-1].split("]")[0].split(",")
                if "columns=" in ts.comments
                else None
            )
            if columns is None:
                columns = [ts.name]
            if len(ts.data.shape) > 1:
                base_column_name = columns[0]
                columns = []
                for i in range(ts.data.shape[1]):
                    columns.append(base_column_name + "_" + str(i))

            df = pd.DataFrame(
                ts.data[()], index=pd.to_timedelta(index, unit="s"), columns=columns
            )

            return df

        def find_timeseries(nwbobj):
            """Recursively searches the NWB file for time series data"""
            ts_dict = {}
            for child in nwbobj.children:
                if isinstance(child, TimeSeries):
                    if child.name in skip_fields:
                        continue
                    ts_dict[child.name] = make_df(child)
                    descriptions[child.name] = child.description
                elif isinstance(child, ProcessingModule):
                    pm_dict = find_timeseries(child)
                    ts_dict.update(pm_dict)
                elif isinstance(child, MultiContainerInterface):
                    name = child.name
                    ts_dfs = []
                    for field in child.children:
                        if isinstance(field, TimeSeries):
                            if name in skip_fields:
                                continue
                            ts_dfs.append(make_df(field))

                    ts_dict[name] = pd.concat(ts_dfs, axis=1)
                    descriptions[name] = field.description
            return ts_dict

        # Create a dictionary containing DataFrames for all time series
        data_dict = find_timeseries(nwbfile)

        if has_units:
            # Calculate data index
            start_time = 0.0
            bin_width = 0.001  # in sec, this will be the case for all provided datasets
            rate = round(1.0 / bin_width, 2)  # in Hz
            # Use obs_intervals, or last trial to determine data end
            # ^ what if we don't have obs_intervals?
            if hasattr(units, "obs_intervals"):
                end_time = round(
                    max(units.obs_intervals.apply(lambda x: x[-1][-1])) * rate
                ) * (bin_width * 1000)
            else:
                end_time = (
                    max(units.spike_times.apply(lambda x: x[-1]))
                    * rate
                    * (bin_width * 1000)
                    + 1
                )

            if end_time < trial_info["end_time"].iloc[-1]:
                print("obs_interval ends before trial end")  # TO REMOVE
                end_time = round(trial_info["end_time"].iloc[-1] * rate) * (
                    bin_width * 1000
                )
            timestamps = (
                np.arange(start_time, end_time, (bin_width * 1000)) / 1000
            ).round(6)
            timestamps_td = pd.to_timedelta(timestamps, unit="s")

            # Check that all timeseries match with calculated timestamps
            for key, val in list(data_dict.items()):

                if not np.all(
                    np.isin(np.round(val.index.total_seconds(), 6), timestamps)
                ):
                    logger.warning(f"Dropping {key} due to timestamp mismatch.")
                    data_dict.pop(key)

            def make_mask(obs_intervals):
                """Creates bool mask to indicate when spk data not in obs_intervals"""
                mask = np.full(timestamps.shape, True)
                for start, end in obs_intervals:
                    start_idx = np.ceil(
                        round((start - timestamps[0]) * rate, 6)
                    ).astype(int)
                    end_idx = np.floor(round((end - timestamps[0]) * rate, 6)).astype(
                        int
                    )
                    mask[start_idx:end_idx] = False
                return mask

            # Prepare variables for spike binning
            masks = (
                [(~units.heldout).to_numpy(), units.heldout.to_numpy()]
                if split_heldout
                else [np.full(len(units), True)]
            )

            for mask, name in zip(masks, ["spikes", "heldout_spikes"]):
                # Check if there are any units
                if not np.any(mask):
                    continue

                # Allocate array to fill with spikes
                spike_arr = np.full(
                    (len(timestamps), np.sum(mask)), 0.0, dtype="float16"
                )

                # Bin spikes using dec. trunc and np.unique
                # - faster than np.histogram with same results

                for idx, (_, unit) in enumerate(units[mask].iterrows()):
                    spike_idx, spike_cnt = np.unique(
                        ((unit.spike_times - timestamps[0]) * rate)
                        .round(6)
                        .astype(int),
                        return_counts=True,
                    )
                    spike_arr[spike_idx, idx] = spike_cnt

                # Replace invalid intervals in spike recordings with NaNs
                if "obs_intervals" in units.columns:
                    neur_mask = make_mask(units[mask].iloc[0].obs_intervals)
                    if np.any(spike_arr[neur_mask]):
                        logger.warning("Spikes found outside of observed interval.")
                    spike_arr[neur_mask] = np.nan
                # Create DataFrames with spike arrays
                data_dict[name] = pd.DataFrame(
                    spike_arr, index=timestamps_td, columns=units[mask].index
                ).astype("float64", copy=False)

        # Create MultiIndex column names
        data_list = []
        for key, val in data_dict.items():
            chan_names = None if type(val.columns) == pd.RangeIndex else val.columns
            val.columns = self._make_midx(
                key, chan_names=chan_names, num_channels=val.shape[1]
            )
            data_list.append(val)
        # Assign time-varying data to `self.data`
        data = pd.concat(data_list, axis=1)
        data.index.name = "clock_time"
        data.sort_index(axis=1, inplace=True)

        # Convert time fields in trial info to timedelta
        # and assign to `self.trial_info`
        def to_td(x):
            if x.name.endswith("_time"):
                return pd.to_timedelta(x, unit="s")
            else:
                return x

        trial_info = trial_info.apply(to_td, axis=0)

        io.close()

        return data, trial_info, descriptions, bin_width
