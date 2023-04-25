# A generalized dataset for working with BRAND exported data in NWB format. Adapted from
# https://github.com/neurallatents/nlb_tools/blob/main/nlb_tools/nwb_interface.py
# Original author: Felix Pei
# and
# https://github.com/snel-repo/snel-toolkit/blob/nlb_tools/snel_toolkit/datasets/nlb.py
# Original author: Andrew Sedler

import json
import logging
import os
from glob import glob

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO, ProcessingModule, TimeSeries
from pynwb.core import MultiContainerInterface

from .base import BaseDataset

logger = logging.getLogger(__name__)


class BRANDDatasetV2(BaseDataset):
    """A class for BRAND data from NWB files."""

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
        self.name = os.path.splitext(os.path.basename(fpath))[0]
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
        # (this is untested, haven't deviated from original NWB code)
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
            dataframes, trial_info, descriptions, bin_widths = self.load(
                filenames[0], split_heldout=split_heldout
            )
            self.dataframes = dataframes
            self.trial_info = trial_info
            self.descriptions = descriptions
            self.bin_widths = bin_widths

            default_rate = max(list(key for key in self.dataframes.keys()))
            self.data = self.dataframes[default_rate]
            self.bin_width = self.bin_widths[default_rate]

            logger.info(
                f"self.data is pointing to dataframe sampled at {default_rate} Hz. "
                f"Pointer can be changed by calling: 'select_data(rate)'"
            )

    def select_data(self, rate):
        # TODO: add error if key not in dict
        logger.info(f"self.data set to to dataframe sampled at {rate} Hz.")
        self.data = self.dataframes[rate]
        self.bin_width = self.bin_widths[rate]

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

        # Load trial info
        trial_info = (
            nwbfile.trials.to_dataframe()
            .reset_index()
            .rename({"id": "trial_id", "stop_time": "end_time"}, axis=1)
        )

        # Load units
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

        # Set channel mask
        self.cols_to_rm = []
        if 'mask' in nwbfile.electrodes.to_dataframe():
            ch_mask_bool = nwbfile.electrodes.to_dataframe()['mask'].values
            if ch_mask_bool.dtype is np.dtype(np.bool):
                self.ch_mask = np.argwhere(ch_mask_bool).squeeze()
                self.ch_unmask = np.argwhere(np.logical_not(ch_mask_bool))
                self.ch_unmask = self.ch_unmask.squeeze()
                print(('Remove / Zero the following channels from '
                       "'binned_spikes_samples':\n"
                       f'{str(self.ch_unmask.reshape(-1).tolist())}'))
                for c in self.ch_unmask:
                    self.cols_to_rm.append(('spikes', c.item()))

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
            # TODO: test if the following works with a single field signal
            columns = [ts.name]
            if ts.data.shape[1] > 1 and len(columns) <= 1:
                columns = []
                for i in range(ts.data.shape[1]):
                    columns.append(i)
            df = pd.DataFrame(
                ts.data[()],
                index=pd.to_timedelta(index.round(3), unit="s"),
                columns=columns,
            )
            return df

        def find_timeseries(nwbobj):
            """Recursively searches the NWB file for time series data"""
            ts_dict = {}
            rates = {}
            for child in nwbobj.children:
                if isinstance(child, TimeSeries):
                    if child.name in skip_fields:
                        continue
                    try:
                        comment_dict = json.loads(child.comments)
                        if "rate" in comment_dict:
                            ts_dict[child.name] = make_df(child)
                            descriptions[child.name] = child.description
                            rates[child.name] = np.round(float(comment_dict["rate"]), 3)
                        else:
                            logger.info(
                                f"TimeSeries {child.name} ignored because of no rate "
                                f"defined in NWB file."
                            )
                    except ValueError:
                        logger.info(
                            f"TimeSeries {child.name} ignored because of no rate "
                            f"defined in NWB file."
                        )
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
                            try:
                                comment_dict = json.loads(field.comments)
                                if "rate" in comment_dict:
                                    ts_dfs.append(make_df(field))
                                    rate_f = np.round(float(comment_dict["rate"]), 3)
                                else:
                                    logger.info(
                                        f"TimeSeries {name}.{field} ignored because "
                                        f"of no rate defined in NWB file."
                                    )
                            except ValueError:
                                logger.info(
                                    f"TimeSeries {name}.{field} ignored because "
                                    f"of no rate defined in NWB file."
                                )
                    if len(ts_dfs) > 0:
                        ts_dict[name] = pd.concat(ts_dfs, axis=1)
                        descriptions[name] = field.description
                        rates[name] = rate_f
            return ts_dict, rates

        # Create a dictionary containing DataFrames for all time series
        data_dict, rates_dict = find_timeseries(nwbfile)

        dataframes = {}
        bin_widths = {}

        # Extract unique values from bin_width_dict
        rate_list = list(set(rates_dict.values()))

        # Add 1000 Hz if not present (for spiking data)
        if has_units and 1000.0 not in rate_list:
            rate_list.append(1000.0)

        for rate in rate_list:

            # Get keys for TimeSeries in dict corresponding to current rate
            rate_keys = [key for key, value in rates_dict.items() if value == rate]

            # Extract dict only with keys corresponding to bin width
            rate_data_dict = {
                key: value for key, value in data_dict.items() if key in rate_keys
            }

            bin_width = np.round(1 / rate, 3)
            bin_widths[rate] = bin_width

            for field in rate_data_dict:
                # field_min_ts = np.min(
                #     np.diff(rate_data_dict[field].index.total_seconds().values).round(3)
                # )
                field_ts = np.diff(
                    rate_data_dict[field].index.total_seconds().values
                ).round(3)
                if np.any(field_ts != bin_width):
                    mismatched_ts = sum(field_ts != bin_width)
                    logger.warning(
                        f"{mismatched_ts} timestamp intervals in TimeSeries {field} "
                        f"are different than bin width ({bin_width})."
                    )

            # Only load units into the 1ms/1kHz Dataframe
            if rate == 1000 and has_units:

                if "obs_intervals" in units.columns:
                    # Find min and max unit ranges
                    start_time = min(
                        [
                            units.obs_intervals[i].min()
                            for i in range(len(units.obs_intervals))
                        ]
                    ).round(3)
                    end_time = max(
                        [
                            units.obs_intervals[i].max()
                            for i in range(len(units.obs_intervals))
                        ]
                    ).round(3)
                else:
                    logger.warning(
                        "'obs_intervals' field not present for units data. Inferring "
                        "min/max times from spike times."
                    )
                    # Find min and max spike times
                    start_time = min(
                        [
                            min(units.spike_times[i])
                            for i in range(len(units.spike_times))
                        ]
                    ).round(3)
                    end_time = max(
                        [
                            max(units.spike_times[i])
                            for i in range(len(units.spike_times))
                        ]
                    ).round(3)

                # Checking if this is not needed anymore
                # if end_time < trial_info["end_time"].iloc[-1]:
                #     print("obs_interval ends before trial end")
                #     end_time = trial_info["end_time"].iloc[-1].round(3)

                timestamps = (
                    np.arange(start_time, end_time + bin_width, bin_width)
                ).round(3)
                timestamps_td = pd.to_timedelta(timestamps, unit="s")

                # Check that all timeseries match with calculated timestamps
                # Removing the following check for now
                # for key, val in list(rate_data_dict.items()):
                #     if not np.all(
                #         np.isin(np.round(val.index.total_seconds(), 3), timestamps)
                #     ):
                #         logger.warning(f"Dropping {key} due to timestamp mismatch.")
                #         rate_data_dict.pop(key)

                def make_mask(obs_intervals):
                    """
                    Creates bool mask to indicate when spk data not in obs_intervals
                    """
                    mask = np.full((timestamps.shape[0], obs_intervals.shape[0]), True)
                    for chan_id, chan in enumerate(obs_intervals):
                        start = chan.squeeze()[0]
                        end = chan.squeeze()[-1]
                        start_idx = np.ceil(
                            round((start - timestamps[0]) * rate, 3)
                        ).astype(int)
                        end_idx = np.floor(
                            round((end - timestamps[0]) * rate, 3)
                        ).astype(int)
                        mask[start_idx : end_idx + 1, chan_id] = False
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
                            .round(3)
                            .astype(int),
                            return_counts=True,
                        )
                        spike_arr[spike_idx, idx] = spike_cnt

                    # Replace invalid intervals in spike recordings with NaNs
                    if "obs_intervals" in units.columns:
                        neur_mask = make_mask(units[mask].obs_intervals)
                        if np.any(spike_arr[neur_mask]):
                            logger.warning("Spikes found outside of observed interval.")
                        spike_arr[neur_mask] = np.nan
                    # Create DataFrames with spike arrays
                    rate_data_dict[name] = pd.DataFrame(
                        spike_arr, index=timestamps_td, columns=units[mask].index
                    ).astype("float64", copy=False)

            # Create MultiIndex column names
            data_list = []
            for key, val in rate_data_dict.items():
                chan_names = None if type(val.columns) == pd.RangeIndex else val.columns
                val.columns = self._make_midx(
                    key, chan_names=chan_names, num_channels=val.shape[1]
                )
                data_list.append(val)

            # Assign time-varying data to `self.data`
            dataframes[rate] = pd.concat(data_list, axis=1)
            dataframes[rate].index.name = "clock_time"
            dataframes[rate].sort_index(axis=1, inplace=True)

        logger.info(
            f"Created separate dataframes for data at following rates: "
            f"{list(key for key in dataframes.keys())}"
        )

        # Convert time fields in trial info to timedelta
        # and assign to `self.trial_info`
        def to_td(x):
            if x.name.endswith("_time"):
                return pd.to_timedelta(x, unit="s")
            else:
                return x

        trial_info = trial_info.apply(to_td, axis=0)

        io.close()

        return dataframes, trial_info, descriptions, bin_widths

    def keep_fields(self, fields):
        """Remove everything other than fields from `self.data`."""

        fields_current = list(self.data.columns.get_level_values(0).unique())
        fields_remove = list(set(fields_current) - set(fields))

        logger.info(f"Removing {fields_remove} field(s) from dataset.")
        self.data.drop(columns=fields_remove, inplace=True)

    def remove_fields(self, fields):
        """Remove fields from `self.data`."""

        logger.info(f"Removing {fields} field(s) from dataset.")
        self.data.drop(columns=fields, inplace=True)


class BRANDDataset(BaseDataset):
    """A class for BRAND data from NWB files.
    For NWB Datasets from T11 Session 01-15 and before
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

        # Set channel mask
        if 'mask' in nwbfile.electrodes.to_dataframe():
            ch_mask_bool = nwbfile.electrodes.to_dataframe()['mask'].values
            if ch_mask_bool.dtype is np.dtype(np.bool):
                self.ch_mask = np.argwhere(ch_mask_bool).squeeze()
                self.ch_unmask = np.argwhere(np.logical_not(ch_mask_bool))
                self.ch_unmask = self.ch_unmask.squeeze()
                print(('Remove / Zero the following channels from '
                       "'binned_spikes_samples':\n"
                       f'{str(self.ch_unmask.reshape(-1).tolist())}'))
                self.cols_to_rm = []
                for c in self.ch_unmask:
                    self.cols_to_rm.append(('spikes', c.item()))
        else:
            self.cols_to_rm = []

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
            if ts.data.shape[1] > 1 and len(columns) <= 1:
                columns = []
                for i in range(ts.data.shape[1]):
                    columns.append(i)
            df = pd.DataFrame(
                ts.data[()],
                index=pd.to_timedelta(index.round(6), unit="s"),
                columns=columns,
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

        # Find min and max timestamps, and highest frequency sampling rate
        start_time = min(
            data_dict[field].index.total_seconds().values[0] for field in data_dict
        )
        end_time = max(
            data_dict[field].index.total_seconds().values[-1] for field in data_dict
        )
        bin_width = min(
            np.median(np.diff(data_dict[field].index.total_seconds().values).round(3))
            for field in data_dict
        )
        for field in data_dict:
            field_med_ts = np.median(
                np.diff(data_dict[field].index.total_seconds().values).round(3)
            )
            field_min_ts = np.min(
                np.diff(data_dict[field].index.total_seconds().values).round(3)
            )
            if field_min_ts < field_med_ts:
                logger.warning(
                    f"Minimum difference in timestamps ({field_min_ts}) is smaller "
                    f"than median ({field_med_ts}) for TimeSeries {field}."
                )
        bin_width = round(bin_width, 3)  # round to nearest millisecond
        rate = round(1.0 / bin_width, 2)  # in Hz

        if has_units:

            if end_time < trial_info["end_time"].iloc[-1]:
                print("obs_interval ends before trial end")  # TO REMOVE
                end_time = round(trial_info["end_time"].iloc[-1] * rate) * (
                    bin_width * 1000
                )
            timestamps = (np.arange(start_time, end_time + bin_width, bin_width)).round(
                6
            )
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

    def keep_fields(self, fields):
        """Remove everything other than fields from `self.data`."""

        fields_current = list(self.data.columns.get_level_values(0).unique())
        fields_remove = list(set(fields_current) - set(fields))

        logger.info(f"Removing {fields_remove} field(s) from dataset.")
        self.data.drop(columns=fields_remove, inplace=True)

    def remove_fields(self, fields):
        """Remove fields from `self.data`."""

        logger.info(f"Removing {fields} field(s) from dataset.")
        self.data.drop(columns=fields, inplace=True)
