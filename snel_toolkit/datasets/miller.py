import logging
import os
import pathlib
from os import path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat

from .base import BaseDataset

# Set up standard per-module logging
logger = logging.getLogger(__name__)

# Example files that can be used for quick tests
AREA2_EXAMPLE_FILE = "/snel/share/share/data/raeed_s1/Han_20171116_COactpas_TD-2.mat"
XDS_EXAMPLE_FILE = (
    "/snel/share/share/data/NW_Emory_Data_Sharing/"
    "Jango/DS18(Jango_2015)/xds_with_raw_EMG/Jango_20150730_001.mat"
)


class Area2Dataset(BaseDataset):
    def load(self, fpath, include_multiunits=False, joint_angles=False):
        """Method for loading data from Chowdhury's Area 2 dataset.
        Currently has only been used with Han_20171116_COactpas_TD-2.mat
        Parameters
        ----------
        fpath: str
            The path to the `.mat` file of interest.
        include_multiunits: bool, optional
            Whether to load the multiunit activity in addition to the
            single unit activity, by default False.
        """

        logger.info(f"Loading Area2 dataset from {fpath}")

        # Account for known naming discrepancies
        tgt_dir_key = "tgtDir" if joint_angles else "targetDir"

        # # Extra motion-tracking fields
        # [
        #     'date_time',
        #     'monkey',
        #     'task',
        #     'emg',
        #     'emg_names',
        #     'marker_names',
        #     'markers',
        #     'muscle_len',
        #     'muscle_names',
        #     'muscle_vel',
        # ]

        # Create mapping from dataset field names to standardized RDS
        trial_info_name_map = {
            "trialID": "trial_id",
            "result": "result",
            "ctrHold": "ctr_hold",
            tgt_dir_key: "target_dir",
            "ctrHoldBump": "ctr_hold_bump",
            "bumpDir": "bump_dir",
            "idx_startTime": "start_time",
            "idx_endTime": "end_time",
            "idx_tgtOnTime": "target_on",
            "idx_goCueTime": "go_cue",
            "idx_bumpTime": "bump_time",
        }
        data_name_map = {
            "pos": "kin_p",
            "vel": "kin_v",
            "force": "force",
            "S1_spikes": "spikes",
        }
        if joint_angles:
            data_name_map.update(
                {
                    "joint_ang": "joint_ang",
                    "joint_vel": "joint_vel",
                }
            )

        with h5py.File(fpath, "r") as h5file:
            ds = h5file["trial_data"]
            bin_width = ds["bin_size"][0][0]
            # Collect channel names
            elecs, units = ds["S1_unit_guide"][:, :]
            # Use either single units or the multiunits
            keep_units = units != -1 if include_multiunits else units != 0
            elecs = elecs[keep_units]
            units = units[keep_units]
            # Collect the trial information
            trial_info_dict = {}
            for key, ti_name in trial_info_name_map.items():
                value = ds[key][:, 0]
                if "idx" in key:
                    # Convert from ms to timedeltas
                    value = pd.to_timedelta(value, unit="ms")
                trial_info_dict[ti_name] = value
            # Collect the data
            data_dict = {}
            for key, data_name in data_name_map.items():
                data = ds[key][:, :].T
                if key == "S1_spikes":
                    # Drop the unused units
                    data = data[:, keep_units]
                data_dict[data_name] = data
            # Get joint names
            if joint_angles:
                # Pull integers representing joint name characters
                int_arrays = [h5file[ref][()] for ref in ds["joint_names"][:, 0]]

                # Convert integers to corresponding characters
                def make_chan_name(array):
                    return "".join([chr(num) for num in array])

                joint_names = [make_chan_name(array) for array in int_arrays]

        # Collect info held by all trials
        max_len = max([len(v) for v in trial_info_dict.values()])
        trial_info_full = {
            k: v for k, v in trial_info_dict.items() if len(v) == max_len
        }
        trial_info = pd.DataFrame(trial_info_full)

        # Fill in info only held by some trials
        trial_info_partial = {
            k: v for k, v in trial_info_dict.items() if len(v) != max_len
        }
        for key, times in trial_info_partial.items():

            def fill_nans(x):
                """Finds which trials each time corresponds to."""
                between = (x.start_time < times) & (times < x.end_time)
                n_between = between.sum()
                if n_between == 1:
                    return times[between][0]
                elif n_between == 0:
                    return np.nan
                else:
                    raise AssertionError(
                        f"{key} takes place during overlapping trials."
                    )

            # Get the appropriate nan-filled series and add it to trial_info
            filled_times = trial_info.apply(fill_nans, axis=1)
            trial_info[key] = filled_times

        # Convert the result representation to a character
        trial_info["result"] = trial_info.result.apply(chr)
        # Use integers for the trial_id
        trial_info["trial_id"] = trial_info.trial_id.astype(int)
        # Use booleans for whether a bump occurred
        trial_info["ctr_hold_bump"] = trial_info.ctr_hold_bump.astype(bool)
        # Use the trial_id as the index of trial_info
        trial_info = trial_info.set_index("trial_id", drop=True)

        # Name each channel
        chan_names = [f"elec{e:03}unit{u:02}" for e, u in zip(elecs, units)]
        name_dict = {
            "kin_p": ["x", "y"],
            "kin_v": ["x", "y"],
            "force": ["x", "y", "z", "x_m", "y_m", "z_m"],
            "spikes": chan_names,
        }
        if joint_angles:
            name_dict.update(
                {
                    "joint_ang": joint_names,
                    "joint_vel": joint_names,
                }
            )
        # Build the main dataframe
        self.init_data_from_dict(
            data_dict, bin_width=bin_width, name_dict=name_dict, trial_info=trial_info
        )

    def load_merged_lfads_output(self, fpath):
        """Loads Reza's merged LFADS rates from `.mat` file. This function
        is specialized for a unique file format.
        Parameters
        ----------
        fpath: str
            The path to the `.mat` file of interest.
        """

        data = loadmat(fpath)
        self.add_continuous_data(data["lfads_rates"].T, "lfads_rates")
        self.add_continuous_data(data["lfads_inputs"].T, "lfads_gen_inputs")

    def get_move_onset(
        self,
        move_field="speed",
        start_field="go_cue",
        end_field="end_time",
        method="peak",
        min_ds=1.9,
        s_thresh=10,
        peak_offset=0,  # ms
        start_offset=0,  # ms
        onset_name="movement_on",
        peak_name="peak_speed",
        peak_divisor=2,
        ignored_trials=None,
    ):
        """Calculates movement onset, inspired by the 'peak' method in
        Matt Perich's MATLAB implementation here:
        https://github.com/mattperich/TrialData/blob/master/Tools/getMoveOnsetAndPeak.m

        Parameters
        ----------
        start_field: str
            The field name of the start of the window to consider.
        end_field: str
            The field name of the end of the window to consider.
        min_ds: float
            The minimum diff (speed) to find movement onset, by default 1.9
        s_thresh: float
            Speed threshold (secondary method if first fails)
        peak_offset: float
            The number of ms after start_field to find max speed
        start_offset: float,
            The number of ms after start_field to find movement onset
        onset_name: str
            The name of the new movement onset trial_info field.
        peak_name: str
            The name of the new peak trial_info field.
        peak_divisor: float
            The number to divide the peak by to find the threshold.
        ignored_trials: pd.Series
            The trials to ignore for this calculation.
        """
        # Ignore trials that don't have the required fields
        ti = self.trial_info
        if ignored_trials is None:
            ignored_trials == pd.Series(False, index=ti.index)
        ignored_trials = (
            ti[start_field].isnull() | ti[end_field].isnull() | ignored_trials
        )
        # Make trials to get rid of unneeded data
        trial_data = self.make_trial_data(ignored_trials=ignored_trials)
        # Keep a copy of the grouped movement data
        grouped_md = trial_data[["trial_id", move_field]].groupby("trial_id")
        # Convert offsets to timedeltas
        start_offset_td = pd.to_timedelta(start_offset, unit="ms")
        peak_offset_td = pd.to_timedelta(peak_offset, unit="ms")
        # Format start and end times for each trial to match trial_data
        td = trial_data
        ti_broadcast = ti.loc[td.trial_id]
        ti_broadcast = ti_broadcast.set_index(td.index)
        window_start = getattr(ti_broadcast, start_field)
        window_end = getattr(ti_broadcast, end_field)
        move_start = window_start + start_offset_td
        peak_start = window_start + peak_offset_td
        # Check whether the clock times are valid
        valid_peak = (peak_start < td.clock_time) & (td.clock_time < window_end)
        valid_move = (move_start < td.clock_time) & (td.clock_time < window_end)
        # Calculate peak accelerations
        dm = grouped_md[move_field].diff()
        ddm = dm.diff()
        peaks = (ddm > 0) & (ddm.shift(-1) < 0)
        # Find peak candidates
        accel_peaks = peaks & valid_peak & (dm > min_ds) & valid_move
        # Group the peaks by trial
        grouped_peaks = accel_peaks.groupby(td.trial_id)
        # Find the trials with no peaks
        no_peaks = grouped_peaks.sum() == 0
        # no_peak_trials = no_peaks[no_peaks].index.values
        # Find the first peak in each trial
        first_peak_ixs = grouped_peaks.idxmax()[~no_peaks]
        # Find the movement onset index in each trial
        thresholds = dm[first_peak_ixs] / peak_divisor
        thresholds.index = first_peak_ixs.index
        td_subset = td[td.trial_id.isin(thresholds.index)]
        thresh_broadcast = thresholds.loc[td_subset.trial_id]
        thresh_broadcast.index = td_subset.index
        firstpeak_broadcast = first_peak_ixs.loc[td_subset.trial_id]
        firstpeak_broadcast.index = td_subset.index
        # Find the last point under-threshold
        index = td_subset.index
        under_thresh = (
            (dm[index] < thresh_broadcast)  # Find below-thresh acceleration
            & valid_move[index]  # Only allow during the valid move period
            & (index < firstpeak_broadcast)  # Only allow before the peak
        )
        # Flip ordering to find the last True value
        peak_onset_ixs = under_thresh[::-1].groupby(td_subset.trial_id[::-1]).idxmax()
        # Use a simple movement threshold as backup
        onset_thresh = ((trial_data[move_field] > s_thresh) & valid_move).groupby(
            trial_data.trial_id
        )
        onset_ixs = onset_thresh.idxmax()
        no_cross = onset_thresh.sum() == 0
        onset_ixs[no_cross] = np.nan
        n_peak = len(peak_onset_ixs)
        n_fail = no_cross.sum()
        n_simple = len(onset_ixs) - n_peak - n_fail
        logger.info(
            f"Move onset calculated by peak on {n_peak} trials "
            f"and by simple threshold on {n_simple} trials. "
            f"{n_fail} trials failed."
        )
        # Update with the successful `peak` calculations
        onset_ixs.update(peak_onset_ixs)
        # Look up corresponding clock times
        onset_times = trial_data.loc[onset_ixs].clock_time
        onset_times.index = onset_ixs.index
        peak_times = trial_data.loc[first_peak_ixs].clock_time
        peak_times.index = first_peak_ixs.index
        # Save the times to trial_info
        self.trial_info[onset_name] = onset_times
        self.trial_info[peak_name] = peak_times


class XDSDataset(BaseDataset):
    def load(self, fpath, sort_channels=False, fixed_channel_count=0):
        """
        Loads in an XDS MATLAB file to build a Dataset object, the
        key component of which is a multiindexed pandas DataFrame.
        TODO: Combine the two XDS loading functions - there seems to
        be a lot of overlap.
        TODO: Standardize naming of the trial_data columns (use e.g.
        trial_id, start_time, end_time)
        TODO: Convert all times to pd.TimeDelta
        TODO: Join base_path and file_name before passing to the load
        function
        TODO: Make sure all of the functionality still works with
        the new trial and trial_info structure.
        """

        # set default name for xds dataset
        if self.name == "":
            file_name = path.basename(fpath)
            self.name = file_name.split("/")[-1].replace(".mat", "")
        # update whether to sort channels or not
        self.sort_channels = sort_channels

        # import xds mat file into python
        try:
            readin = loadmat(fpath)
            logger.info("Loading XDS file saved in standard MAT format")
            xds = readin["xds"]
            h5 = False
        except NotImplementedError:
            logger.info("Loading XDS file saved in MAT v7.3 format")
            h5 = True
            f = h5py.File(fpath, "r")
            xds = f["xds"]

        def to_str(str_arr):
            if len(str_arr.shape) == 0:
                return chr(str_arr)
            else:
                return "".join([chr(a) for a in str_arr])

        # to_str = lambda str_arr : "".join([chr(a) for a in str_arr])
        # ----- collect meta data -----
        meta = {}
        if not h5:
            meta["monkey_name"] = xds["meta"][0][0][0][0]["monkey"][0]
            meta["task_name"] = xds["meta"][0][0][0][0]["task"][0]
            meta["duration"] = xds["meta"][0][0][0][0]["duration"][0][0]
            meta["collect_date"] = xds["meta"][0][0][0][0]["dateTime"][0]
            meta["raw_file_name"] = xds["meta"][0][0][0][0]["rawFileName"][0]
            meta["array"] = xds["meta"][0][0][0][0]["array"][0]
            # dataset info
            meta["bin_width"] = xds["bin_width"][0][0][0][0]
            self.bin_width = xds["bin_width"][0][0][0][0]
            meta["sorted"] = xds["sorted"][0][0][0][0]
        else:
            meta["monkey_name"] = to_str(xds["meta"]["monkey"].value)
            meta["task_name"] = to_str(xds["meta"]["task"].value)
            meta["duration"] = xds["meta"]["duration"].value.item()
            meta["collect_date"] = to_str(xds["meta"]["dateTime"].value)
            meta["raw_file_name"] = to_str(xds["meta"]["rawFileName"].value)
            meta["array"] = to_str(xds["meta"]["array"].value)
            # dataset info
            meta["bin_width"] = xds["bin_width"].value.item()
            self.bin_width = meta["bin_width"]
            meta["sorted"] = xds["sorted"].value.item()
        self.meta = meta

        # ----- collect continuous data -----
        if not h5:
            time_frame = np.squeeze(xds["time_frame"][0][0])
            self.spike_times = xds["spikes"][0][0][0].tolist()
            # collect feature names
            feat_names = {}
            feat_names["spikes"] = [
                name[0] for name in xds["unit_names"][0][0][0].tolist()
            ]
            # some xds datasets may not have 'has_curs'
            if "has_curs" in xds.dtype.names and xds["has_curs"][0][0][0]:
                feat_names["curs_p"] = ["x", "y"]
                feat_names["curs_v"] = ["x", "y"]
                feat_names["curs_a"] = ["x", "y"]
            if xds["has_kin"][0][0][0]:
                feat_names["kin_p"] = ["x", "y"]
                feat_names["kin_v"] = ["x", "y"]
                feat_names["kin_a"] = ["x", "y"]
            if xds["has_EMG"][0][0][0]:
                feat_names["emg"] = [
                    name[0] for name in xds["EMG_names"][0][0][0].tolist()
                ]
            if xds["has_force"][0][0][0]:
                feat_names["force"] = ["x", "y"]
            self.feat_names = feat_names

            try:
                raw_emg_time = np.squeeze(xds["raw_EMG_time_frame"][0][0])
                feat_names["emg_notch"] = [
                    name[0] for name in xds["EMG_names"][0][0][0].tolist()
                ]
                logger.info("Raw EMG found. Will also load Raw EMG data.")
            except KeyError:
                logger.info("No Raw EMG found.")
            try:
                raw_force_time = np.squeeze(xds["raw_force_time_frame"][0][0])
                feat_names["force_notch"] = ["x", "y"]
                logger.info("Raw force found. Will also load raw force data.")
            except KeyError:
                logger.info("No raw force found.")
        else:
            time_frame = xds["time_frame"].value.squeeze()
            self.spike_times = [f[item[0]].value for item in xds["spikes"].value]
            # collect feature names
            feat_names = {}
            feat_names["spikes"] = [
                to_str(f[item[0]].value) for item in xds["unit_names"].value
            ]
            if "has_curs" in xds.keys() and xds["has_curs"].value.item():
                feat_names["curs_p"] = ["x", "y"]
                feat_names["curs_v"] = ["x", "y"]
                feat_names["curs_a"] = ["x", "y"]
            if xds["has_kin"].value.item():
                feat_names["kin_p"] = ["x", "y"]
                feat_names["kin_v"] = ["x", "y"]
                feat_names["kin_a"] = ["x", "y"]
            if xds["has_EMG"].value.item():
                feat_names["emg"] = [
                    to_str(f[item[0]].value) for item in xds["EMG_names"].value
                ]
            if xds["has_force"].value.item():
                feat_names["force"] = ["x", "y"]
            self.feat_names = feat_names

            try:
                raw_emg_time = xds["raw_EMG_time_frame"].value.squeeze()
                feat_names["emg_notch"] = [
                    to_str(f[item[0]].value) for item in xds["EMG_names"].value
                ]
                logger.info("Raw EMG found. Will also load Raw EMG data.")
            except KeyError:
                logger.info("No Raw EMG found.")
            try:
                raw_force_time = xds["raw_force_time_frame"].value.squeeze()
                feat_names["force_notch"] = ["x", "y"]
                logger.info("Raw force found. Will also load raw force data.")
            except KeyError:
                logger.info("No raw force found.")

        # ----- compile trial info -----
        # load the data in xds format
        if not h5:
            trial_info_header = [
                head[0][0] for head in xds["trial_info_table_header"][0][0].tolist()
            ]
            trial_info_table = xds["trial_info_table"][0][0].tolist()
        else:
            trial_info_header = [
                to_str(f[head].value)
                for head in xds["trial_info_table_header"].value.squeeze()
            ]
            trial_info_table = []
            for i in range(xds["trial_info_table"].shape[1]):  # each trial
                row = []
                for j in range(xds["trial_info_table"].shape[0]):  # each field
                    if trial_info_header[j] == "result":
                        val = to_str(f[xds["trial_info_table"][j, i]].value.squeeze())
                    else:
                        val = f[xds["trial_info_table"][j, i]].value.squeeze()
                    row.append(val)
                trial_info_table.append(row)

        # iterate turn trial_info_table into a list with a dict for each trial
        trial_info = []
        for row in trial_info_table:
            trial = {}
            for head, value in zip(trial_info_header, row):
                # get rid of extra dimensions
                val = np.squeeze(value)
                if val.size == 1:
                    val = val.item()
                    if val == "none":
                        val = ""
                # LW: fix changing goCue labeling to be consistent
                if str(head) == "tgtDir":
                    val = np.round(val, decimals=0)

                if str(head) == "goCue":
                    head = "goCueTime"

                # Renaming start/end time, making times timedelta
                if str(head) == "startTime":
                    head = "start_time"
                    val = pd.to_timedelta(val, unit="s")

                if str(head) == "endTime":
                    head = "end_time"
                    val = pd.to_timedelta(val, unit="s")

                if str(head) == "goCueTime":
                    val = pd.to_timedelta(val, unit="s")

                if str(head) == "tgtOnTime":
                    val = pd.to_timedelta(val, unit="s")

                trial[head] = val
            trial_info.append(trial)
        self.trial_info = pd.DataFrame(trial_info)

        # ----- create master dataframe -----
        frames = []
        for signal_type, channels in self.feat_names.items():

            # create tuples that name each column as multiIndex
            midx_tuples = [(signal_type, channel) for channel in channels]
            midx = pd.MultiIndex.from_tuples(
                midx_tuples, names=("signal_type", "channel")
            )

            # get spike counts from raw spike times
            if signal_type == "spikes":
                spike_counts = []
                for times in self.spike_times:
                    times = np.squeeze(times)
                    # add extra bin_width to specify the right boundary of
                    # the largest bin
                    bins = np.arange(
                        time_frame[0],
                        time_frame[-1] + 2 * self.bin_width,
                        self.bin_width,
                    )
                    bins = bins[: time_frame.shape[0] + 1]
                    counts, _ = np.histogram(times, bins)
                    spike_counts.append(counts)
                spike_counts = np.array(spike_counts).T
                # the time indices specify the left boundaries of the bins
                signal_type_data = pd.DataFrame(
                    spike_counts, index=time_frame, columns=midx
                )

                # filling unavailable channels with 0
                if fixed_channel_count:
                    elec_nums = []
                    for chan in channels:
                        elec_nums.append(int("".join([s for s in chan if s.isdigit()])))
                    for c in range(1, fixed_channel_count + 1):
                        col_name = "elec{}".format(c)
                        if c not in elec_nums:
                            logger.info("Zero-filling channel {}".format(c))
                            signal_type_data["spikes", col_name] = 0

                cols = signal_type_data.spikes.columns
                # extract the the numbers out
                cols_num = cols.str.extract(r"(\d+)", expand=False).astype(int)
                # zero pad column names
                cols = ["elec%04d" % c for c in cols_num]
                # replace with new names
                signal_type_data.columns.set_levels(cols, level=1, inplace=True)

            elif signal_type == "emg_notch":
                signal_type = "raw_EMG"

                raw_signal = xds[signal_type][0][0]
                sR = 1.0 / (
                    np.unique(np.round(np.diff(raw_emg_time), 8))[0]
                )  # sampling rate

                b, a = scipy.signal.iirnotch(60, 4, fs=sR)
                notch_signal = scipy.signal.filtfilt(b, a, raw_signal, axis=0)

                # added for peak in 2016 datasets
                b, a = scipy.signal.iirnotch(94, 4, fs=sR)
                notch_signal = scipy.signal.filtfilt(b, a, notch_signal, axis=0)
                # notch_signal = self.apply_pl_harmonic_filter(
                #     raw_signal, sR, n_harmonics=23)

                # visual check
                v = 2
                NPERSEG = 2 ** 16
                f, Pxx_den = scipy.signal.welch(raw_signal[:, v], sR, nperseg=NPERSEG)
                fig, ax = plt.subplots(1, 2, figsize=(20, 7))
                # ax[0].loglog(f, Pxx_den, label='raw')
                ax[0].semilogy(f, Pxx_den, label="raw")
                ax[0].set_xlabel("log freq [Hz]")
                ax[0].set_ylabel("log PSD [V**2/Hz]")
                # ax[0].set_title(
                #     self.name + ' Spectra (Raw), ' + feat_names['emg_notch'][v])

                # high pass filter at 65Hz
                b, a = scipy.signal.butter(4, 65, fs=sR, btype="high")
                hp_signal = scipy.signal.filtfilt(b, a, notch_signal, axis=0)

                # notch signal
                f, Pxx_den = scipy.signal.welch(notch_signal[:, v], sR, nperseg=NPERSEG)
                ax[0].loglog(f, Pxx_den, label="filt")
                # ax[0].semilogy(f, Pxx_den, label='notch')
                f, Pxx_den = scipy.signal.welch(hp_signal[:, v], sR, nperseg=NPERSEG)
                ax[0].loglog(f, Pxx_den, label="filt")
                # ax[0].semilogy(f, Pxx_den, label='hp')
                ax[0].loglog(f, Pxx_den, label="hp")
                # ax[0].set_xlabel('log frequency [Hz]')
                # ax[0].set_ylabel('log PSD [V**2/Hz]')
                ax[0].legend()
                ax[0].set_title(
                    self.name + "EMG Spectra Comparison" + feat_names["emg_notch"][v]
                )

                # rectify
                rect_signal = np.abs(hp_signal)

                # # low pass filter at 10Hz
                b, a = scipy.signal.butter(4, 10, fs=sR, btype="low")
                # lp_signal = scipy.signal.filtfilt(b, a, rect_signal, axis=0)

                # visual check
                f, Pxx_den = scipy.signal.welch(rect_signal[:, v], sR, nperseg=NPERSEG)
                ax[1].loglog(f, Pxx_den)
                # ax[1].semilogy(f, Pxx_den)
                ax[1].set_xlabel("log freq [Hz]")
                ax[1].set_ylabel("log PSD [V**2/Hz]")
                ax[1].set_title(" Rectfied EMG Spectra" + feat_names["emg_notch"][v])
                plt.suptitle(self.name)
                plt.show(block=False)

                # resample down from 10kHz to 1kHz
                resampled_signal = scipy.signal.resample_poly(
                    rect_signal, up=1, down=sR / 1000.0
                )  # uses FIR filter like matlab function - anti aliasing

                signal_type_data = pd.DataFrame(
                    resampled_signal, index=time_frame, columns=midx
                )

            elif signal_type == "force_notch":
                signal_type = "raw_force"

                raw_signal = xds[signal_type][0][0]
                sR = 1.0 / (
                    np.unique(np.round(np.diff(raw_force_time), 8))[0]
                )  # sampling rate

                notch_cent_freq = [60, 94, 180, 220, 300]  # 60, 94 ...
                notch_bw_freq = [2, 2, 2, 2, 2]
                notch_signal = raw_signal
                for n_freq, n_bw in zip(notch_cent_freq, notch_bw_freq):
                    b, a = scipy.signal.iirnotch(n_freq, n_bw, fs=1 / self.bin_size)
                    notch_signal = scipy.signal.filtfilt(b, a, notch_signal, axis=0)

                # resample down from 10kHz to 1kHz
                resampled_signal = scipy.signal.resample_poly(
                    rect_signal, up=1, down=sR / 1000.0
                )  # uses FIR filter like matlab function - anti aliasing
                signal_type_data = pd.DataFrame(
                    resampled_signal, index=time_frame, columns=midx
                )
            else:
                # adapt to naming discrepancies
                if signal_type == "emg":
                    signal_type = "EMG"

                if not h5:
                    # create a dataframe for each signal_type
                    signal = xds[signal_type][0][0]
                else:
                    signal = xds[signal_type].value.squeeze().T

                # correct for zero-upsampling in the xds file - find median interval
                # and replace with nan
                if signal_type.startswith("kin"):
                    kin_mat = signal
                    nonzero_ixs = np.all(kin_mat, axis=1).nonzero()
                    nonzero_ixs = nonzero_ixs[0].astype("int64")
                    intervals = nonzero_ixs - np.roll(nonzero_ixs, 1)
                    interval = np.median(intervals).astype("int64")

                    if sum(intervals == interval) < kin_mat.shape[0] // 2:
                        """if more than half of the values are zero, then the signal
                        is zero-filled so apply sample and hold
                        """
                        logger.warning(
                            "Sample and hold the zero-filled values for the kin field!"
                        )

                        # MRK: sample and hold for kinematic data
                        for p in np.arange(interval):
                            if nonzero_ixs[0] - p * interval < interval:
                                break
                        first_ixs = nonzero_ixs[0] - p * interval

                        i = 0
                        for t in np.arange(first_ixs, kin_mat.shape[0]):
                            if i % interval == 0:
                                hold_val = kin_mat[t, :]
                            else:
                                kin_mat[t, :] = hold_val
                            i += 1
                        signal = kin_mat

                signal_type_data = pd.DataFrame(signal, index=time_frame, columns=midx)

            signal_type_data.index.name = "clock_time"
            frames.append(signal_type_data.copy())

        # concatenate continuous data into one large dataframe with Timedelta indices
        self.data = pd.concat(frames, axis=1)
        if self.sort_channels:
            self.data = self.data.sort_index(axis=1)
        self.data.index = pd.to_timedelta(self.data.index, unit="s")

        if h5:
            f.close()

    def load_XDS_resample(
        self, base_path, file_name, sort_channels=False, fixed_channel_count=0
    ):
        """
        # TODO: Eliminate this?
        Loads in an XDS MATLAB file to build a Dataset object, the
        key component of which is a multiindexed pandas DataFrame.
        Added resampling for Jango15 dataset
        """

        # set default name for xds dataset
        if self.name is None:
            self.name = file_name.split("/")[-1].replace(".mat", "")
        # update whether to sort channels or not
        self.sort_channels = sort_channels

        # generate full path for data file
        file_name = os.path.join(base_path, file_name)

        # import xds mat file into python
        readin = loadmat(file_name)
        xds = readin["xds"]

        # ----- collect meta data -----
        meta = {}
        meta["monkey_name"] = xds["meta"][0][0][0][0]["monkey"][0]
        meta["task_name"] = xds["meta"][0][0][0][0]["task"][0]
        meta["duration"] = xds["meta"][0][0][0][0]["duration"][0][0]
        meta["collect_date"] = xds["meta"][0][0][0][0]["dateTime"][0]
        meta["raw_file_name"] = xds["meta"][0][0][0][0]["rawFileName"][0]
        meta["array"] = xds["meta"][0][0][0][0]["array"][0]
        # dataset info
        meta["bin_width"] = xds["bin_width"][0][0][0][0]
        self.bin_width = xds["bin_width"][0][0][0][0]
        meta["sorted"] = xds["sorted"][0][0][0][0]
        self.meta = meta

        # ----- collect continuous data -----
        time_frame = np.squeeze(xds["time_frame"][0][0])
        self.spike_times = xds["spikes"][0][0][0].tolist()
        # collect feature names
        feat_names = {}
        feat_names["spikes"] = [name[0] for name in xds["unit_names"][0][0][0].tolist()]
        if xds["has_kin"][0][0][0]:
            feat_names["kin_p"] = ["x", "y"]
            feat_names["kin_v"] = ["x", "y"]
            feat_names["kin_a"] = ["x", "y"]
        if xds["has_EMG"][0][0][0]:
            feat_names["emg"] = [name[0] for name in xds["EMG_names"][0][0][0].tolist()]
        if xds["has_force"][0][0][0]:
            feat_names["force"] = ["x", "y"]
        self.feat_names = feat_names

        # ----- compile trial info -----
        # load the data in xds format
        trial_info_header = [
            head[0][0] for head in xds["trial_info_table_header"][0][0].tolist()
        ]
        trial_info_table = xds["trial_info_table"][0][0].tolist()

        # iterate turn trial_info_table into a list with a dict for each trial
        trial_info = []
        for row in trial_info_table:
            trial = {}
            for head, value in zip(trial_info_header, row):
                # get rid of extra dimensions
                val = np.squeeze(value)
                if val.size == 1:
                    val = val.item()
                    if val == "none":
                        val = ""
                # LW: fix changing goCue labeling to be consistent
                if str(head) == "goCue":
                    head = "goCueTime"
                trial[head] = val
            trial_info.append(trial)
        self.trial_info = pd.DataFrame(trial_info)

        # ----- create master dataframe -----
        frames = []
        for signal_type, channels in self.feat_names.items():

            # create tuples that name each column as multiIndex
            midx_tuples = [(signal_type, channel) for channel in channels]
            midx = pd.MultiIndex.from_tuples(
                midx_tuples, names=("signal_type", "channel")
            )

            # get spike counts from raw spike times
            if signal_type == "spikes":
                spike_counts = []
                for times in self.spike_times:
                    times = np.squeeze(times)
                    # add extra bin_width to specify the right boundary of
                    # the largest bin
                    bins = np.arange(
                        time_frame[0],
                        time_frame[-1] + 2 * self.bin_width,
                        self.bin_width,
                    )
                    counts, _ = np.histogram(times, bins)
                    spike_counts.append(counts)
                spike_counts = np.array(spike_counts).T
                # the time indices specify the left boundaries of the bins
                signal_type_data = pd.DataFrame(
                    spike_counts, index=time_frame, columns=midx
                )

                # filling unavailable channels with 0
                if fixed_channel_count:
                    elec_nums = []
                    for chan in channels:
                        elec_nums.append(int("".join([s for s in chan if s.isdigit()])))
                    for c in range(1, fixed_channel_count + 1):
                        col_name = "elec{}".format(c)
                        if c not in elec_nums:
                            logger.info("Zero-filling channel {}".format(c))
                            signal_type_data["spikes", col_name] = 0

                cols = signal_type_data.spikes.columns
                # extract the the numbers out
                cols_num = cols.str.extract(r"(\d+)", expand=False).astype(int)
                # zero pad column names
                cols = ["elec%04d" % c for c in cols_num]
                # replace with new names
                signal_type_data.columns.set_levels(cols, level=1, inplace=True)

            else:
                # adapt to naming discrepancies
                if signal_type == "emg":
                    signal_type = "EMG"

                # create a dataframe for each signal_type
                signal = xds[signal_type][0][0]

                # correct for zero-upsampling in the xds file - find median interval
                # and replace with nan
                if signal_type.startswith("kin") or signal_type.startswith("force"):
                    kin_mat = signal
                    nonzero_ixs = np.all(kin_mat, axis=1).nonzero()
                    nonzero_ixs = nonzero_ixs[0].astype("int64")
                    intervals = nonzero_ixs - np.roll(nonzero_ixs, 1)
                    interval = np.median(intervals).astype("int64")

                    if sum(intervals == interval) < kin_mat.shape[0] // 2:
                        """if more than half of the values are zero, then the signal
                        is zero-filled so apply sample and hold
                        """
                        logger.warning("Resampling and notch filtering the kin field!")

                        # # MRK: sample and hold for kinematic data
                        # for p in np.arange(interval):
                        #     if nonzero_ixs[0] - p*interval < interval:
                        #         break
                        # first_ixs = nonzero_ixs[0] - p*interval

                        # i = 0
                        # for t in np.arange(first_ixs, kin_mat.shape[0]):
                        #     if i % interval == 0:
                        #         hold_val = kin_mat[t,:]
                        #     else:
                        #         kin_mat[t,:] = hold_val
                        #     i += 1
                        # signal = kin_mat
                        signal = self.apply_notch_filter(
                            self.apply_resampling(signal, signal_type)
                        )
                        # pdb.set_trace()

                signal_type_data = pd.DataFrame(signal, index=time_frame, columns=midx)

            signal_type_data.index.name = "clock_time"
            frames.append(signal_type_data.copy())

        # concatenate continuous data into one large dataframe with Timedelta indices
        self.data = pd.concat(frames, axis=1)
        if self.sort_channels:
            self.data = self.data.sort_index(axis=1)
        self.data.index = pd.to_timedelta(self.data.index, unit="s")

    def get_move_onset(
        self,
        move_field="speed",
        start_field="goCueTime",
        end_field="end_time",
        onset_name="move_onset",
        win_start_offset=0,  # ms
        win_end_offset=0,  # ms
        threshold=0.15,
        ignored_trials=None,
    ):
        """Calculates movement onset by finding peak in search window.
        Then a threshold is set based on percentage of peak.
        Crossing point is set as movement onset.
        Calculated forwards from search start and backwards from peak to
        ensure consistent calculation.

        Parameters
        ----------
        start_field: str
            The field name of the start of the window to consider.
        win_start_offset: float,
            The number of ms after start_field to start movement onset search
        win_end_offset: float
            The number of ms after start_field to end movement onset search
        onset_name: str
            The name of the new movement onset trial_info field.
        onset_threshold: float
            The percentage of the max move field to compute onset.
        ignored_trials: pd.Series
            The trials to ignore for this calculation.

              |xxxxxx   entire length of trial  xxxxxxxxxxx|
              |      |==search for move onset==|           |
              |      |                         |           |
        start_field  |                         |           end_field
                     win_start_offset           win_end_offset
              |======|
              |================================|
        """
        # Ignore trials that don't have the required fields
        ti = self.trial_info
        if ignored_trials is None:
            ignored_trials == pd.Series(False, index=ti.index)
        ignored_trials = ti[start_field].isnull() | ignored_trials

        # Make trials to get rid of unneeded data
        trial_data = self.make_trial_data(ignored_trials=ignored_trials)

        # Keep a copy of the grouped movement data
        if type(trial_data[move_field]) is pd.DataFrame:
            new_move_field = "norm_" + move_field
            trial_data[new_move_field] = trial_data[move_field].apply(
                lambda x: np.linalg.norm(x), axis=1
            )
            move_field = new_move_field
        grouped_md = trial_data[["trial_id", move_field]].groupby("trial_id")
        # Convert offsets to timedeltas
        start_offset_td = pd.to_timedelta(win_start_offset, unit="ms")
        end_offset_td = pd.to_timedelta(win_end_offset, unit="ms")
        # Format start and end times for each trial to match trial_data
        td = trial_data
        ti_broadcast = ti.loc[td.trial_id]
        ti_broadcast = ti_broadcast.set_index(td.index)

        window_start = getattr(ti_broadcast, start_field)
        # window_end = getattr(ti_broadcast, end_field)

        # define search window for move onset
        search_start = window_start + start_offset_td
        search_end = window_start + end_offset_td

        # check whether search windows are valid clock times
        valid_range = (search_start < td.clock_time) & (search_end > td.clock_time)

        # find max speed for each trial

        max_speed = grouped_md[move_field].max()
        # broadcast max speed and convert index to match trial data
        max_speed_broadcast = max_speed.loc[td.trial_id]
        max_speed_broadcast = max_speed_broadcast.to_frame().set_index(td.index)

        # calculate threshold for move onset
        threshold_broadcast = max_speed_broadcast * threshold

        m = td[move_field]

        # if multi-dimensional, compute norm to calculate onset
        if type(m) is pd.DataFrame:
            m = m.apply(lambda x: np.linalg.norm(x), axis=1)

        # find threshold crossings (negative to positive)
        crossings = m - threshold_broadcast.squeeze()
        onsets = (crossings < 0) & (crossings.shift(-1) > 0)

        valid_onsets = onsets & valid_range

        onset_ixs = valid_onsets.groupby(td.trial_id).idxmax()
        n_onset = len(onset_ixs)

        logger.info(f"Move onset calculated on {n_onset} trials.")

        onset_times = td.loc[onset_ixs].clock_time
        onset_times.index = onset_ixs.index
        self.trial_info[onset_name] = onset_times


class SimpleJangoDataset(BaseDataset):
    def __init__(self, fpath: str, fixed_channels: int = 100):
        """Calls the load function to reduce the number of steps for loading data.

        Args:
            fpath (str): The path to the data file to load.
            fixed_channels (int, optional): The fixed number of channels to use.
            Defaults to 100.
        """
        self.load(fpath, fixed_channels=fixed_channels)

    def load(self, fpath: str, fixed_channels: int = 100):
        """Loads data from Jango 2015 files, similarly to XDSDataset. Limits
        code complexity by not loading and processing EMG and not covering cases that
        don't apply to Jango 2015 files.

        Args:
            fpath (str): The path to the data file to load.
            fixed_channels (int, optional): The fixed number of channels to use.
            Defaults to 100.
        """
        # Use the filename stem to name the object
        self.name = pathlib.Path(fpath).stem
        # Load the XDS file (.mat < 7.3)
        xds = loadmat(fpath)["xds"]
        # Collect metadata
        xds_meta = xds["meta"][0][0][0][0]
        bin_width = xds["bin_width"][0][0][0][0]
        self.meta = {
            "monkey_name": xds_meta["monkey"][0],
            "task_name": xds_meta["task"][0],
            "duration": xds_meta["duration"][0],
            "collect_date": xds_meta["dateTime"][0],
            "raw_file_name": xds_meta["rawFileName"][0],
            "array": xds_meta["array"][0],
            "bin_width": bin_width,
            "sorted": xds["sorted"][0][0][0][0],
        }
        # Collect continuous data
        time_stamps = np.squeeze(xds["time_frame"][0][0])
        spike_times = xds["spikes"][0][0][0].tolist()
        # Standardize unit names and ordering
        unit_indices = (
            np.array([int(name[0][4:]) for name in xds["unit_names"][0][0][0]]) - 1
        )
        unit_names = [f"{i:04d}" for i in np.arange(fixed_channels)]
        name_dict = {
            "spikes": unit_names,
            "kin_p": ["x", "y"],
            "kin_v": ["x", "y"],
            "kin_a": ["x", "y"],
            # 'emg': [name[0] for name in xds['EMG_names'][0][0][0]],
            # 'emg_notch': [name[0] for name in xds['EMG_names'][0][0][0]],
            "force": ["x", "y"],
        }
        # # TODO: Add EMG import?
        # raw_emg_time_stamps = np.squeeze(xds['raw_EMG_time_frame'][0][0])
        # Collect trial info into a DataFrame
        header = [head[0][0] for head in xds["trial_info_table_header"][0][0]]
        table = xds["trial_info_table"][0][0].tolist()
        trial_info = pd.DataFrame(table, columns=header[:-3])
        # Eliminate extra dimensions from MATLAB
        for col in trial_info.columns:
            trial_info[col] = trial_info[col].apply(np.squeeze)
        # Rename some columns
        trial_info = trial_info.rename(
            columns={
                "startTime": "start_time",
                "endTime": "end_time",
                "goCue": "goCueTime",
            }
        )
        # Convert times into timedeltas
        for col in trial_info.columns:
            if "time" in col.lower():
                trial_info[col] = pd.to_timedelta(trial_info[col], unit="s")
        # Bin the spikes
        extra_point = np.around([time_stamps[-1] + bin_width], decimals=3)
        hist_bins = np.concatenate([time_stamps, extra_point])
        spikes = [np.histogram(st, bins=hist_bins)[0] for st in spike_times]
        spikes = np.stack(spikes, axis=-1)
        # Fill spikes into fixed array structure
        spikes_ordered = np.zeros((len(spikes), fixed_channels))
        spikes_ordered[:, unit_indices] = spikes
        # Add the data to the dataframe
        data_dict = {
            "spikes": spikes_ordered,
            "kin_p": xds["kin_p"][0][0],
            "kin_v": xds["kin_v"][0][0],
            "kin_a": xds["kin_a"][0][0],
            "force": xds["force"][0][0],
        }
        self.init_data_from_dict(
            data_dict,
            bin_width,
            name_dict=name_dict,
            trial_info=trial_info,
            time_stamps=time_stamps,
        )
