import pandas as pd
import numpy as np
import logging
from scipy.io import loadmat

from .base import BaseDataset

logger = logging.getLogger(__name__)


class MazeDataset(BaseDataset):
    def load(self, filepath):
        """Loads the Maze dataset as a 'continuous' dataframe,
        inserting rows of np.NaNs between trials.
        Parameters
        ----------
        filepath : str
            Path to the Maze .mat file
        TODO: Think about whether np.nan rows are necessary,
            and how to handle trialized data in general
        TODO: Possibly switch to using init_data_from_dict
        """

        ds = loadmat(filepath)["R"]

        nTrials = ds.shape[1]
        nUnits = 202
        nCol = nUnits + 4
        trial_data = [np.full([1, nCol], np.nan)]
        trial_info = {
            "trialID": [],
            "start_time": [],
            "end_time": [],
            "offlineMoveOnsetTime": [],
            "conditionCode": [],
            "targetPosX": [],
            "targetPosY": [],
            "endpointAngle": [],
        }

        # Loop across trials, making trial_info and data array
        pastEnd = 0
        tcount = 0
        for n in range(nTrials):

            # Exclude trials
            if (
                ds[0, n]["unhittable"][0, 0] == 1
                or ds[0, n]["possibleRTproblem"][0, 0] == 1
                or ds[0, n]["photoBoxError"][0, 0] == 1
                or ds[0, n]["trialType"][0, 0] <= 0
                or ds[0, n]["isConsistent"][0, 0] != 1
            ):

                continue

            # Make trial array, change spike times to array
            trialDur = ds[0, n]["trialEndsTime"][0, 0]
            datamat = np.zeros([trialDur, nCol])
            spikeTimes = ds[0, n]["unit"]["spikeTimes"]
            for i in range(nUnits):
                inTrialIdx = spikeTimes[0, i] < trialDur
                spikeIdx = np.floor(spikeTimes[0, i][inTrialIdx]).astype(
                    "int32"
                )
                if len(spikeIdx) != 0:
                    datamat[spikeIdx, i] = 1

            # Extract kinematic info
            handposx = np.squeeze(ds[0, n]["HAND"]["X"][0, 0])
            handposy = np.squeeze(ds[0, n]["HAND"]["Y"][0, 0]) - 8

            handvelx = np.gradient(handposx) / 0.001
            handvely = np.gradient(handposy) / 0.001

            datamat[:, nUnits] = handposx
            datamat[:, nUnits + 1] = handposy
            datamat[:, nUnits + 2] = handvelx
            datamat[:, nUnits + 3] = handvely

            # Append to list
            trial_data.append(datamat)
            trial_data.append(np.full([1, nCol], np.nan))

            # Add to trial_info dict
            trial_info["trialID"].append(ds[0, n]["trialID"][0, 0])
            trial_info["start_time"].append(
                pd.to_timedelta(pastEnd + 1, unit="ms")
            )
            trial_info["end_time"].append(
                pd.to_timedelta(pastEnd + trialDur + 1, unit="ms")
            )
            trial_info["offlineMoveOnsetTime"].append(
                pd.to_timedelta(
                    ds[0, n]["offlineMoveOnsetTime"][0, 0] + pastEnd,
                    unit="ms",
                )
            )
            trial_info["conditionCode"].append(
                1000 * ds[0, n]["trialType"][0, 0]
                + ds[0, n]["trialVersion"][0, 0]
            )

            activeFly = 1
            if ds[0, n]["numFlies"][0, 0] > 1:
                activeFly = ds[0, n]["activeFly"][0, 0]
            xPos = ds[0, n]["PARAMS"]["flyX"][0, 0][0, activeFly - 1]
            yPos = ds[0, n]["PARAMS"]["flyY"][0, 0][0, activeFly - 1]
            trial_info["targetPosX"].append(xPos)
            trial_info["targetPosY"].append(yPos)
            trial_info["endpointAngle"].append(
                np.arctan2(yPos, xPos) * 180 / np.pi
            )

            pastEnd = pastEnd + trialDur + 1

            tcount += 1

        # Make column names
        feat_names = {}
        feat_names["spikes"] = ["%04d" % x for x in range(nUnits)]
        feat_names["kin_p"] = ["x", "y"]
        feat_names["kin_v"] = ["x", "y"]

        dfCols = [("spikes", unit) for unit in feat_names["spikes"]]
        dfCols.extend(
            [("kin_p", "x"), ("kin_p", "y"), ("kin_v", "x"), ("kin_v", "y")]
        )

        # Concat arrays to main dataframe
        self.data = pd.DataFrame(
            np.concatenate(trial_data, axis=0),
            dtype="float32",
            columns=pd.MultiIndex.from_tuples(
                dfCols, names=("signal_type", "channel")
            ),
        )

        # Create trial info dataframe
        self.trial_info = pd.DataFrame(trial_info)

        # Assign attributes
        times = pd.Series(np.arange(self.data.shape[0]))
        self.data.index = pd.to_timedelta(times, unit="ms")
        self.data.index.name = "clock_time"

        # self.name = "maze" + filepath.split(",")[1].replace("-", "")
        self.feat_names = feat_names
        self.bin_width = 0.001

        # Names the array of each channel (1=PMd and 2=M1)
        self.array_lookup = loadmat(filepath)["SU"]["arrayLookup"][0, 0][0]

        logger.info(f"{tcount} trials loaded from {filepath}")

    # Alternate trializing function drafts
    def add_trials(self, start="start_time", end="end_time"):
        """Function that adds trial time and
        trial id columns to continuous dataframe.
        Parameters
        ----------
        start : str, optional
            Column name of trial start time in trial_info dataframe
        end : str, optional
            Column name of trial end time in trial_info dataframe
        TODO: Add checks for overlapping time indices
        TODO: Add bin width flexibility
        TODO: Fix reliance on assumption that clock_time is continuous and starts at 0
        TODO: Compare performance with make_trial_data and see which one is better
        """

        trial_times = np.full([self.data.shape[0]], np.nan)
        trial_ids = np.full([self.data.shape[0]], np.nan)
        for idx, row in self.trial_info.iterrows():
            tstart = int(row.start_time / pd.to_timedelta(1, unit="ms"))
            tend = int(row.end_time / pd.to_timedelta(1, unit="ms"))
            tlength = tend - tstart
            trial_times[tstart:tend] = np.arange(tlength)
            trial_ids[tstart:tend] = idx
        self.data["trial_time"] = pd.to_timedelta(trial_times, unit="ms")
        self.data["trial_id"] = trial_ids.astype("int16")

    def add_align(self, align_field, align_range, align_name="align_time"):
        """Function that calculates and adds
        time indices for aligned trials
        Parameters
        ----------
        align_field : str
            Field in trial_info to serve as alignment point
        align_range : tuple of int
            The offsets to add to the alignment field to 
            calculate the alignment window, in ms
        align_name : str, optional
            Name of the align time column when added
        
        TODO: Same as add_trials
        """

        align_times = np.full([self.data.shape[0]], np.nan)
        for idx, row in self.trial_info.iterrows():
            tstart = tend = int(
                row[align_field] / pd.to_timedelta(1, unit="ms")
            )
            tstart += align_range[0]
            tend += align_range[1]
            tlength = tend - tstart
            align_times[tstart:tend] = np.arange(tlength)
        self.data[align_name] = pd.to_timedelta(align_times, unit="ms")
