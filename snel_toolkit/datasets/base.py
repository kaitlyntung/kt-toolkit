import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import scipy.signal as signal

from ..utils import rgetattr, parmap

# Set up the logger for this file
logger = logging.getLogger(__name__)

# TODO: Add replacements for these remaining filtering functions
# - apply_resampling
# - apply_pl_harmonic_filter
# - apply_notch_filter

class BaseDataset(ABC):
    """This abstract base class defines general standard functionality 
    for a dataset and should be subclassed within a module in 
    `rds.datasets` for more specialized functionality (e.g. loading).
    """
    def __init__(self, name):
        """TODO: Would it be helpful to do anything here?
        """
        self.name = name
    
    @abstractmethod
    def load(self):
        """This abstract method should be overridden with a 
        dataset-specific function that initializes self.data, 
        self.trial_info, and self.bin_width.
        """
        pass

    @property
    def bin_size(self):
        """Calculates the bin size in seconds."""
        return (self.data.index[1] - self.data.index[0]).total_seconds()

    def init_data_from_dict(self, 
                            data_dict, 
                            bin_width, 
                            name_dict={}, 
                            trial_info=pd.DataFrame()):
        """Helper function for use in custom loading functions. Allows 
        easier creation of the continuous DataFrame.
        Parameters
        ----------
        data_dict : dict
            A dictionary mapping from signal_type names to numpy arrays.
            The numpy arrays must be time-major and have the same number 
            of rows.
        bin_width : float
            The sampling interval, in seconds.
        name_dict : dict, optional
            A dictionary mapping from signal_type names to lists naming 
            the columns of each numpy array, by default {}
        trial_info : pd.DataFrame, optional
            A dataframe containing information about the trials, 
            by default pd.DataFrame()
        """
        
        self.bin_width = bin_width
        self.trial_info = trial_info
        for signal_type, data in data_dict.items():
            if signal_type not in name_dict:
                # Default to integer string channel names
                chan_nums = range(data.shape[1])
                name_dict[signal_type] = [f'{x:04d}' for x in chan_nums]
        # Create the master DataFrame
        frames = []
        for signal_type, channels in name_dict.items():
            # Create a MultiIndex for the columns
            midx = pd.MultiIndex.from_product(
                [[signal_type], channels], 
                names=('signal_type', 'channel'))
            # Create a DataFrame for each signal_type
            signal = data_dict[signal_type]
            time_frame = bin_width * np.arange(len(signal))
            signal_type_data = pd.DataFrame(
                signal, index=time_frame, columns=midx)
            signal_type_data.index.name = 'clock_time'
            frames.append(signal_type_data.copy())

        # Concatenate continuous data into one large DataFrame 
        # with Timedelta indices
        self.data = pd.concat(frames, axis=1)
        self.data.index = pd.to_timedelta(self.data.index, unit='s')
        # Log the creation of `self.data`
        n_rows, n_cols = self.data.shape
        logger.info('Initialized `self.data` with '
            f'{n_rows} rows and {n_cols} columns.')

    def resample(self, target_bin):
        """Rebins spikes and performs antialiasing + downsampling on 
        continuous signals.
        Parameters
        ----------
        target_bin : float
            The target bin size in seconds. Note that it must be an 
            integer multiple of self.bin_width.
        """

        logger.info(f'Resampling data to {target_bin} sec.')
        # Check that resample_factor is an integer
        resample_factor = target_bin / self.bin_width
        assert resample_factor.is_integer(), \
            'target_bin must be an integer multiple of bin_width.'
        # resamples the analog columns and rebins the spike columns
        def resample_column(x):
            # Always resample so we get the correct indices
            resamp = x.resample('{}S'.format(target_bin)).sum()
            # Make sure `resample` output has same length as decimate
            resamp = resamp[:int(np.ceil(len(x) / resample_factor))]
            # Replace values for non-spike columns
            signal_type = x.name[0]
            if 'spikes' not in signal_type:
                # Apply order 500 Chebychev filter and then downsample
                decimated_x = signal.decimate(
                    x, int(resample_factor), n=500, ftype='fir')
                resamp = pd.Series(decimated_x, index=resamp.index)
            return resamp
        
        # replace data with the resampled data
        self.data = self.data.apply(resample_column)
        self.bin_width = target_bin

        # Round time fields in trial_info
        # timecol = self.trial_info.loc[:, self.trial_info.dtypes == 'timedelta64[ns]']
        # self.trial_info.loc[:, self.trial_info.dtypes == 'timedelta64[ns]'] = \
        #     np.round(timecol / pd.to_timedelta(target_bin, unit='s')) * pd.to_timedelta(target_bin, unit='s')

    def smooth_spk(self, 
                   gauss_width, 
                   signal_type='spikes', 
                   name=None, 
                   overwrite=False):
        """Applies Gaussian smoothing to the data. Most often
        applied to spikes.
        TODO: Clean up this implementation.
        Parameters
        ----------
        gauss_width : int
            The standard deviation of the Gaussian to use for
            smoothing.
        signal_type : str, optional
            The group of signals to smooth, by default 'spikes'
        name : str, optional
            The name to use for the smoothed data when adding 
            it back to the DataFrame, by default None
        overwrite : bool, optional
            Whether to overwrite the signal, by default False
        """

        assert name or overwrite, \
            ('You must either provide a name for the smoothed '
            'data or specify to overwrite the existing data.')

        logger.info(f'Smoothing {signal_type} with a '
            f'{gauss_width} ms Gaussian.')

        # pandas rolling works only on timestamp indices, not timedelta, 
        # so compute window and std with respect to bins
        gauss_bin_std = gauss_width / (1000.0 * self.bin_width)
        # the window extends 3 x std in either direction (USED TO BE 5)
        win_len = int(6 * gauss_bin_std)

        # MRK : a much faster implementation than pandas rolling
        window = signal.gaussian(win_len, gauss_bin_std, sym=True)
        window /=  np.sum(window) 
        spike_vals = self.data[signal_type].values
        spike_vals = [spike_vals[:,i] for i in range(spike_vals.shape[1])]
        
        def filt(args):
            """ MRK: this is used to parallelize spike smoothing
            Parallelized function for filtering spiking data
            """
            x, window = args
            y = signal.lfilter(window, 1., x)
            # shift the signal (acausal filter)
            shift_len = len(window) //2
            y = np.concatenate(
                [y[shift_len:], np.full(shift_len, np.nan)], axis=0)
            return y

        y_list = parmap(
            filt, zip(spike_vals, [window for _ in range(len(spike_vals))]))

        col_names = self.data[signal_type].columns
        if name is None and overwrite:
            smoothed_name = signal_type
        elif name:
            smoothed_name = signal_type + '_' + name
        else:
            assert 0, "Either name or overwrite should be set!"

        for col, v in zip(col_names, y_list):
            self.data[smoothed_name, col] = v
    
    def smooth_cts(self, 
                   signal_type, 
                   filt_type, 
                   crit_freq, 
                   order=4, 
                   max_ripple=1, 
                   name=None, 
                   overwrite=False, 
                   use_causal=False):
        """Smooths groups of columns of continuous-valued data.
        TODO: Revisit these parameter descriptions
        TODO: Clean up this implementation
        TODO: Allow more filter options from scipy (e.g. harmonic, notch)
        Parameters
        ----------
        signal_type : str
            The group of signals to smooth.
        filt_type : {'butter', 'chebyshev'}
            The filter to use.
        crit_freq : float
            The cutoff frequency in Hz.
        order : int, optional
            The order of the filter, by default 4
        max_ripple : int, optional
            Maximum ripple tolerance for the Chebyshev filter, by default 1
        name : str, optional
            The name to use for the smoothed data when adding 
            it back to the DataFrame, by default None, by default None
        overwrite : bool, optional
            Whether to overwrite the signal, by default False
        use_causal : bool, optional
            Whether to enforce causal smoothing, by default False
        """
        
        assert name or overwrite, \
            ('You must either provide a name for the smoothed '
            'data or specify to overwrite the existing data.')

        logger.info(f'Smoothing {signal_type} with {crit_freq} '
            f'Hz {filt_type} filter.')

        # choose a filter type
        samp_freq = 1.0 / self.bin_width
        if filt_type == 'butter':
            sos = signal.butter(
                order, crit_freq, output='sos', fs=samp_freq)
        elif filt_type == 'chebyshev':
            sos = signal.cheby1(
                order, max_ripple, crit_freq, output='sos', fs=samp_freq)
        else:
            raise NotImplementedError(
                f'The `{filt_type}` filter has not been implemented.')

        
        # smooth using the specified sos on the columns of interest
        def smooth_cts_columns(x):
            col_signal_type = x.name[0]
            if col_signal_type == signal_type:
                if use_causal:
                   values = signal.sosfilt(sos, x.values)
                else:
                   values = signal.sosfiltfilt(sos, x.values)
                smoothed = pd.Series(values, index=x.index)
            else:
                smoothed = x
            return smoothed
        
        # overwrite or add columns to the dataframe
        if overwrite:
            self.data = self.data.apply(smooth_cts_columns)
        else:
            new_signal_name = signal_type + '_' + name
            # get a slice of the dataframe while keeping the signal_type level
            type_slice = self.data.xs(
                signal_type, level='signal_type', axis=1, drop_level=False)
            # compute smoothed signal
            type_slice = type_slice.apply(smooth_cts_columns)
            # rename the signal
            type_slice.columns = type_slice.columns.remove_unused_levels()
            type_slice.columns.set_levels(
                [new_signal_name], level='signal_type', inplace=True)
            # append the new signal to the dataframe
            self.data = pd.concat([self.data, type_slice], axis=1)

    def get_pair_xcorr(self, 
                       signal_type, 
                       threshold=None, 
                       zero_chans=False, 
                       channels=None,  
                       max_points=None,
                       removal='corr'):
        """ Calculate the cross-correlations between channels.
        if threshold is set, remove the highly correlated neurons 
        from the dataframe.
        signal_type : str
            The signal type to remove correlated channels from. 
            Most of the time, it will be 'spikes'.
        threshold : float, optional
            The threshold above which to remove neurons, 
            by default None uses no threshold.
        zero_chans : bool, optional
            Whether to zero channels out or remove them 
            entirely, by default False.
        channels : list of str, optional
            NOT IMPLEMENTED. Channels to calculate correlation on,
            by default None.
        max_points : int, optional
            The number of points to use when calculating the correlation,
            taken from the beginning of the data, by default None.
        removal : {'corr', 'rate'}, optional
            Whether to remove neurons in the order of the number 
            of above-threshold correlations to other neurons or 
            of highest firing rate, by default 'corr'. The `rate` option
            is for backwards compatibility with older MATLAB functions.
        """
        assert removal in ['corr', 'rate']

        if max_points is not None:
            data = self.data[:max_points]
        else:
            data = self.data
        # todo: add functionality for channels
        if channels is not None:
            raise NotImplementedError

        np_data = data[signal_type].values
        chan_names = data[signal_type].columns
        n_dim = np_data.shape[1]
        pairs = [(i,k) for i in range(n_dim) for k in range(i)]

        def xcorr_func(args):
            i, k = args
            c =  np.sum(np_data[:,i] * np_data[:,k]).astype(np.float32) 
            if c == 0: return 0.0
            # normalize
            c /= np.sqrt(np.sum(np_data[:,i]**2) * np.sum(np_data[:,k]**2)) 
            return c

        corr_list = parmap(xcorr_func, pairs )

        pair_corr = zip(pairs, corr_list)
        
        chan_names_to_drop = []
        if threshold:
            pair_corr_tmp = list(pair_corr) # create a copy
            if removal == 'corr':
                # sort pairs based on the xcorr values
                pair_corr_tmp.sort(key=lambda x: x[1], reverse=False)
                while pair_corr_tmp:
                    pair, corr = pair_corr_tmp.pop(-1)
                    if corr > threshold:
                        # get corr for all the other pairs which include the neurons from this pair
                        c1 = [p[1] for p in pair_corr if pair[0] in p[0]]
                        c2 = [p[1] for p in pair_corr if pair[1] in p[0]]
                        cnt1 = sum(1 for c in c1 if c > threshold)
                        cnt2 = sum(1 for c in c2 if c > threshold)
                        # determine which channel has more number of highly correlated pairs
                        if cnt1 > cnt2:
                            chan_dropp = pair[0]
                        elif cnt1 < cnt2:
                            chan_dropp = pair[1]
                        else:
                            # if equal, remove the channel with higher mean correlations
                            if np.mean(c1) > np.mean(c1):
                                chan_dropp = pair[0]
                            else:
                                chan_dropp = pair[1]
                        # remove all the pairs with chan_drop included
                        pair_corr_tmp = [p for p in pair_corr_tmp if chan_dropp not in p[0]]
                        chan_names_to_drop.append(chan_names[chan_dropp])
            elif removal == 'rate':
                # Compute the firing rates for all neurons
                neuron_rates = np.mean(np_data, axis=0)
                # Get the pairs with correlation above the threshold
                high_corr_pairs = [pair for pair, corr in pair_corr_tmp if corr > threshold]
                # While there are still correlated pairs, start removing them
                while high_corr_pairs:
                    # Get all unique neurons in these pairs
                    high_corr_channels = np.unique(np.concatenate(high_corr_pairs))
                    # Get rates of highly correlated neurons
                    high_corr_rates = neuron_rates[high_corr_channels]
                    # Select the neuron with the highest rate to drop
                    drop_neuron = high_corr_channels[np.argmax(high_corr_rates)]
                    # Remove the neuron from the highly correlated pairs
                    high_corr_pairs = [p for p in high_corr_pairs if drop_neuron not in p]
                    # Keep track of what the channel names are for the dropped channels
                    chan_names_to_drop.append(chan_names[drop_neuron])
            
            if zero_chans:
                logger.info(f'Zeroing channel names: {chan_names_to_drop}')
                for col in chan_names_to_drop:
                    self.data[signal_type, col] = 0
            else:
                logger.info(f'Removing channel names: {chan_names_to_drop}')
                self.data.drop( [(signal_type, cc) for cc in chan_names_to_drop], axis=1, inplace=True) 
                self.data.columns = self.data.columns.remove_unused_levels()

        return pair_corr, chan_names_to_drop
    
    def calculate_onset(self, 
                        field_name, 
                        onset_threshold, 
                        peak_prominence=0.1,
                        peak_distance_s=0.1,
                        multipeak_threshold=0.2):
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
        
        logger.info(f'Calculating {field_name} onset.')
        sig = rgetattr(self.data, field_name)
        # Find peaks - parameters from the MATLAB script
        peaks, properties = signal.find_peaks(
            sig, 
            prominence=peak_prominence, 
            distance=peak_distance_s / self.bin_width)
        peak_times = pd.Series(self.data.index[peaks])

        # Find the onset for each trial
        onset, onset_index = [], []
        for index, row in self.trial_info.iterrows():
            trial_start, trial_end = row['start_time'], row['end_time']
            # Find the peaks within the trial boundaries
            trial_peaks = peak_times[
                (trial_start < peak_times) & (peak_times < trial_end)]
            peak_signals = sig.loc[trial_peaks]
            # Remove trials with multiple larger peaks
            if multipeak_threshold is not None and len(trial_peaks) > 1:
                # Make sure the first peak is relatively large
                if peak_signals[0]*multipeak_threshold < peak_signals[1:].max():
                    continue
            elif len(trial_peaks) == 0:
                # If no peaks are found for this trial, skip it
                continue
            # Find the point just before speed crosses the threshold
            signal_threshold = onset_threshold * peak_signals[0]
            under_threshold = sig[trial_start:trial_peaks.iloc[0]] < signal_threshold
            if under_threshold.sum() > 0:
                onset.append(under_threshold[::-1].idxmax())
                onset_index.append(index)
        # Add the movement onset for each trial to the DataFrame
        onset_name = field_name.split('.')[-1] + '_onset'
        logger.info(f'`{onset_name}` field created in trial_info.')
        self.trial_info[onset_name] = pd.Series(onset, index=onset_index)
        
        return peak_times

    def make_trial_data(self, 
                        align_field=None, 
                        align_range=(None, None),
                        margin=0,
                        ignored_trials=None,
                        allow_overlap=False):
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
        margin : int, optional
            The number of ms of extra data to include on either end of 
            each trial, labeled with the `margin` column for easy 
            removal. Margins are useful for decoding and smoothing.
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
        
        # Find alignment points
        trial_start = trial_info['start_time']
        trial_end = trial_info['end_time']
        bin_width = pd.to_timedelta(self.bin_width, unit='s')
        if align_field is not None:
            align_points = trial_info[align_field].dt.round(bin_width)
            align_left = align_right = align_points
        else:
            align_field = 'start and end'
            align_left = trial_start
            align_right = trial_end
        
        # Find start and end points based on the alignment range
        start_offset, end_offset = pd.to_timedelta(align_range, unit='ms')
        if pd.isnull(start_offset):
            align_start = trial_start
        else:
            align_start = align_left + start_offset
        if pd.isnull(end_offset):
            align_end = trial_end
        else:
            align_end = align_right + end_offset

        # Add margins to either end of the data
        margin_delta = pd.to_timedelta(margin, unit='ms')
        margin_start = align_start - margin_delta
        margin_end = align_end + margin_delta

        # Store the alignment data in a dataframe
        align_data = pd.DataFrame({
            'margin_start': margin_start,
            'margin_end': margin_end,
            'align_start': align_start, 
            'align_end': align_end,
            'trial_start': trial_start,
            'align_left': align_left}).dropna()
        # Bound the end by the next trial / alignment start
        align_data['end_bound'] = (
            pd.concat([trial_start, align_start], axis=1)
            .min(axis=1)
            .shift(-1))
        trial_dfs = []
        num_overlap_trials = 0
        for trial_id, row in align_data.iterrows():
            # Handle overlap with the start of the next trial
            endpoint = row.margin_end
            if not pd.isnull(row.end_bound) and \
                row.align_end > row.end_bound:

                num_overlap_trials += 1
                if not allow_overlap:
                    # Allow overlapping margins, but not aligned data
                    endpoint = row.end_bound + margin_delta
            # Take a slice of the continuous data
            trial_df = (
                self.data
                .loc[row.margin_start:endpoint]
                .iloc[:-1] # Omit the last entry
                .reset_index() # Move clock_time to column
            )
            # Add trial identifiers
            trial_df['trial_id'] = trial_id
            # Add times to orient the trials
            clock_time = trial_df['clock_time']
            trial_df['trial_time'] = \
                clock_time - row.trial_start.ceil(bin_width)
            trial_df['align_time'] = \
                clock_time - row.align_left.ceil(bin_width)
            trial_df['margin'] = (
                (clock_time < row.align_start) | (row.align_end < clock_time))
            trial_dfs.append(trial_df)
        # Summarize alignment
        logger.info(f'Aligned {len(trial_dfs)} trials to '
            f'{align_field} with offset of {align_range} ms '
            f'and margin of {margin}.')
        # Report any overlapping trials to the user.
        if num_overlap_trials > 0:
            if allow_overlap:
                logger.warning(
                    f'Allowed {num_overlap_trials} overlapping trials.')
            else:
                logger.warning(
                    f'Shortened {num_overlap_trials} trials to prevent overlap.')
        # Combine all trials into one DataFrame
        trial_data = pd.concat(trial_dfs).reset_index(drop=True)
        # Sanity check to make sure there are no duplicated `clock_time`'s
        if not allow_overlap:
            # Duplicated points in the margins are allowed
            td_nonmargin = trial_data[~trial_data.margin]
            assert td_nonmargin.clock_time.duplicated().sum() == 0, \
                'Duplicated points still found. Double-check overlap code.'
        # Make sure NaN's caused by adding trialized data to self.data are ignored
        nans_found = trial_data.isnull().sum().max()
        if nans_found > 0:
            pct_nan = (nans_found / len(trial_data)) * 100
            logger.warning(f'NaNs found in `self.data`. Dropping {pct_nan:.2f}% '
                'of points to remove NaNs from `trial_data`.')
            trial_data = trial_data.dropna()

        return trial_data
    
    def _make_midx(self, signal_type, chan_names=None, num_channels=None):
        """Creates a pd.MultiIndex for a given signal_type."""

        if chan_names is not None:
            # If using custom channel names, make sure they match data
            assert len(chan_names) == num_channels
        elif 'rates' in signal_type:
            # If merging rates, use the same names as the spikes
            chan_names = self.data.spikes.columns
        else:
            # Otherwise, generate names for the channels
            chan_names = [f'{i:04d}' for i in range(num_channels)]
        # Create the MultiIndex for this data
        midx = pd.MultiIndex.from_product(
            [[signal_type], chan_names], names=('signal_type', 'channel'))
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

        logger.info(f'Adding continuous {signal_type} to the main DataFrame.')
        # Make MultiIndex columns
        midx = self._make_midx(signal_type, chan_names, cts_data.shape[1])
        # Build the DataFrame and attach it to the current dataframe
        new_data = pd.DataFrame(
            cts_data, index=self.data.index, columns=midx)
        self.data = pd.concat([self.data, new_data], axis=1)

    def add_trialized_data(self, trial_data, signal_type, chan_names=None):
        """Adds a trialized data field to the main DataFrame.
        
        Parameters
        ----------
        trial_data : pd.DataFrame
            A trial_data dataframe containing a data field
            that will be added to the continuous dataframe.
        signal_type : str
            The label for the data to be added.
        chan_names : list of str, optional
            The channel names for the data when added.
        """

        logger.info(f'Adding trialized {signal_type} to the main DataFrame')
        new_data = trial_data[['clock_time', signal_type]].set_index('clock_time')
        self.data = pd.concat([self.data, new_data], axis=1)