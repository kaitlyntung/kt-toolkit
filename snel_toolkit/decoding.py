import logging

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# Set up the logger for this file
logger = logging.getLogger(__name__)


def prepare_decoding_data(
    trial_data,
    x_field,
    y_field,
    valid_ratio=0.2,
    ms_lag=0,
    n_history=0,
    return_groups=False,
):
    """Prepares trialized data for linear decoding by adding lag
    and history features and splitting the data up into training
    and validation sets.
    TODO: Add option for heldout samples instead of trials
    TODO: Update to stratify split by condition, in addition to grouping
    by trial and evenly distributing through time, since sometimes
    certain conditions can be more difficult to decode.
    Parameters
    ----------
    trial_data : pd.DataFrame
        A trialized DataFrame created by `BaseDataset.make_trial_data`.
    x_field : str or list of str
        The column group corresponding to the independent variables.
    y_field : str or list of str
        The column group corresponding to the dependent variables.
    valid_ratio : float, optional
        The ratio of total data to use for validation, by default 0.2.
    ms_lag : int, optional
        The number of ms to lag the dependent variables behind the
        independent variables, by default 0.
    n_history : int, optional
        The number of bins to use as history, by default 0
    return_groups : bool, optional
        Whether to load group (trial) labels for grouped cross-validation,
        by default False
    Returns
    -------
    tuple of tuple of np.ndarrays
        Training and validation 2D arrays, grouped by training and
        validation sets and then by independent and dependent
        variables. Dimensions are num_samples x num_features.

    """

    logger.info(
        f"Preparing data for decoding {x_field} -> {y_field} "
        f"with lag of {ms_lag} ms and {n_history} bins of history"
    )

    # Use lists of fields to enable multiple column groups
    def default_list(x):
        return [x] if type(x) != list else x

    y_field = default_list(y_field)
    x_field = default_list(x_field)
    # Shift all of the time indices to index neural activity from past
    lag_freq = pd.to_timedelta(ms_lag, unit="ms")
    bin_width = trial_data.clock_time.iloc[1] - trial_data.clock_time.iloc[0]
    x_lagged = (
        trial_data[x_field + ["clock_time"]]
        .set_index("clock_time")  # Use clock_time to match with y_field
        .shift(freq=lag_freq)  # Shift the x_field forward in time
    )
    x_history = []
    for i in range(n_history + 1):
        # Shift forward neural activity from the past
        x_past = x_lagged[x_field].shift(freq=(i * bin_width))
        # Rename columns with history labels
        x_past.columns = [f"hist{i}_{c}" for c in x_past.columns.get_level_values(-1)]
        x_history.append(x_past)
    # Concatenate all of the history columns
    x_data = pd.concat(x_history, axis=1).dropna()
    x_data.columns = pd.MultiIndex.from_product([x_field, x_data.columns])
    # Unify data from x_field and y_field on clock_time
    fields = y_field + ["clock_time", "trial_id", "margin"]
    y_data = trial_data[fields].set_index("clock_time")
    # Drop any margins to ensure same y_data, regardless of lag
    # Note that we need to make sure margin is bool (not object)
    y_data = y_data[~y_data.margin.astype(bool)]
    all_data = pd.concat([x_data, y_data], join="inner", axis=1)
    # Ensure that the data matches up
    assert len(all_data) > 0, f"The `ms_lag` of {ms_lag} is invalid for this bin size."
    # Decide whether to split data into train/valid sets
    if valid_ratio == 0:
        # Don't split the data
        x_data = all_data[x_field].values
        y_data = all_data[y_field].values
        if not return_groups:
            return (x_data, y_data)
        # Return trial_id's for proper downstream CV splits
        groups = all_data.trial_id.values
        return (x_data, y_data, groups)

    else:
        # Generate trial-wise training and validation splits
        trial_ids = trial_data["trial_id"].unique()
        valid_ixs = np.arange(len(trial_ids)) % int(1 / valid_ratio) == 0
        valid_trials = trial_ids[valid_ixs]
        # Split the data into training and validation trials
        valid_data_ixs = all_data["trial_id"].isin(valid_trials)
        valid_data = all_data[valid_data_ixs]
        train_data = all_data[~valid_data_ixs]
        # Extract numpy arrays from the DataFrames
        x_train = train_data[x_field].values
        x_valid = valid_data[x_field].values
        y_train = train_data[y_field].values
        y_valid = valid_data[y_field].values

        if not return_groups:
            return (x_train, y_train), (x_valid, y_valid)
        # Return trial_id's for proper downstream CV splits
        groups_train = train_data.trial_id.values
        groups_valid = valid_data.trial_id.values
        return (x_train, y_train, groups_train), (x_valid, y_valid, groups_valid)


class NeuralDecoder:
    """A decoder that performs separate cross-validated hyperparameter
    sweeps for each dimension. Uses a familiar sklearn API.
    """

    def __init__(self, decoder_params):
        """Initializes the NeuralDecoder
        Parameters
        ----------
        decoder_params : dict
            A dictionary of parameters to pass to GridSearchCV.
        """
        self.decoder_params = decoder_params
        self.models = []

    def fit(self, x, y):
        """Trains the model.
        Parameters
        ----------
        x : np.array
            A 2D array of inputs, n_samples x n_features
        y : np.array
            A 2D array of outputs, n_samples x n_features
        """
        # Use a separate decoder for each independent variable
        y_vecs = np.split(y, y.shape[1], axis=1)
        for y_vec in y_vecs:
            model = GridSearchCV(**self.decoder_params)
            model.fit(x, y_vec)
            self.models.append(model)

    def predict(self, x):
        """Predicts the dependent variables for a given set of
        independent variables.
        Parameters
        ----------
        x : np.array
            A 2D array of inputs, n_samples x n_features
        Returns
        -------
        np.array
            A 2D array of predictions, n_samples x n_features
        """
        y_vecs = []
        for model in self.models:
            y_vecs.append(model.predict(x))
        return np.concatenate(y_vecs, axis=-1)

    def score(self, x, y):
        """Evaluates the model using the r2_score scoring function.
        Parameters
        ----------
        x : np.array
            A 2D array of inputs, n_samples x n_features
        y : np.array
            A 2D array of true outputs, n_samples x n_features
        Returns
        -------
        np.array
            An array of performance metrics for each dimension.
        """
        y_hat = self.predict(x)
        return r2_score(y, y_hat, multioutput="raw_values")
