import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
class PSTH:
    def __init__(self, trial_conds):
        """[summary]
        Parameters
        ----------
        trial_conds : [type]
            [description]
        """
        # A discrete pandas series with trial_id index
        self.trial_conds = trial_conds

    def get_condition_data(self, trial_data, cond, field):
        # Get the trials that match this condition and return their data
        trials = self.trial_conds[self.trial_conds == cond].index
        all_cond_data = trial_data[trial_data.trial_id.isin(trials)]
        cond_data = all_cond_data[field].copy()
        cond_data['align_time'] = all_cond_data.align_time

        return cond_data

    def compute_trial_average(self, trial_data, field, conditions=None):
        
        # Make sure that all required fields are in the dataframe
        required_fields = ['align_time', field]
        assert all([c in trial_data for c in required_fields])
        # Use all available conditions by default
        if conditions is None:
            conditions = self.trial_conds.unique()
        # Cover case where a single condition is passed
        elif type(conditions) != list:
            conditions = [conditions]

        psth_means = {}
        psth_sems = {}
        for cond in conditions:
            cond_data = self.get_condition_data(trial_data, cond, field)
            # Skip the condition if there are no data for this condition
            if len(cond_data) < 1:
                continue
            psth_means[cond] = cond_data.groupby('align_time').mean()
            psth_sems[cond] = cond_data.groupby('align_time').sem()

        return psth_means, psth_sems

    def plot(self, 
             psth_means, 
             psth_sems=None, 
             neurons=None, 
             max_neurons=40,
             max_conditions=10,
             ncols=8,
             cmap=plt.cm.rainbow,
             save_path=None):
        """Plot PSTHs in subplots.
        Parameters
        ----------
        psth_means : dict of DataFrames
            PSTH means returned from the `compute_trial_average` function.
        psth_sems : dict of DataFrames, optional
            PSTH standard error returned from the `compute_trial_average` 
            function, by default None doesn't plot error.
        neurons : list, optional
            The neurons to plot, by default None
        max_neurons : int, optional
            The max number of neurons to plot, by default 40
        max_conditions : int, optional
            The max number of conditions to plot, by default 10
        ncols : int, optional
            The number of subplot columns to use, by default 8
        cmap : matplotlib colormap, optional
            The colormap to use for coloring conditions, by default 
            plt.cm.rainbow
        save_path : str, optional
            The path to save the figure, by default None doesn't save
        """

        # Choose the first few conditions to plot
        plot_conds = sorted(psth_means.keys())[:max_conditions]
        # Assign a color to each condition
        colors = cmap(np.linspace(0, 1, max_conditions))
        # Choose the first few neurons to plot if they aren't specified
        if neurons is None:
            example_psth = psth_means[plot_conds[0]]
            neurons = sorted(example_psth.columns)[:max_neurons]
            n_neurons = max_neurons
        else:
            n_neurons = len(neurons)
        # Create an array of subplots
        nrows = int(np.ceil(n_neurons/ncols))
        fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize=(20,10))
        # Plot a neuron on each axis
        for neuron, ax in zip(neurons, axes.flatten()):
            for cond, color in zip(plot_conds, colors):
                # Plot the mean of the PSTH
                trace = psth_means[cond][neuron]
                ax.plot(trace.index, trace, c=color, label=cond)
                if psth_sems is not None:
                    # Plot standard error of the mean
                    sem = psth_sems[cond][neuron]
                    ax.fill_between(
                        trace.index, 
                        trace-sem, 
                        trace+sem, 
                        color=color, 
                        alpha=0.3)
        # Add a legend for the conditions
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, 
            labels, 
            title='Conditions',
            loc='center left', 
        )
        # Adjust the layout and make room for the legend
        fig.tight_layout()
        fig.subplots_adjust(right=0.95)
        # Save figure if path is specified
        if save_path is not None:
            plt.savefig(save_path)
