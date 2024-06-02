import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import argparse

from clc_training_data_creator.datasets_def import load_dataset, TrainingDataOneTrack

def plot_matrices_on_panel(feature_tensors, plots_per_side, figsize=(15, 15)): 
    """
    Given a list of the feature tensors, this function selects plots_per_side * plots_per_side of them and draws them 
    """
    fig, axes = plt.subplots(plots_per_side, plots_per_side, figsize=figsize)
    
    total_plots = plots_per_side * plots_per_side
    # Plot every kth example with k = N / total_plots where N is the number of feature_tensors
    indices_to_draw = np.linspace(0, len(feature_tensors) - 1, total_plots, dtype=np.int32)
    for i, (ax, index) in enumerate(zip(fig.axes, indices_to_draw)):
        ax.axis('equal') # Same scale for x- and y-axis
        ax.set_title(f"Lane candidate #{i}")
        matrix = np.atleast_2d(feature_tensors[index]) # Make 2D as its only a vector and we need to show a matrix (= 2D-array)
        # Plot the matrix. We need to set min and max such that the colormap is same for every matrix and not autoscaled
        ax.matshow(matrix, cmap="nipy_spectral", vmin=-1, vmax=1)
        for (i, j), z in np.ndenumerate(matrix): # Plot the matrix values
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    return fig, axes

def load_dataset_and_visualize(dataset_filename, save_folder):
    print(f"Loading dataset ...")
    dataset: List[TrainingDataOneTrack] = load_dataset(dataset_filename)
    print(f"Loaded dataset.")

    for idx, single_track in enumerate(dataset):
        track_folder = os.path.join(save_folder, f"Track_{idx}")
        os.makedirs(track_folder, exist_ok=True)  # Create folder for each track
        for idx2, ranking_problem in enumerate(single_track.training_records):
            feature_tensors = [feat.variance_based_features for feat in ranking_problem.lane_candidates_feat]

            fig, axes = plot_matrices_on_panel(feature_tensors, 3, figsize=(10, 10))  # Increased figsize
            fig.suptitle(f"Tensors")
            fig.tight_layout()
            plot_filename = os.path.join(track_folder, f"Plot_{idx2}.png")
            fig.savefig(plot_filename)
            plt.close()

save_folder = "C:\\Users\\Benjamin\\Downloads\\Starkstrom\\Pictures"
dataset_filename = "C:\\Users\\Benjamin\\Downloads\\Starkstrom\\dataset.pkl"

load_dataset_and_visualize(dataset_filename, save_folder)
