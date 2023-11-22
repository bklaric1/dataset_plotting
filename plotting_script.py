import numpy as np
import matplotlib.pyplot as plt
from typing import List
from clc_training_data_creator.datasets_def import load_dataset, TrainingDataOneTrack


def load_dataset_and_visualize(dataset_filename):
    print(f"Loading dataset ...")

    dataset : List[TrainingDataOneTrack] = load_dataset(dataset_filename)

    print(f"loaded dataset.")

    for single_track in dataset:
        for ranking_problem in single_track.training_records:
            feature_tensors = [feat.variance_based_features for feat in ranking_problem.lane_candidates_feat]

            #Loading is done, just figure out how to plot the dataset

            plt.show()
            plt.close()
            return
            

load_dataset_and_visualize('/Users/benjaminklaric/Dataset & Training/dataset.pkl')