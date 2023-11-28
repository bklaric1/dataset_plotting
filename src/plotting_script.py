import numpy as np
import matplotlib.pyplot as plt
from typing import List
from clc_training_data_creator.datasets_def import load_dataset, TrainingDataOneTrack, LaneFeatureRecord


def load_dataset_and_visualize(dataset_filename):
    print(f"Loading dataset ...")

    dataset : List[LaneFeatureRecord] = load_dataset(dataset_filename)

    print(f"loaded dataset.")

    for single_track in dataset:
        for ranking_problem in single_track.training_records:
            feature_tensors = [feat.variance_based_features for feat in ranking_problem.lane_candidates_feat]

            left_points = LaneFeatureRecord.left_points[0]
            right_points = LaneFeatureRecord.right_points[1]

            fig, ax = plt.subplots()

            ax.plot(left_points, right_points, linewidth=2.0)

            ax.set(xlim=(0, 8), xticks=np.arange(1, 8),  ylim=(0, 8), yticks=np.arange(1, 8))

            plt.show()
            plt.close()
            return


load_dataset_and_visualize('/Users/benjaminklaric/Dataset & Training/dataset.pkl')