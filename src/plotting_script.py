import numpy as np
import matplotlib.pyplot as plt
from typing import List
from clc_training_data_creator.datasets_def import load_dataset, TrainingDataOneTrack, LaneFeatureRecord


def load_dataset_and_visualize(dataset_filename):
    print(f"Loading dataset ...")

    dataset : List[TrainingDataOneTrack] = load_dataset(dataset_filename)

    print(f"loaded dataset.")

    for single_track in dataset:
        for ranking_problem in single_track.training_records:

            lane_lenght = []
            ious_lenght = []

            ious_lenght = ranking_problem.lane_candidates_ious

            feat: LaneFeatureRecord
            for feat in ranking_problem.lane_candidates_feat: #iteriert so viel Mal, wie viele Fahrbahnen es gibt

                a = (feat.left_boundary_len + feat.right_boundary_len)/2

                lane_lenght.append(a)
    
            fig, ax = plt.subplots()
            ax.set_xlabel("lane_lenght")
            ax.set_ylabel("ious_lenght")
            colors = np.random.rand(len(lane_lenght))
            plt.scatter(lane_lenght, ious_lenght, s=10, c=colors, alpha=0.5)

            plt.show()
            plt.close()

load_dataset_and_visualize('/Users/benjaminklaric/Dataset & Training/dataset.pkl')