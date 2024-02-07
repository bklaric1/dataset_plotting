import numpy as np
import matplotlib.pyplot as plt
from typing import List
from clc_training_data_creator.datasets_def import load_dataset, TrainingDataOneTrack, LaneFeatureRecord, TrainingDataRecord


def load_dataset(dataset_filename):
    print(f"Loading dataset ...")

    dataset : List[TrainingDataOneTrack] = load_dataset(dataset_filename)

    print(f"loaded dataset.")

    return dataset

load_dataset('/Users/benjaminklaric/Dataset & Training/dataset.pkl')

def group(dataset):
    
    features = []
    ious = []
    
    for single_track in dataset:
        for ranking_problem in single_track.training_records:

            ranking: TrainingDataRecord
            for i in range (len(ranking.problem.lane_candidates_feat) - 1):

                features.append(ranking.lane_candidates_ious[i])

                if(ranking.problem.lane_candidates_ious[i] >= 0.9):
                    ious.append(1)
                else:
                    ious.append(0)
    return features, ious

#TODO: add UMAP part from Slack and plot the values and print the lists