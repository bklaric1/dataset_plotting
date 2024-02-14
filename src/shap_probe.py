import numpy as np
import matplotlib.pyplot as plt
from typing import List
from clc_training_data_creator.datasets_def import load_dataset, TrainingDataOneTrack, TrainingDataRecord, LaneFeatureRecord
import shap
import torch

def load_dataset_shap(dataset_filename):
    print(f"Loading dataset ...")
    dataset: List[TrainingDataOneTrack] = load_dataset(dataset_filename)
    print(f"Loaded dataset.")
    return dataset

def shap_plot(dataset):
    feature_vectors = []

    for single_track in dataset:
        for ranking_problem in single_track.training_records:
            for lane_feature_record in ranking_problem.lane_candidates_feat:
                feature_vector = lane_feature_record.variance_based_features
                feature_vectors.append(feature_vector)

    features_array = np.array(feature_vectors)
    X = torch.tensor(features_array, dtype=torch.float32)

    model = torch.load('C:\\Users\\Benjamin\\git\\clc_lane_rating_nn\\models\\clc_ranker.pth')
    model.eval()

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    shap.plots.waterfall(shap_values[0])

dataset = load_dataset_shap('C:\\Users\\Benjamin\\Downloads\\dataset.pkl')
shap_plot(dataset)
