import numpy as np
import matplotlib.pyplot as plt
from typing import List
from clc_training_data_creator.datasets_def import load_dataset, TrainingDataOneTrack, TrainingDataRecord, LaneFeatureRecord
import shap
import torch
from torch import from_numpy as torch_t_from_numpy

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

    feature_pair = [torch_t_from_numpy(feature_vectors[0].astype(np.float32)), torch_t_from_numpy(feature_vectors[1].astype(np.float32))]

    feature_pair = torch.unsqueeze(torch.stack(feature_pair).squeeze(), dim=0)
    
    model = torch.load('C:\\Users\\Benjamin\\git\\clc_lane_rating_nn\\models\\clc_ranker.pth', map_location=torch.device('cpu')) # - Windows
    #model = torch.load('/Users/benjaminklaric/git/clc_lane_rating_nn/models/clc_ranker.pth', map_location=torch.device('cpu')) # - Mac
    model.eval()

    print(feature_pair)

    feature_pair = feature_pair.clone().detach().requires_grad_(True)
    out = model(feature_pair)
    out.backward()

    print("feature_pair has following dimensions:", feature_pair.shape)
    print(feature_pair.grad)

    def f(feature_pair):
        return model(torch.tensor(feature_pair, dtype=torch.float32, requires_grad=False)).detach().numpy()

    # Get the feature pair for display
    feature_pair_display = feature_pair[0, 0, :]

    # Flatten the feature pair for the explainer
    feature_pair_explainer = feature_pair.detach().numpy()  # Convert to numpy array

    print("this is feature_pair_explainer", feature_pair_explainer)
    print("feature_pair_explainer has following dimensions:", feature_pair_explainer.shape)

    # Create the SHAP explainer
    explainer = shap.KernelExplainer(f, feature_pair_explainer)

    # Compute SHAP values
    shap_values = explainer.shap_values(feature_pair_explainer, nsamples=20)

    # Plot the SHAP force plot
    shap.force_plot(explainer.expected_value, shap_values, feature_pair_display)

dataset = load_dataset_shap('C:\\Users\\Benjamin\\Downloads\\Starkstrom\\dataset.pkl') # - Windows
#dataset = load_dataset_shap('/Users/benjaminklaric/Dataset & Training/dataset.pkl') # - Mac
shap_plot(dataset)