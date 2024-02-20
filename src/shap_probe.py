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
    
    model = torch.load('C:\\Users\\Benjamin\\git\\clc_lane_rating_nn\\models\\clc_ranker.pth')
    model.eval()

    model.to("cpu")

    print(feature_pair)

    feature_pair = torch.tensor(feature_pair, requires_grad=True)
    out = model(feature_pair)
    out.backward()

    print(feature_pair.grad)

    #print(f"Hallo: {feature_pair.shape}")

    #print(model(feature_pair))
    

    #explainer = shap.Explainer(model, X)
    #shap_values = explainer(X)

    #shap.plots.waterfall(shap_values[0])

dataset = load_dataset_shap('C:\\Users\\Benjamin\\Downloads\\dataset.pkl')
shap_plot(dataset)
