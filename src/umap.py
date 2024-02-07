'''
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

'''

"""
This script reads the dataset and visualizes the tensors which are used as input to the neural net.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import argparse

from clc_training_data_creator.datasets_def import load_dataset, TrainingDataOneTrack, TrainingDataRecord

def plot_matrices_on_panel(feature_tensors, plots_per_side): 
    """
    Given a list of the feature tensors, this function selects plots_per_side * plots_per_side of them and draws them 
    """
    fig, axes = plt.subplots(plots_per_side, plots_per_side)
    
    total_plots = plots_per_side * plots_per_side
    # Plot every kth example with k = N / total_plots where N is the number of feature_tensors
    indices_to_draw = np.linspace(0, len(feature_tensors) -1 , total_plots, dtype=np.int32)
    for i, (ax, index) in enumerate(zip(fig.axes, indices_to_draw)):
        ax.axis('equal') # Same scale for x- and y-axis
        ax.set_title(f"Lane candidate #{i}")
        matrix = np.atleast_2d(feature_tensors[index]) # Make 2D as its only a vector and we need to show a matrix (= 2D-array)
        # Plot the matrix. We need to set min and max such that the colormap is same for every matrix and not autoscaled
        ax.matshow(matrix, cmap="nipy_spectral", vmin=-1, vmax=1)
        for (i, j), z in np.ndenumerate(matrix): # Plot the matrix values
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    return fig, axes

from umap import UMAP
import matplotlib.pyplot as plt

def main(dataset_filename):
    feature_vectors, labels = get_candidates_feature_vecs(dataset_filename)

    #print(f"Got canidates: feature_vectors:\n{feature_vectors}\nlabels:\n{labels}")
    # Step 2: Structure your data
    # X: 2D array where each row represents a feature vector
    # y: 1D array containing the class labels corresponding to each feature vector

    # Step 3: Apply UMAP
    umap = UMAP(n_components=2)  # You can specify the number of components you want
    embedding = umap.fit_transform(feature_vectors, labels)  # Fit UMAP and transform the data

    print(f"embedding.shape: {embedding.shape}")
    # Plot the results, make scatter plot where each point has as it color the label value. 
    # (which is either 0 or 1). The colormap determines which color 0 and 1 have
    # Step 3: Split the embedding based on class labels
    embedding_class_0 = embedding[np.where(labels == 0)]
    embedding_class_1 = embedding[np.where(labels == 1)]

    # Step 4: Plot the points for each class separately
    plt.scatter(embedding_class_0[:, 0], embedding_class_0[:, 1], c='blue', label='Bad lane', s=5, alpha=.3)
    plt.scatter(embedding_class_1[:, 0], embedding_class_1[:, 1], c='red', label='Good lane', s=5, alpha=.3)

    plt.title('UMAP Projection of Feature Vectors')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    #plt.colorbar()
    plt.savefig("projected_features.png", dpi=300)


def get_candidates_feature_vecs(dataset_filename):
    print(f"Loading dataset ...")
    dataset : List[TrainingDataOneTrack] = load_dataset(dataset_filename)
    print(f"loaded dataset.")

    iou_thresh = .9

    feature_vectors = []
    labels = []

    max_number_of_samples = 10000
    for single_track in dataset:
        ranking_problem: TrainingDataRecord
        for ranking_problem in single_track.training_records:
            
            for i in range(len(ranking_problem.lane_candidates_feat)):
                feature_vector = ranking_problem.lane_candidates_feat[i].variance_based_features
                iou_value = ranking_problem.lane_candidates_ious[i]
                
                labels.append(int(iou_value > iou_thresh))
                feature_vectors.append(feature_vector)

                if len(feature_vectors) >= max_number_of_samples:
                    return feature_vectors, np.array(labels)
            
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_filename", default="", required=True, help="Path to the dataset")
args = parser.parse_args()

main(args.dataset_filename)