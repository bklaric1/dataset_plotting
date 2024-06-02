import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from clc_training_data_creator.plotting import plot_map_points, create_polygon_out_of_driving_lane, plot_polygon
from shapely.geometry import Polygon, GeometryCollection
from clc_lane_rating_nn.validation_plotting import plot_intersection_poly, plot_best_vs_gt
from clc_training_data_creator.datasets_def import load_dataset, TrainingDataOneTrack, LaneFeatureRecord


def load_dataset_and_visualize(dataset_filename, save_folder):
    print(f"Loading dataset ...")

    dataset: List[TrainingDataOneTrack] = load_dataset(dataset_filename)

    print(f"Loaded dataset.")

    for idx, single_track in enumerate(dataset):
        #track_folder = os.path.join(save_folder, f"Track_{idx}")
        #os.makedirs(track_folder, exist_ok=True)
        for idx2, ranking_problem in enumerate(single_track.training_records):
            for candidate in enumerate(ranking_problem.lane_candidates):
                fig, ax = plt.subplots()
                
                plot_best_vs_gt(ax, ranking_problem, 0) 

                #plot_filename = os.path.join(track_folder, f"Plot_{idx2}.png")
                #plt.savefig(plot_filename)
                plt.show()
                plt.close()

save_folder = "C:\\Users\\Benjamin\\Downloads\\Starkstrom\\Pictures"
dataset_filename = "C:\\Users\\Benjamin\\Downloads\\Starkstrom\\dataset.pkl"

load_dataset_and_visualize(dataset_filename, save_folder)
