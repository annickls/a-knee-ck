#%%
import csv
import yaml
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform

ref_points = {
    "sensor": [
        {"x":0.0, "y": -0.0255, "z":0.0},
        {"x":0.0, "y": 0.0355, "z":0.025},
        {"x":-0.025, "y": 0.0, "z":0.025},
        {"x":-0.025, "y": 0.0, "z":-0.027},
        {"x":-0.0455, "y": 0.0, "z":0.0}],
    "femur": [ # "main"
        {"x":-0.02, "y": 0.0, "z":0.0},
        {"x":0.0185, "y": 0.0, "z":0.0},
        {"x":0.0, "y": -0.0176, "z":-0.018},
        {"x":0.0, "y": 0.0194, "z":-0.018},
        {"x":0.0, "y": 0.0, "z":-0.0254}],
    "tibia": [ # "V3"
        {"x":0.018, "y": 0.0, "z":0.0},
        {"x":-0.028, "y": 0.0, "z":0.0},
        {"x":0.0, "y": -0.0176, "z":-0.02},
        {"x":0.0, "y": 0.0234, "z":-0.02},
        {"x":0.0, "y": 0.0, "z":-0.0354}]
}
#%%

def convert_dict_list_to_point_array(dict_list):
    """Convert a list of dicts with x,y,z keys to a numpy array of points"""
    return np.array([[p['x'], p['y'], p['z']] for p in dict_list])

def convert_point_array_to_dict_list(point_array):
    """Convert a numpy array of points to a list of dicts with x,y,z keys"""
    return [{'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])} for p in point_array]

def sort_points_relative(points1, points2):

    # Compute pairwise distances for each list
    distances_1 = squareform(pdist(points1))
    distances_2 = squareform(pdist(points2))
    # Sum the distances for each point
    sum_distances_1 = np.sum(distances_1, axis=1)
    sum_distances_2 = np.sum(distances_2, axis=1)
    # Get the sorted indices based on the sum of distances
    sorted_indices_1 = np.argsort(sum_distances_1)
    sorted_indices_2 = np.argsort(sum_distances_2)
    # Sort the points based on the computed indices
    sorted_points1 = points1[sorted_indices_1]
    sorted_points2 = points2[sorted_indices_2]

    return sorted_points1, sorted_points2

def convert_csv_to_yaml(csv_files, yaml_file):
    yaml_data = {}
    # Read the CSV file
    point_data = []

    for csv_file in csv_files:

        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            # Skip the header rows
            for _ in range(3):  # Skip the first n lines
                next(csv_reader)
            
            # Extract the point data
            for row in csv_reader:
                if row:
                    x = float(row[1])
                    y = float(row[2])
                    z = float(row[3])
                    point_data.append({"x": x, "y": y, "z": z})

        # Find the corresponding reference marker
        point_name = csv_file.removeprefix(current_folder+"/").removesuffix(".fcsv")
        point_array = convert_dict_list_to_point_array(point_data)
        point_data = []
        point_name_stripped = point_name.removesuffix("_slicer")
        ref_point_array = convert_dict_list_to_point_array(ref_points[point_name_stripped])
        point_array_sorted, ref_point_array_sorted = sort_points_relative(point_array, ref_point_array)
        yaml_data[point_name] = convert_point_array_to_dict_list(point_array_sorted)
        yaml_data[point_name_stripped+"_ref"] = convert_point_array_to_dict_list(ref_point_array_sorted)
    
    # Write to YAML file
    with open(yaml_file, 'w') as f:
        f.write("# Marker coordinates (from the marker \"tibia_body\")\n")
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":

    current_folder = os.path.dirname(os.path.abspath(__file__))
    config_folder = os.path.join(current_folder, "config")
    config_folder = current_folder
    csv_files = [os.path.join(config_folder, file) for file in os.listdir(config_folder) if file.endswith("_slicer.fcsv")]

    convert_csv_to_yaml(csv_files, config_folder+"/marker_coordinates.yaml")
    print("Conversion completed. YAML file created successfully.")