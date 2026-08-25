#%%
import csv
import yaml
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform

ref_points = {
    "sensor": [
        {"x":0.0, "y": -25.5, "z":0.0},
        {"x":0.0, "y": 35.5, "z":0.0},
        {"x":25.0, "y": 0.0, "z":25.0},
        {"x":25.0, "y": 0.0, "z": -27.0},
        {"x":45.5, "y": 0.0, "z":0.0}],
    "tibia":[
        {"x":0.0, "y": 0.0, "z": 0.0},
        {"x":0.0, "y": 10.84, "z": 30.0},
        {"x":0.0, "y": -27.47, "z": 65.84},
        {"x":0.0, "y": 7.47, "z": 65.84}],
    "femur":[
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 0.0, "y": 45.65, "z": 0.0},
        {"x": 0.0, "y": 23.68, "z": -42.86},
        {"x": 0.0, "y": 0.0, "z": -86.16}],
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
            # Print entire content
            # content = f.read()
            # print(content)
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
        point_name = csv_file.removeprefix(config_folder+"/").removesuffix(".fcsv")
        point_array = convert_dict_list_to_point_array(point_data)
        point_data = []
        point_name_stripped = point_name.removesuffix("_marker")
        ref_point_array = convert_dict_list_to_point_array(ref_points[point_name_stripped])
        point_array_sorted, ref_point_array_sorted = sort_points_relative(point_array, ref_point_array)
        yaml_data[point_name_stripped+"_slicer"] = convert_point_array_to_dict_list(point_array_sorted)
        yaml_data[point_name_stripped+"_ref"] = convert_point_array_to_dict_list(ref_point_array_sorted)
    
    # Write to YAML file
    with open(yaml_file, 'w') as f:
        f.write("# Marker coordinates (from the marker \"tibia_body\")\n")
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":

    current_folder = os.path.dirname(os.path.abspath(__file__))
    config_folder = os.path.join("data_for_gui", "Model_demo")
    # config_folder = current_folder
    csv_files = [os.path.join(config_folder, file) for file in os.listdir(config_folder) if file.endswith("_marker.fcsv")]

    convert_csv_to_yaml(csv_files, config_folder+"/marker_coordinates.yaml")
    print("Conversion completed. YAML file created successfully.")