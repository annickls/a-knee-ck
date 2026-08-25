#%%
import numpy as np
import os
import yaml
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation as R

ref_points = {
    "sensor": [
        {"x":0.027, "y": 0.0, "z":0.025},
        {"x":-0.025, "y": 0.0, "z":0.025},
        {"x":0.0, "y": 0.0355, "z":0.0},
        {"x":0.0, "y": -0.0255, "z":0.0},
        {"x":0.0, "y": 0.0, "z":0.043}],
    "tibia":[
        {"x":0.0, "y": 0.0, "z": 0.0},
        {"x":0.0, "y": 10.84, "z": 30.0},
        {"x":0.0, "y": -27.47, "z": 65.84},
        {"x":0.0, "y": 7.47, "z": 65.84}],
    "femur":[
        {"x": 0.0, "y": 0, "z": 0},
        {"x": 0.0, "y": 0.04565, "z": 0},
        {"x": 0.0, "y": 0.02368, "z": -0.04286},
        {"x": 0.0, "y": 0, "z": -0.08616}],
}

def kabsch(filePath, bone):
    """
    Calculate the optimal rigid transformation matrix from P -> Q using Kabsch algorithm
    and returns the rotation matrix and translation, to that
    Q = R * P + t
    -> Test with (R @ bone_stl.T).T + t
    """

    np.set_printoptions(suppress=True)
    with open(filePath, "r") as file:
        content = yaml.safe_load(file)

    def readYaml(marker):
        array = np.array([])
        for i in range(4):
            array = np.append(array, [content[marker][i]["x"], content[marker][i]["y"], content[marker][i]["z"]])
        array = array.reshape([4,3])
        return array

    bone_ref = readYaml(bone+"_ref")
    bone_stl = readYaml(bone+"_slicer")

    p = bone_stl
    q = bone_ref
    
    centroid_p = np.mean(p, axis=0)
    centroid_q = np.mean(q, axis=0)

    p_centered = p - centroid_p
    q_centered = q - centroid_q

    H = p_centered.T@q_centered

    U, _, vt = np.linalg.svd(H)

    R = vt.T @  U.T

    if np.linalg.det(R) < 0:
        vt[-1, :] *= -1
        R = vt.T @ U.T

    t = centroid_q - R@centroid_p

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    print(f"Rotation matrix from STL nach Ref für {bone}: \n{R}")
    print(f"Translation: \n{t}\n")
    return t, R

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

def convert_dict_list_to_point_array(dict_list):
    """Convert a list of dicts with x,y,z keys to a numpy array of points"""
    return np.array([[p['x'], p['y'], p['z']] for p in dict_list])

def convert_point_array_to_dict_list(point_array):
    """Convert a numpy array of points to a list of dicts with x,y,z keys"""
    return [{'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])} for p in point_array]

#%%
filepath_kabsch = "/home/alexandergerard/a-knee-ck/data_for_gui/Model_demo/marker_coordinates.yaml"

# kabsch("data_for_gui\P2_pre\marker_coordinates.yaml", "femur")
with open(filepath_kabsch, "r") as file:
    content = yaml.safe_load(file)
femur_points = convert_dict_list_to_point_array(content["femur_slicer"])
femur_ref_points = convert_dict_list_to_point_array(content["femur_ref"])
translation, rotation = kabsch(filepath_kabsch, "femur")

femur_points_rotated = (rotation@(femur_points.T)).T  + translation

translation, rotation = kabsch(filepath_kabsch, "tibia")

print(femur_points_rotated)
# %%
