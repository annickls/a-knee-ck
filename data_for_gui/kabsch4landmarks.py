import numpy as np
import pandas as pd
import os
import glob
from scipy.spatial.distance import pdist, squareform

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

def kabsch(p, q):
    """Calculate the optimal rigid transformation matrix from Q -> P using Kabsch algorithm"""

    centroid_p = np.mean(p, axis=0)
    centroid_q = np.mean(q, axis=0)

    p_centered = p - centroid_p
    q_centered = q - centroid_q

    H = np.dot(p_centered.T, q_centered)

    U, _, vt = np.linalg.svd(H)

    R = np.dot(vt.T, U.T)

    if np.linalg.det(R) < 0:
        vt[-1, :] *= -1
        R = np.dot(vt.T, U.T)

    t = centroid_q - np.dot(centroid_p, R.T)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def read_fcsv(filename):
    """
    Read a .fcsv file and return a DataFrame with x, y, z coordinates and labels as index.
    
    Parameters:
    filename (str): Path to the .fcsv file
    
    Returns:
    pandas.DataFrame: DataFrame with x, y, z columns and labels as index
    """
    # Read the file, skipping comment lines that start with #
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Filter out comment lines
    data_lines = [line for line in lines if not line.strip().startswith('#')]
    
    # Parse the data
    data = []
    for line in data_lines:
        if line.strip():  # Skip empty lines
            parts = line.strip().split(',')
            # Extract relevant columns: id, x, y, z, label
            # Based on the header: id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                label = parts[11] if len(parts) > 11 else f"point_{parts[0]}"
                data.append({'x': x, 'y': y, 'z': z, 'label': label})
            except (ValueError, IndexError):
                continue
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('label', inplace=True)
    
    return df

def write_fcsv(filename, df, original_file):
    """
    Write a DataFrame to .fcsv format, preserving the original file structure.
    
    Parameters:
    filename (str): Path to output .fcsv file
    df (pandas.DataFrame): DataFrame with x, y, z columns and labels as index
    original_file (str): Path to original .fcsv file to copy header and format from
    """
    # Read original file to get header and format
    with open(original_file, 'r') as file:
        lines = file.readlines()
    
    # Extract header lines and data format
    header_lines = [line for line in lines if line.strip().startswith('#')]
    data_lines = [line for line in lines if not line.strip().startswith('#') and line.strip()]
    
    # Write new file
    with open(filename, 'w') as file:
        # Write header
        for header_line in header_lines:
            file.write(header_line)
        
        # Write data with same format as original
        for i, (label, row) in enumerate(df.iterrows(), 1):
            # Use original format but update coordinates
            if data_lines:
                original_parts = data_lines[min(i-1, len(data_lines)-1)].strip().split(',')
                # Replace coordinates and label
                original_parts[1] = str(row['x'])
                original_parts[2] = str(row['y'])
                original_parts[3] = str(row['z'])
                if len(original_parts) > 11:
                    original_parts[11] = str(label)
                file.write(','.join(original_parts) + '\n')
            else:
                # Fallback format
                file.write(f"{i},{row['x']},{row['y']},{row['z']},0,0,0,1,1,1,0,{label},,vtkMRMLScalarVolumeNode1,2,0\n")


def transform_landmarks(bone_name):
    """
    Transform landmarks for a specific bone (femur or tibia) from preOP to postOP coordinates.
    
    Parameters:
    bone_name (str): Either 'femur' or 'tibia'
    """
    # Read kabsch points for transformation
    preop_kabsch = read_fcsv(os.path.join(folderPreOP, f"kabsch_{bone_name}.fcsv"))
    postop_kabsch = read_fcsv(os.path.join(folderPostOP, f"kabsch_{bone_name}.fcsv"))
    
    # Check if specific labels are present in both point lists
    specific_labels = ["spitze_distal", "spitze_proximal", "knochen_distal", "knochen_proximal"]
    labels_present = all(label in preop_kabsch.index and label in postop_kabsch.index 
                        for label in specific_labels)
    label_warning = any(label in preop_kabsch.index and label in postop_kabsch.index 
                        for label in specific_labels)

    if labels_present:
        # Sort points by their name (labels)
        common_labels = sorted(set(preop_kabsch.index) & set(postop_kabsch.index))
        preop_points = preop_kabsch.loc[common_labels][['x', 'y', 'z']].values
        postop_points = postop_kabsch.loc[common_labels][['x', 'y', 'z']].values
        print(f"Sort by names was used for bone {bone_name}")
    else:
        if label_warning:
            print("Some corresponding labels were found but not all of them!")
        # Sort points to ensure correspondence using distance-based method
        preop_points, postop_points = sort_points_relative(
            preop_kabsch[['x', 'y', 'z']].values,
            postop_kabsch[['x', 'y', 'z']].values
        )
    
    # Calculate transformation matrix
    T = kabsch(preop_points, postop_points)
    
    # Read landmarks to transform
    landmarks = read_fcsv(os.path.join(folderPreOP, f"{bone_name}_landmarks.fcsv"))
    
    # Apply transformation
    landmarks_coords = landmarks[['x', 'y', 'z']].values
    # Convert to homogeneous coordinates
    landmarks_homogeneous = np.hstack([landmarks_coords, np.ones((landmarks_coords.shape[0], 1))])
    # Apply transformation
    transformed_homogeneous = (T @ landmarks_homogeneous.T).T
    # Extract coordinates
    transformed_coords = transformed_homogeneous[:, :3]
    
    # Create transformed DataFrame
    transformed_landmarks = landmarks.copy()
    transformed_landmarks[['x', 'y', 'z']] = transformed_coords
    
    # Write to postOP folder
    output_file = os.path.join(folderPostOP, f"{bone_name}_landmarks.fcsv")
    original_file = os.path.join(folderPreOP, f"{bone_name}_landmarks.fcsv")
    write_fcsv(output_file, transformed_landmarks, original_file)
    
    print(f"Transformed {bone_name} landmarks saved to {output_file}")

folderPreOP = "data_for_gui/preOP"
folderPostOP = "data_for_gui"

# Process both femur and tibia
transform_landmarks("femur")
transform_landmarks("tibia")