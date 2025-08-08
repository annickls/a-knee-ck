import numpy as np
import pandas as pd
import os
import glob


originalPtsFile = 
newPtsFile = 

oldLandmarksFile = 

def kabsch(p, q):
    """Calculate the optimal rigid transformation matrix from Q -> P using Kabsch algorithm"""

    centroid_p = np.mean(p, axis=0)
    centroid_q = np.mean(q, axis=0)

    p_centered = p - centroid_p
    q_centered = q - centroid_q

    H = np.dot(p_centered.T, q_centered)

    U, vt = np.linalg.svd(H)

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



