import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Read the data
# Update this path to match your file location
file_path = '/home/annick/a-knee-ck/auswertung/auswertung_hoehenlinien/20250610_162958_0deg_neutral.csv'

try:
    df = pd.read_csv(file_path, comment='#')
    print(f"Successfully loaded data from: {file_path}")
    print(f"Data shape: {df.shape}")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    print("Please check the file path and make sure the file exists.")
    exit(1)

tx = df.iloc[:, 4]  # Tx column (5th column, 0-indexed)
rotation = df.iloc[:, 23]  # Rotation column (24th column, 0-indexed)
flexion = df.iloc[:, 21]  # Flexion column (22nd column, 0-indexed)  
print(min(rotation))
print(max(rotation))
print(rotation[240])