import numpy as np
import pandas as pd
import pyvista as pv
import time
from scipy.spatial.transform import Rotation as R
import re

# Load the femur STL file
def load_femur(stl_path):
    print(f"Loading femur model from {stl_path}...")
    femur = pv.read(stl_path)
    return femur

# Load quaternion data with special parsing for your format
def load_quaternion_data(quat_file):
    print(f"Loading quaternion data from {quat_file}...")
    
    # Read the file as a single string
    with open(quat_file, 'r') as f:
        content = f.read().strip()
    
    # Your data format appears to be groups of 4 comma-separated values
    # Use regex to find all quaternions in the file
    pattern = r'(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)'
    quaternions = re.findall(pattern, content)
    
    print(f"Found {len(quaternions)} quaternions in the file")
    
    if len(quaternions) == 0:
        print("ERROR: No quaternions found. Check the pattern match.")
        print("File content snippet:", content[:200])
        # Create dummy data
        quaternions = [(0, 0, 0, 1)] * 10
    
    # Convert to dataframe
    data = pd.DataFrame(quaternions, columns=['qx', 'qy', 'qz', 'qw'])
    
    # Convert string values to float
    for col in ['qx', 'qy', 'qz', 'qw']:
        data[col] = data[col].astype(float)
    
    # Add time column
    data['time'] = range(len(data))
    
    # Print sample of parsed data
    print("Sample of parsed quaternion data:")
    print(data.head())
    
    # Verify quaternions
    norms = np.sqrt(data['qx']**2 + data['qy']**2 + data['qz']**2 + data['qw']**2)
    mean_norm = norms.mean()
    print(f"Mean quaternion norm: {mean_norm}")
    
    # Check if quaternions change
    q_diff = data[['qx', 'qy', 'qz', 'qw']].diff().abs().sum(axis=1).sum()
    print(f"Total quaternion difference between frames: {q_diff}")
    
    return data

# Set up the visualization
def setup_visualization():
    print("Setting up visualization...")
    plotter = pv.Plotter(notebook=False, off_screen=False)
    plotter.set_background("white")
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z', line_width=6)
    return plotter

# Debug helper function to visualize axis
def add_coordinate_axes(plotter, scale=50):
    # Add coordinate axes for reference
    origin = np.array([0, 0, 0])
    x_axis = np.array([scale, 0, 0])
    y_axis = np.array([0, scale, 0])
    z_axis = np.array([0, 0, scale])
    
    plotter.add_arrows(origin, x_axis, color='red', shaft_radius=3)
    plotter.add_arrows(origin, y_axis, color='green', shaft_radius=3)
    plotter.add_arrows(origin, z_axis, color='blue', shaft_radius=3)
    
    # Add labels
    plotter.add_point_labels([x_axis], ['X'], font_size=24, text_color='red')
    plotter.add_point_labels([y_axis], ['Y'], font_size=24, text_color='green')
    plotter.add_point_labels([z_axis], ['Z'], font_size=24, text_color='blue')

# Animate the femur using quaternion data
def animate_femur(femur, quat_data, plotter):
    print("Starting animation...")
    
    # Add reference coordinate system
    add_coordinate_axes(plotter)
    
    # Add the femur to the scene with more visible edges
    actor = plotter.add_mesh(femur, color="tan", show_edges=True, edge_color="black", line_width=2)
    
    # Calculate the center of the femur for rotation
    center = femur.center
    print(f"Femur center at: {center}")
    
    # Add a point at the center for debugging
    plotter.add_mesh(pv.Sphere(radius=10, center=center), color='red')
    
    # Set up camera position for better viewing
    plotter.camera_position = 'xy'
    plotter.camera.zoom(0.8)
    
    # Start interactive rendering
    plotter.show(interactive_update=True)
    
    # Add delay before animation
    print("Display initialized. Starting animation in 2 seconds...")
    time.sleep(2)
    
    # Animation settings
    frame_delay = 0.5  # Seconds between frames (slower to see movement better)
    total_frames = len(quat_data)
    print(f"Animating {total_frames} frames with {frame_delay}s delay between frames")
    
    # Exaggerate rotation for visibility
    rotation_scale = 2.0  # Increase this to make rotations more visible
    
    # Store initial femur position
    initial_position = femur.center
    
    # Animation loop
    for i, row in quat_data.iterrows():
        # Progress update
        print(f"Frame {i+1}/{total_frames}")
            
        # Extract quaternion values
        qx, qy, qz, qw = row['qx'], row['qy'], row['qz'], row['qw']
        print(f"Quaternion: [{qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f}]")
        
        # Create rotation from quaternion
        rotation = R.from_quat([qx, qy, qz, qw])
        
        # Optionally exaggerate rotation for visibility
        if rotation_scale != 1.0:
            angles = rotation.as_rotvec()
            scaled_rotation = R.from_rotvec(angles * rotation_scale)
            rotation_matrix = scaled_rotation.as_matrix()
        else:
            rotation_matrix = rotation.as_matrix()
        
        # Reset position and orientation
        actor.SetPosition(0, 0, 0)
        actor.SetOrientation(0, 0, 0)
        
        # Set rotation origin to femur center
        actor.SetOrigin(center)
        
        # Apply the rotation
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        actor.SetUserMatrix(pv.vtkmatrix_from_array(transform))
        
        # Update the renderer and wait
        plotter.update()
        time.sleep(frame_delay)
    
    print("Animation complete.")
    plotter.show()  # Keep window open after animation finishes

def main():
    # Paths to your files - update these to match your file locations
    stl_file = "Right_femur.stl"
    quat_file = "quat_stop_data_20250321_111619.txt"
    
    # Load data
    femur = load_femur(stl_file)
    quat_data = load_quaternion_data(quat_file)
    
    # Setup and run animation
    plotter = setup_visualization()
    animate_femur(femur, quat_data, plotter)

if __name__ == "__main__":
    main()