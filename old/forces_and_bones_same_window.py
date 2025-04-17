import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from stl import mesh
import matplotlib.colors as colors

# Load sensor data
def load_sensor_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            row = [float(val) for val in line.strip().split(',')]
            data.append(row)
    return np.array(data)

# Function to load STL files
def load_stl(file_path):
    try:
        return mesh.Mesh.from_file(file_path)
    except FileNotFoundError:
        print(f"Warning: Could not find STL file: {file_path}")
        return None

# Function to plot STL mesh
def plot_stl(ax, stl_mesh, color, alpha=0.3):
    if stl_mesh is None:
        return None
    
    # Create a collection of polygons for the STL mesh
    for i in range(len(stl_mesh.vectors)):
        polygon = Poly3DCollection([stl_mesh.vectors[i]])
        polygon.set_facecolor(color)
        polygon.set_alpha(alpha)
        polygon.set_edgecolor('black')
        ax.add_collection3d(polygon)
    
    # Get mesh bounds for setting plot limits
    stl_bounds = np.array([
        stl_mesh.vectors.min(axis=(0, 1)),
        stl_mesh.vectors.max(axis=(0, 1))
    ])
    
    return stl_bounds

# Function to visualize forces and torques
def visualize_forces_torques(sensor_data, femur_stl_path, tibia_stl_path):
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Separate forces and torques
    forces = sensor_data[:, :3]  # First 3 columns are forces (Fx, Fy, Fz)
    torques = sensor_data[:, 3:] # Last 3 columns are torques (Tx, Ty, Tz)
    
    # Load STL meshes
    femur_mesh = load_stl(femur_stl_path)
    tibia_mesh = load_stl(tibia_stl_path)
    
    # Plot STL meshes if available
    all_bounds = []
    
    if femur_mesh is not None:
        femur_bounds = plot_stl(ax, femur_mesh, 'lightgreen')
        all_bounds.append(femur_bounds)
    
    if tibia_mesh is not None:
        tibia_bounds = plot_stl(ax, tibia_mesh, 'lightblue')
        all_bounds.append(tibia_bounds)
    
    # Set plot limits and determine origin point based on STL dimensions
    if all_bounds:
        all_bounds = np.vstack(all_bounds)
        min_bound = all_bounds.min(axis=0)
        max_bound = all_bounds.max(axis=0)
        
        # Calculate midpoint/origin and add padding for plot bounds
        origin_point = (min_bound + max_bound) / 2
        padding = (max_bound - min_bound) * 0.2
        plot_bounds = np.array([
            min_bound - padding,
            max_bound + padding
        ])
    else:
        # Default bounds if no STL files
        plot_bounds = np.array([
            [-10, -10, -10],
            [10, 10, 10]
        ])
        origin_point = np.array([0, 0, 0])
    
    # Scale factors for visualization
    bounds_size = plot_bounds[1] - plot_bounds[0]
    scale_factor = np.min(bounds_size) * 0.2  # Scale to 20% of smallest dimension
    
    # Show time-varying forces and torques
    num_time_steps = len(sensor_data)
    
    # Create a list of unique vectors from our data by sampling
    sampling_rate = max(1, num_time_steps // 10)  # Show at most 10 arrows from time series
    sampled_forces = forces[::sampling_rate]
    sampled_torques = torques[::sampling_rate]
    
    # Normalize and scale vectors for visualization
    for i, (force, torque) in enumerate(zip(sampled_forces, sampled_torques)):
        # Calculate magnitude
        force_mag = np.linalg.norm(force)
        torque_mag = np.linalg.norm(torque)
        
        # Normalize vectors
        force_normalized = force / force_mag if force_mag > 0 else force
        torque_normalized = torque / torque_mag if torque_mag > 0 else torque
        
        # Scale vectors
        force_scaled = force_normalized * scale_factor * force_mag / np.max(np.linalg.norm(forces, axis=1))
        torque_scaled = torque_normalized * scale_factor * torque_mag / np.max(np.linalg.norm(torques, axis=1))
        
        # Create labels only for the first arrow
        force_label = 'Force' if i == 0 else ''
        torque_label = 'Torque' if i == 0 else ''
        
        # Adjust transparency based on vector order (more recent = more opaque)
        alpha = 0.3 + 0.7 * (i / len(sampled_forces))
        
        # Plot force vector (red)
        ax.quiver(origin_point[0], origin_point[1], origin_point[2],
                 force_scaled[0], force_scaled[1], force_scaled[2],
                 color='red', alpha=alpha, arrow_length_ratio=0.2, label=force_label)
        
        # Plot torque vector (blue)
        ax.quiver(origin_point[0], origin_point[1], origin_point[2],
                 torque_scaled[0], torque_scaled[1], torque_scaled[2],
                 color='blue', alpha=alpha, arrow_length_ratio=0.2, label=torque_label)
    
    # Calculate average force and torque for display
    avg_force = np.mean(forces, axis=0)
    avg_torque = np.mean(torques, axis=0)
    
    # Set plot limits
    ax.set_xlim(plot_bounds[0][0], plot_bounds[1][0])
    ax.set_ylim(plot_bounds[0][1], plot_bounds[1][1])
    ax.set_zlim(plot_bounds[0][2], plot_bounds[1][2])
    
    # Mark the origin point with a sphere
    ax.scatter(origin_point[0], origin_point[1], origin_point[2], 
              color='black', s=100, label='Origin')
    
    # Plot force and torque magnitudes over time in a small subplot
    force_magnitudes = np.linalg.norm(forces, axis=1)
    torque_magnitudes = np.linalg.norm(torques, axis=1)
    
    # Create a small subplot for time series data
    ax_force = fig.add_axes([0.15, 0.02, 0.3, 0.1])
    ax_torque = fig.add_axes([0.55, 0.02, 0.3, 0.1])
    
    ax_force.plot(force_magnitudes, 'r-')
    ax_force.set_title('Force Magnitude')
    ax_force.set_xlabel('Time steps')
    
    ax_torque.plot(torque_magnitudes, 'b-')
    ax_torque.set_title('Torque Magnitude')
    ax_torque.set_xlabel('Time steps')
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Forces and Torques with Femur and Tibia')
    
    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Add text with force and torque information
    avg_force_text = f"Avg Force: ({avg_force[0]:.2f}, {avg_force[1]:.2f}, {avg_force[2]:.2f})"
    avg_torque_text = f"Avg Torque: ({avg_torque[0]:.2f}, {avg_torque[1]:.2f}, {avg_torque[2]:.2f})"
    
    ax.text2D(0.05, 0.95, avg_force_text, transform=ax.transAxes, color='red')
    ax.text2D(0.05, 0.90, avg_torque_text, transform=ax.transAxes, color='blue')
    
    # Set equal aspect ratio for better visualization
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    return fig, ax

# Main execution
if __name__ == "__main__":
    # File paths
    sensor_data_file = "print_data.F_sensor_temp_data_7.txt"
    femur_stl_path = "femur.stl"
    tibia_stl_path = "tibia.stl"
    
    # Load sensor data
    sensor_data = load_sensor_data(sensor_data_file)
    
    # Visualize forces and torques with STL files
    fig, ax = visualize_forces_torques(sensor_data, femur_stl_path, tibia_stl_path)
    
    # Print some statistics about the data
    forces = sensor_data[:, :3]
    torques = sensor_data[:, 3:]
    
    print("Force statistics:")
    print(f"Mean: {np.mean(forces, axis=0)}")
    print(f"Max: {np.max(forces, axis=0)}")
    print(f"Min: {np.min(forces, axis=0)}")
    
    print("\nTorque statistics:")
    print(f"Mean: {np.mean(torques, axis=0)}")
    print(f"Max: {np.max(torques, axis=0)}")
    print(f"Min: {np.min(torques, axis=0)}")
    
    plt.show()