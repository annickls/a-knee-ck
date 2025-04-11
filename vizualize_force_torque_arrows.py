import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as animation

def load_data(filename):
    """Load force and torque data from the txt file."""
    forces = []
    torques = []
    
    with open(filename, 'r') as file:
        for line in file:
            values = [float(val) for val in line.strip().split(',')]
            if len(values) >= 6:  # Ensure we have at least 6 values (3 forces + 3 torques)
                forces.append(values[0:3])
                torques.append(values[3:6])
    
    return np.array(forces), np.array(torques)

def plot_data_time_series(forces, torques):
    """Create traditional time series plots of the force and torque data."""
    # Create a figure with a 2x1 grid for forces and torques
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Time array (assuming constant sampling rate)
    time = np.arange(len(forces))
    
    # Plot forces
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time, forces[:, 0], 'r-', label='Force X')
    ax1.plot(time, forces[:, 1], 'g-', label='Force Y')
    ax1.plot(time, forces[:, 2], 'b-', label='Force Z')
    ax1.set_title('Forces over Time')
    ax1.set_xlabel('Time (samples)')
    ax1.set_ylabel('Force (N)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot torques
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time, torques[:, 0], 'r-', label='Torque X')
    ax2.plot(time, torques[:, 1], 'g-', label='Torque Y')
    ax2.plot(time, torques[:, 2], 'b-', label='Torque Z')
    ax2.set_title('Torques over Time')
    ax2.set_xlabel('Time (samples)')
    ax2.set_ylabel('Torque (Nm)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('knee_stability_time_series.png', dpi=300)
    plt.show()

def plot_arrow_visualization(forces, torques, sample_rate=10):
    """
    Create 3D visualizations of force and torque data as arrows.
    The arrows point in the direction of the force/torque and their
    size is proportional to the magnitude.
    
    sample_rate: Only plot every nth sample to avoid overcrowding
    """
    # Sample the data to avoid overcrowding
    sampled_indices = np.arange(0, len(forces), sample_rate)
    sampled_forces = forces[sampled_indices]
    sampled_torques = torques[sampled_indices]
    
    # Create a figure with two subplots
    fig = plt.figure(figsize=(15, 7))
    
    # Force arrows plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Calculate force magnitudes for scaling arrows
    force_magnitudes = np.sqrt(np.sum(sampled_forces**2, axis=1))
    max_force_mag = np.max(force_magnitudes) if len(force_magnitudes) > 0 else 1
    
    # Plot force arrows
    origin = np.zeros((len(sampled_forces), 3))  # All arrows start at origin
    
    for i, force in enumerate(sampled_forces):
        # Normalize and scale arrows by magnitude
        magnitude = np.sqrt(np.sum(force**2))
        # Scale arrow length based on magnitude
        length_scale = magnitude / max_force_mag if max_force_mag > 0 else 0
        # Scale arrow width based on magnitude (between 0.5 and 3)
        width_scale = 0.1 + 1.1 * (magnitude / max_force_mag) if max_force_mag > 0 else 0.5
        
        ax1.quiver(origin[i, 0], origin[i, 1], origin[i, 2],
                  force[0], force[1], force[2],
                  color=plt.cm.viridis(i/len(sampled_forces)),
                  linewidth=width_scale,
                  arrow_length_ratio=0.1,
                  length=length_scale * 1.8)  # Scale length to fit in plot
    
    # Set plot limits and labels
    force_max = np.max(np.abs(sampled_forces)) * 1.2
    ax1.set_xlim([-force_max, force_max])
    ax1.set_ylim([-force_max, force_max])
    ax1.set_zlim([-force_max, force_max])
    ax1.set_title('Force Vectors')
    ax1.set_xlabel('Force X (N)')
    ax1.set_ylabel('Force Y (N)')
    ax1.set_zlabel('Force Z (N)')
    
    # Add a colorbar to show time progression
    sm1 = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm1.set_array(np.arange(len(sampled_forces)))
    cbar1 = fig.colorbar(sm1, ax=ax1, pad=0.1)
    cbar1.set_label('Time Progression')
    
    # Torque arrows plot
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Calculate torque magnitudes for scaling arrows
    torque_magnitudes = np.sqrt(np.sum(sampled_torques**2, axis=1))
    max_torque_mag = np.max(torque_magnitudes) if len(torque_magnitudes) > 0 else 1
    
    # Plot torque arrows
    for i, torque in enumerate(sampled_torques):
        # Normalize and scale arrows by magnitude
        magnitude = np.sqrt(np.sum(torque**2))
        # Scale arrow length based on magnitude
        length_scale = magnitude / max_torque_mag if max_torque_mag > 0 else 0
        # Scale arrow width based on magnitude (between 0.5 and 3)
        width_scale = 0.2 + 1.5 * (magnitude / max_torque_mag) if max_torque_mag > 0 else 0.5
        
        ax2.quiver(origin[i, 0], origin[i, 1], origin[i, 2],
                  torque[0], torque[1], torque[2],
                  color=plt.cm.plasma(i/len(sampled_torques)),
                  linewidth=width_scale,
                  arrow_length_ratio=0.1,
                  length=length_scale * 0.8)
    
    # Set plot limits and labels
    torque_max = np.max(np.abs(sampled_torques)) * 1.2
    ax2.set_xlim([-torque_max, torque_max])
    ax2.set_ylim([-torque_max, torque_max])
    ax2.set_zlim([-torque_max, torque_max])
    ax2.set_title('Torque Vectors')
    ax2.set_xlabel('Torque X (Nm)')
    ax2.set_ylabel('Torque Y (Nm)')
    ax2.set_zlabel('Torque Z (Nm)')
    
    # Add a colorbar to show time progression
    sm2 = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
    sm2.set_array(np.arange(len(sampled_torques)))
    cbar2 = fig.colorbar(sm2, ax=ax2, pad=0.1)
    cbar2.set_label('Time Progression')
    
    plt.tight_layout()
    plt.savefig('knee_stability_arrows.png', dpi=300)
    plt.show()

def create_animated_arrows(forces, torques, every_nth=5, save_animation=True):
    """
    Create an animated 3D visualization of force and torque vectors.
    
    Parameters:
    - forces: numpy array of force vectors
    - torques: numpy array of torque vectors
    - every_nth: plot every nth frame to reduce animation size
    - save_animation: whether to save the animation as a gif file
    """
    # Sample data to reduce animation size
    sampled_indices = np.arange(0, len(forces), every_nth)
    sampled_forces = forces[sampled_indices]
    sampled_torques = torques[sampled_indices]
    
    # Calculate maximum magnitudes for scaling
    force_mags = np.sqrt(np.sum(sampled_forces**2, axis=1))
    max_force_mag = np.max(force_mags) if len(force_mags) > 0 else 1
    
    torque_mags = np.sqrt(np.sum(sampled_torques**2, axis=1))
    max_torque_mag = np.max(torque_mags) if len(torque_mags) > 0 else 1
    
    # Create figure
    fig = plt.figure(figsize=(15, 7))
    
    # Force plot
    ax1 = fig.add_subplot(121, projection='3d')
    force_max = np.max(np.abs(sampled_forces)) * 1.2
    ax1.set_xlim([-force_max, force_max])
    ax1.set_ylim([-force_max, force_max])
    ax1.set_zlim([-force_max, force_max])
    ax1.set_title('Force Vectors Over Time')
    ax1.set_xlabel('Force X (N)')
    ax1.set_ylabel('Force Y (N)')
    ax1.set_zlabel('Force Z (N)')
    
    # Torque plot
    ax2 = fig.add_subplot(122, projection='3d')
    torque_max = np.max(np.abs(sampled_torques)) * 1.2
    ax2.set_xlim([-torque_max, torque_max])
    ax2.set_ylim([-torque_max, torque_max])
    ax2.set_zlim([-torque_max, torque_max])
    ax2.set_title('Torque Vectors Over Time')
    ax2.set_xlabel('Torque X (Nm)')
    ax2.set_ylabel('Torque Y (Nm)')
    ax2.set_zlabel('Torque Z (Nm)')
    
    # Create quiver objects for initial frame
    force_quiver = ax1.quiver(0, 0, 0, 0, 0, 0, color='red', linewidth=2, arrow_length_ratio=0.1)
    torque_quiver = ax2.quiver(0, 0, 0, 0, 0, 0, color='blue', linewidth=2, arrow_length_ratio=0.1)
    
    # Add text annotations for time and magnitude
    force_time_text = ax1.text2D(0.05, 0.95, "", transform=ax1.transAxes)
    force_mag_text = ax1.text2D(0.05, 0.90, "", transform=ax1.transAxes)
    torque_time_text = ax2.text2D(0.05, 0.95, "", transform=ax2.transAxes)
    torque_mag_text = ax2.text2D(0.05, 0.90, "", transform=ax2.transAxes)
    
    def update(frame):
        # Update force arrow
        force = sampled_forces[frame]
        force_mag = np.sqrt(np.sum(force**2))
        width_scale = 0.5 + 2.5 * (force_mag / max_force_mag) if max_force_mag > 0 else 0.5
        
        # Replace force quiver
        ax1.clear()
        ax1.set_xlim([-force_max, force_max])
        ax1.set_ylim([-force_max, force_max])
        ax1.set_zlim([-force_max, force_max])
        ax1.set_title('Force Vectors Over Time')
        ax1.set_xlabel('Force X (N)')
        ax1.set_ylabel('Force Y (N)')
        ax1.set_zlabel('Force Z (N)')
        
        ax1.quiver(0, 0, 0, 
                   force[0], force[1], force[2],
                   color='red', 
                   linewidth=width_scale,
                   arrow_length_ratio=0.1,
                   length=force_mag/max_force_mag if max_force_mag > 0 else 0)
        
        # Update torque arrow
        torque = sampled_torques[frame]
        torque_mag = np.sqrt(np.sum(torque**2))
        width_scale = 0.5 + 2.5 * (torque_mag / max_torque_mag) if max_torque_mag > 0 else 0.5
        
        # Replace torque quiver
        ax2.clear()
        ax2.set_xlim([-torque_max, torque_max])
        ax2.set_ylim([-torque_max, torque_max])
        ax2.set_zlim([-torque_max, torque_max])
        ax2.set_title('Torque Vectors Over Time')
        ax2.set_xlabel('Torque X (Nm)')
        ax2.set_ylabel('Torque Y (Nm)')
        ax2.set_zlabel('Torque Z (Nm)')
        
        ax2.quiver(0, 0, 0, 
                   torque[0], torque[1], torque[2],
                   color='blue', 
                   linewidth=width_scale,
                   arrow_length_ratio=0.1,
                   length=torque_mag/max_torque_mag if max_torque_mag > 0 else 0)
        
        # Update text
        force_time_text = ax1.text2D(0.05, 0.95, f"Time: {frame*every_nth}", transform=ax1.transAxes)
        force_mag_text = ax1.text2D(0.05, 0.90, f"Force Mag: {force_mag:.2f}N", transform=ax1.transAxes)
        torque_time_text = ax2.text2D(0.05, 0.95, f"Time: {frame*every_nth}", transform=ax2.transAxes)
        torque_mag_text = ax2.text2D(0.05, 0.90, f"Torque Mag: {torque_mag:.2f}Nm", transform=ax2.transAxes)
        
        return []
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(sampled_forces), interval=100, blit=False)
    
    # Save animation if requested
    if save_animation:
        try:
            print("Saving animation... (this may take a while)")
            anim.save('knee_stability_animation.gif', writer='pillow', fps=10, dpi=100)
            print("Animation saved as 'knee_stability_animation.gif'")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Try installing additional dependencies: pip install pillow")
    
    plt.tight_layout()
    plt.show()

def calculate_statistics(forces, torques):
    """Calculate and print basic statistics for the data."""
    print("=== Force Statistics (N) ===")
    axes = ['X', 'Y', 'Z']
    
    for i in range(3):
        print(f"Force {axes[i]}:")
        print(f"  Mean: {np.mean(forces[:, i]):.2f}")
        print(f"  Max: {np.max(forces[:, i]):.2f}")
        print(f"  Min: {np.min(forces[:, i]):.2f}")
        print(f"  Std Dev: {np.std(forces[:, i]):.2f}")
        print()
    
    print("=== Torque Statistics (Nm) ===")
    for i in range(3):
        print(f"Torque {axes[i]}:")
        print(f"  Mean: {np.mean(torques[:, i]):.2f}")
        print(f"  Max: {np.max(torques[:, i]):.2f}")
        print(f"  Min: {np.min(torques[:, i]):.2f}")
        print(f"  Std Dev: {np.std(torques[:, i]):.2f}")
        print()

def main():
    # File path - replace with your actual file path
    file_path = "print_data.F_sensor_temp_data_5.txt"
    
    try:
        # Load the data
        forces, torques = load_data(file_path)
        
        print(f"Successfully loaded {len(forces)} data points.")
        
        # Create time series plots (traditional line graphs)
        plot_data_time_series(forces, torques)
        
        # Create 3D arrow visualizations
        plot_arrow_visualization(forces, torques, sample_rate=10)
        
        # Calculate and display statistics
        calculate_statistics(forces, torques)
        
        # Create animated visualization (optional)
        # Uncomment the line below to enable animation
        # Note: Animation may be slow/memory-intensive for large datasets
        # create_animated_arrows(forces, torques, every_nth=10)
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()