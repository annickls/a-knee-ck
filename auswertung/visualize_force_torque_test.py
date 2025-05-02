import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

def plot_data(forces, torques):
    """Create visualizations of the force and torque data."""
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
    plt.savefig('knee_stability_results.png', dpi=300)
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

def create_3d_visualization(forces, torques):
    """Create 3D visualizations of force and torque data."""
    fig = plt.figure(figsize=(15, 7))
    
    # 3D Force plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(forces[:, 0], forces[:, 1], forces[:, 2], 'b-', alpha=0.6)
    ax1.scatter(forces[:, 0], forces[:, 1], forces[:, 2], c=np.arange(len(forces)), 
                cmap='viridis', s=10, alpha=0.8)
    ax1.set_title('3D Force Trajectory')
    ax1.set_xlabel('Force X (N)')
    ax1.set_ylabel('Force Y (N)')
    ax1.set_zlabel('Force Z (N)')
    
    # 3D Torque plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(torques[:, 0], torques[:, 1], torques[:, 2], 'r-', alpha=0.6)
    ax2.scatter(torques[:, 0], torques[:, 1], torques[:, 2], c=np.arange(len(torques)), 
                cmap='plasma', s=10, alpha=0.8)
    ax2.set_title('3D Torque Trajectory')
    ax2.set_xlabel('Torque X (Nm)')
    ax2.set_ylabel('Torque Y (Nm)')
    ax2.set_zlabel('Torque Z (Nm)')
    
    plt.tight_layout()
    plt.savefig('knee_stability_3d.png', dpi=300)
    plt.show()

def main():
    # File path - replace with your actual file path
    file_path = "print_data.F_sensor_temp_data_6.txt"
    
    try:
        # Load the data
        forces, torques = load_data(file_path)
        
        print(f"Successfully loaded {len(forces)} data points.")
        
        # Create time series plots
        plot_data(forces, torques)
        
        # Calculate and display statistics
        calculate_statistics(forces, torques)
        
        # Create 3D visualizations
        create_3d_visualization(forces, torques)
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()