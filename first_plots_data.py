# Function to create a unit sphere mesh
def create_sphere_mesh(radius=1.0, resolution=12):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from scipy.spatial.transform import Rotation as R
from scipy.spatial import Delaunay
import math
from stl import mesh  # Import numpy-stl for STL file handling
import os

# Read the data correctly using column names from the header
print("Reading data from 20250430_141205_0deg_neutral.txt...")

# Extract column names from the first line (comment line) of the file
with open("20250509_121004_0deg_neutral.txt", 'r') as f:
    first_line = f.readline().strip()
    
# Remove the # and split by comma
if first_line.startswith('#'):
    column_names = first_line[1:].strip().split(', ')
    data = pd.read_csv("20250509_121004_0deg_neutral.txt", skiprows=1, names=column_names)
else:
    # If for some reason there's no header, use default pandas read_csv
    data = pd.read_csv("20250509_121004_0deg_neutral.txt")

# Print the column names to verify
print("Columns in the dataset:", data.columns.tolist())

# Function to calculate angle between two vectors
def angle_between_vectors(v1, v2):
    """Calculate the angle in degrees between two vectors"""
    # Calculate dot product
    dot_product = np.dot(v1, v2)
    
    # Calculate magnitudes
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    
    # Calculate angle (make sure we're within valid range for arccos)
    cos_angle = min(max(dot_product / (v1_mag * v2_mag), -1.0), 1.0)
    angle = np.arccos(cos_angle) * (180.0 / np.pi)
    
    return angle

# Calculate angles between femur and tibia based on their positions
print("Calculating joint angles based on positions...")
angles = []

for i in range(len(data)):
    # Extract position components
    # Assuming columns 7-9 are femur position and 14-16 are tibia position
    femur_pos = np.array([data.iloc[i, 7], data.iloc[i, 8], data.iloc[i, 9]])
    tibia_pos = np.array([data.iloc[i, 14], data.iloc[i, 15], data.iloc[i, 16]])
    
    # For good anatomical representation, we would ideally have:
    # 1. A point for the hip joint
    # 2. A point for the knee joint
    # 3. A point for the ankle joint
    # Then calculate femur vector (hip to knee) and tibia vector (knee to ankle)
    
    # If we don't have all three points, we can use the femur and tibia position vectors
    # as an approximation to calculate the angle between the segments
    
    # Create vectors from origin to each position
    femur_vector = femur_pos
    tibia_vector = tibia_pos
    
    # Calculate angle between vectors
    angle = angle_between_vectors(femur_vector, tibia_vector)
    angles.append(angle)

# Add the calculated angle to the dataframe
data['FemurTibiaAngle'] = angles

# Calculate torque and force magnitudes
data['TorqueMagnitude'] = np.sqrt(data.iloc[:, 4]**2 + data.iloc[:, 5]**2 + data.iloc[:, 6]**2)
data['ForceMagnitude'] = np.sqrt(data.iloc[:, 1]**2 + data.iloc[:, 2]**2 + data.iloc[:, 3]**2)

print(f"Angle range: {min(angles):.2f}° to {max(angles):.2f}°")
print(f"Torque magnitude range: {data['TorqueMagnitude'].min():.2f} to {data['TorqueMagnitude'].max():.2f} N·m")
print(f"Force magnitude range: {data['ForceMagnitude'].min():.2f} to {data['ForceMagnitude'].max():.2f} N")

# Setup plotting with a standard style
try:
    # Try to use seaborn style if available
    plt.style.use('seaborn-darkgrid')  # More widely available version
except:
    # Fall back to default style if seaborn not available
    print("Default matplotlib style will be used.")
    
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Create figure with 4 subplots (2 rows, 2 columns)
fig = plt.figure(figsize=(18, 12))

# Extract all force vectors from the dataset
force_vectors = []
force_magnitudes = []

for i in range(len(data)):
    # force components (columns 1, 2, 3 are Fx, Fy, Fz)
    fx = data.iloc[i, 1]
    fy = data.iloc[i, 2]
    fz = data.iloc[i, 3]
    force_vec = np.array([fx, fy, fz])
    
    # Normalize the vector to get direction only
    magnitude = np.linalg.norm(force_vec)
    if magnitude > 0:
        normalized_vec = force_vec / magnitude
        force_vectors.append(normalized_vec)
        force_magnitudes.append(magnitude)

# Find maximum magnitude for scaling
max_magnitude = max(force_magnitudes)

# Scale factor for better visualization
magnitude_scale = 2.0 / max_magnitude  # Scale to 2.0 units for maximum magnitude

# Create a colormap normalized to force magnitudes
norm = colors.Normalize(vmin=min(force_magnitudes), vmax=max(force_magnitudes))
cmap = plt.cm.plasma_r  # Reversed plasma colormap

# Create endpoints for all force vectors
endpoints = []
endpoint_colors = []

for vec, mag in zip(force_vectors, force_magnitudes):
    # Normalize the direction vector
    normalized_vec = vec / np.linalg.norm(vec)
    
    # Calculate endpoint based on magnitude
    scaled_magnitude = 0.5 + (mag * magnitude_scale)
    endpoint = normalized_vec * scaled_magnitude
    endpoints.append(endpoint)
    
    # Get color from colormap
    color = cmap(norm(mag))
    endpoint_colors.append(color)

# Convert endpoints to a numpy array
endpoints = np.array(endpoints)

# PLOT 1: Force Vector Mesh (NEW)
ax1 = fig.add_subplot(221, projection='3d')

# Create a mesh from the endpoints by triangulation
# First, project points onto a unit sphere for better triangulation
if len(endpoints) > 3:  # Need at least 4 points for 3D triangulation
    try:
        # Project points to unit sphere for triangulation
        normalized_points = np.array([p/np.linalg.norm(p) for p in endpoints])
        
        # Try to create triangulation
        tri = Delaunay(normalized_points)
        
        # Plot the triangulation as a mesh
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Create triangles
        triangles = endpoints[tri.simplices]
        
        # Create a collection of triangles
        triangle_collection = Poly3DCollection(triangles, alpha=0.6)
        
        # Set face colors based on average magnitude for each triangle
        face_colors = []
        for triangle in tri.simplices:
            # Average the color values of the vertices
            avg_color = np.mean([endpoint_colors[i] for i in triangle], axis=0)
            face_colors.append(avg_color)
        
        triangle_collection.set_facecolor(face_colors)
        triangle_collection.set_edgecolor('lightgray')
        triangle_collection.set_linewidth(0.4)
        
        # Add the collection to the plot
        ax1.add_collection3d(triangle_collection)
        
        # Also plot the original points
        for point, color in zip(endpoints, endpoint_colors):
            ax1.scatter(point[0], point[1], point[2], color=color, s=1, alpha=0.8)
    
    except Exception as e:
        print(f"Error creating triangulation: {e}")
        # Fallback to just plotting the points
        for point, color in zip(endpoints, endpoint_colors):
            ax1.scatter(point[0], point[1], point[2], color=color, s=1, alpha=0.8)
else:
    # If fewer than 4 points, just plot them
    for point, color in zip(endpoints, endpoint_colors):
        ax1.scatter(point[0], point[1], point[2], color=color, s=1, alpha=0.8)

# Add a wireframe sphere for reference
radius = 1.0
sphere_x, sphere_y, sphere_z = create_sphere_mesh(radius=radius, resolution=20)
ax1.plot_wireframe(sphere_x, sphere_y, sphere_z, color='gray', alpha=0.1, linewidth=0.5)

# Set plot properties
ax1.set_xlabel('X Direction')
ax1.set_ylabel('Y Direction')
ax1.set_zlabel('Z Direction')
ax1.set_title('Force Vector Mesh Visualization')
ax1.set_box_aspect([1, 1, 1])

# Set plot limits with some margin
margin = 0.5
ax1.set_xlim([-radius - margin, radius + margin])
ax1.set_ylim([-radius - margin, radius + margin])
ax1.set_zlim([-radius - margin, radius + margin])

# Add colorbar for force magnitude
cbar1 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, pad=0.1)
cbar1.set_label('Force Magnitude (N)')

# PLOT 2: Tibia Position Path in 3D space
ax2 = fig.add_subplot(222, projection='3d')

# Extract tibia position columns (assuming they are columns 14, 15, 16)
# If the position columns are different, adjust these indices
tibia_pos_x = data.iloc[:, 7].to_numpy()  # Tibia position X
tibia_pos_y = data.iloc[:, 8].to_numpy()  # Tibia position Y
tibia_pos_z = data.iloc[:, 9].to_numpy()  # Tibia position Z
time_array = data.iloc[:, 0].to_numpy()

# Plot the position path
scatter = ax2.scatter(tibia_pos_x, tibia_pos_y, tibia_pos_z, 
                     c=time_array, cmap='viridis', s=40, alpha=0.8)
ax2.plot(tibia_pos_x, tibia_pos_y, tibia_pos_z, 'b-', linewidth=1.5)

ax2.set_xlabel('X position')
ax2.set_ylabel('Y position')
ax2.set_zlabel('Z position')
ax2.set_title('Tibia Position Path')

# Add colorbar for time reference inside the plot
cbar = fig.colorbar(scatter, ax=ax2, pad=0.1)  # Adjust pad to move colorbar to the side
cbar.set_label('Time (s)')

# PLOT 3: Sphere with Force Direction Squares
ax3 = fig.add_subplot(223, projection='3d')

# Resolution for discretizing force directions
azimuth_resolution = 20  # Horizontal/longitude resolution (around the equator)
polar_resolution = 20    # Vertical/latitude resolution (from pole to pole)

# Function to discretize a direction vector to spherical coordinates grid cells
def discretize_direction(direction, azimuth_res, polar_res):
    # Convert to spherical coordinates
    theta = np.arctan2(direction[1], direction[0])  # azimuth angle (-π to π)
    phi = np.arccos(np.clip(direction[2], -1.0, 1.0))  # polar angle (0 to π)
    
    # Discretize angles
    theta_bins = 2 * azimuth_res  # Full 360° for azimuth
    phi_bins = polar_res          # 180° for polar angle
    
    theta_bin = int((theta + np.pi) / (2 * np.pi / theta_bins))
    phi_bin = int(phi / (np.pi / phi_bins))
    
    # Handle edge cases
    if theta_bin == theta_bins:
        theta_bin = 0
    if phi_bin == phi_bins:
        phi_bin = phi_bins - 1
    
    return (theta_bin, phi_bin)

# Dictionary to store force magnitudes for each discretized direction
direction_magnitudes = {}

# Process all force vectors
for i, (vec, mag) in enumerate(zip(force_vectors, force_magnitudes)):
    # Discretize the direction
    dir_key = discretize_direction(vec, azimuth_resolution, polar_resolution)
    
    # Add magnitude to this direction
    if dir_key in direction_magnitudes:
        direction_magnitudes[dir_key].append(mag)
    else:
        direction_magnitudes[dir_key] = [mag]

# Function to create a square patch on the sphere surface
def create_square_on_sphere(theta_bin, phi_bin, azimuth_res, polar_res, radius=1.0, scale=0.9):
    # Calculate the angular ranges
    theta_bins = 2 * azimuth_res
    phi_bins = polar_res
    
    delta_theta = 2 * np.pi / theta_bins
    delta_phi = np.pi / phi_bins
    
    # Calculate center angles
    theta_center = (theta_bin + 0.5) * delta_theta - np.pi
    phi_center = (phi_bin + 0.5) * delta_phi
    
    # Calculate the corners of the patch in spherical coordinates
    # Scale factor creates gaps between patches
    half_delta_theta = delta_theta * scale / 2
    half_delta_phi = delta_phi * scale / 2
    
    theta_corners = [theta_center - half_delta_theta, theta_center + half_delta_theta, 
                    theta_center + half_delta_theta, theta_center - half_delta_theta]
    phi_corners = [phi_center - half_delta_phi, phi_center - half_delta_phi,
                  phi_center + half_delta_phi, phi_center + half_delta_phi]
    
    # Convert corners to Cartesian coordinates
    x_vals = []
    y_vals = []
    z_vals = []
    
    for t, p in zip(theta_corners, phi_corners):
        x = radius * np.sin(p) * np.cos(t)
        y = radius * np.sin(p) * np.sin(t)
        z = radius * np.cos(p)
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)
    
    # Add a 5th point that equals the first to close the polygon
    x_vals.append(x_vals[0])
    y_vals.append(y_vals[0])
    z_vals.append(z_vals[0])
    
    return np.array([x_vals, y_vals, z_vals])

# First draw a transparent wireframe sphere for reference
radius = 1.0
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
x = radius * np.outer(np.cos(u), np.sin(v))
y = radius * np.outer(np.sin(u), np.sin(v))
z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax3.plot_wireframe(x, y, z, color='gray', alpha=0.1, linewidth=0.5)

# Plot squares at each discretized direction with color based on average force magnitude
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

for dir_key, magnitudes in direction_magnitudes.items():
    # Calculate average magnitude
    avg_magnitude = np.mean(magnitudes)
    
    # Get color from colormap
    color = cmap(norm(avg_magnitude))
    
    # Create square vertices on the sphere surface
    theta_bin, phi_bin = dir_key
    square_vertices = create_square_on_sphere(theta_bin, phi_bin, azimuth_resolution, polar_resolution)
    
    # Create vertices for the Poly3DCollection
    x_vals, y_vals, z_vals = square_vertices
    vertices = [[x_vals[i], y_vals[i], z_vals[i]] for i in range(4)]
    
    # Create a polygon collection for the square
    poly = Poly3DCollection([vertices], alpha=0.9)
    
    # Set face color based on force magnitude
    poly.set_facecolor(color)
    poly.set_edgecolor(color)
    poly.set_linewidth(0.5)
    
    # Add the polygon to the plot
    ax3.add_collection3d(poly)

# Set plot properties
ax3.set_xlabel('X Direction')
ax3.set_ylabel('Y Direction')
ax3.set_zlabel('Z Direction')
ax3.set_title('Force Direction with Colored Squares on Sphere')
ax3.set_box_aspect([1, 1, 1])

# Set plot limits with some margin
margin = 0.2
ax3.set_xlim([-radius - margin, radius + margin])
ax3.set_ylim([-radius - margin, radius + margin])
ax3.set_zlim([-radius - margin, radius + margin])

# Add colorbar for force magnitude
cbar3 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3, pad=0.1)
cbar3.set_label('Average Force Magnitude (N)')

# Function to load and transform an STL file
def load_stl_for_plot(ax, stl_file_path, scale_factor, translation, rotation_deg, alpha=0.3, face_color=[0.8, 0.8, 0.9]):
    """Load and transform an STL file for plotting
    
    Args:
        ax: Matplotlib 3D axis
        stl_file_path: Path to STL file
        scale_factor: Scale factor for the model
        translation: [x, y, z] translation
        rotation_deg: [x, y, z] rotation in degrees
        alpha: Transparency level
        face_color: RGB tuple for face color
        
    Returns:
        vertices: Transformed vertices
        success: Boolean indicating if loading was successful
    """
    vertices = None
    success = False
    
    try:
        if os.path.exists(stl_file_path):
            print(f"Loading STL file: {stl_file_path}")
            # Load the STL file
            stl_mesh = mesh.Mesh.from_file(stl_file_path)
            
            # Extract vertices for plotting
            vertices = stl_mesh.vectors.copy()  # Create a copy to modify
            
            # Find the center of the STL model
            x_min, x_max = vertices[:,:,0].min(), vertices[:,:,0].max()
            y_min, y_max = vertices[:,:,1].min(), vertices[:,:,1].max()
            z_min, z_max = vertices[:,:,2].min(), vertices[:,:,2].max()
            
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            center_z = (z_min + z_max) / 2
            
            # Calculate the max dimension for normalization
            max_dim = max(x_max - x_min, y_max - y_min, z_max - z_min)
            
            # 1. First center the model to origin
            vertices[:,:,0] -= center_x
            vertices[:,:,1] -= center_y
            vertices[:,:,2] -= center_z
            
            # 2. Scale the model
            vertices = vertices * (scale_factor / max_dim)
            
            # 3. Apply rotation (converting degrees to radians)
            # Create rotation matrix from Euler angles (degrees)
            rotation = R.from_euler('xyz', [
                np.radians(rotation_deg[0]),  # X rotation
                np.radians(rotation_deg[1]),  # Y rotation
                np.radians(rotation_deg[2])   # Z rotation
            ], degrees=False)
            
            # Apply rotation to each vertex of each face
            for i in range(vertices.shape[0]):  # For each face
                for j in range(vertices.shape[1]):  # For each vertex of the face
                    # Apply rotation to the vertex
                    vertices[i,j,:] = rotation.apply(vertices[i,j,:])
            
            # 4. Apply translation
            vertices[:,:,0] += translation[0]  # X translation
            vertices[:,:,1] += translation[1]  # Y translation
            vertices[:,:,2] += translation[2]  # Z translation
            
            # Plot the STL mesh
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            # Create a collection of triangles
            triangle_collection = Poly3DCollection(vertices, alpha=alpha)
            
            # Set face colors
            triangle_collection.set_facecolor(face_color)
            triangle_collection.set_edgecolor('gray')
            
            # Add the collection to the plot
            ax.add_collection3d(triangle_collection)
            
            print(f"STL model loaded successfully.")
            print(f"  - Scale factor: {scale_factor}")
            print(f"  - Translation: {translation}")
            print(f"  - Rotation: {rotation_deg} degrees")
            
            success = True
        else:
            print(f"STL file not found: {stl_file_path}")
    except Exception as e:
        print(f"Error loading STL file: {e}")
    
    return vertices, success

# PLOT 4: Individual Force Vectors with Two STL Models
ax4 = fig.add_subplot(224, projection='3d')

# === STL MODEL POSITIONING CONTROLS FOR PLOT 4 (FIRST MODEL) ===
stl1_file_path = "model.stl"  # First STL file path
stl1_scale_factor = 4.9  # Scale factor for first model
stl1_translation = [0.0, 2.2, 0.0]  # Translation for first model
stl1_rotation_deg = [0, 180, -90]  # Rotation for first model
stl1_face_color = [0.8, 0.8, 0.9]  # Light blue-gray

# === STL MODEL POSITIONING CONTROLS FOR PLOT 4 (SECOND MODEL) ===
stl2_file_path = "second_model.stl"  # Second STL file path
stl2_scale_factor = 4.0  # Scale factor for second model
stl2_translation = [0.2, -1.4, -1.7]  # Translation for second model
stl2_rotation_deg = [180, -60, 90]  # Rotation for second model
stl2_face_color = [0.9, 0.8, 0.8]  # Light red-gray for distinction

# Load the first STL model for plot 4
vertices1, stl1_loaded = load_stl_for_plot(ax4, stl1_file_path, 
                                          stl1_scale_factor, 
                                          stl1_translation, 
                                          stl1_rotation_deg,
                                          alpha=0.3,
                                          face_color=stl1_face_color)

# Load the second STL model for plot 4
vertices2, stl2_loaded = load_stl_for_plot(ax4, stl2_file_path, 
                                          stl2_scale_factor, 
                                          stl2_translation, 
                                          stl2_rotation_deg,
                                          alpha=0.3,
                                          face_color=stl2_face_color)

# If neither STL was loaded, create a wireframe sphere as fallback
if not (stl1_loaded or stl2_loaded):
    sphere_x, sphere_y, sphere_z = create_sphere_mesh(radius=1.0, resolution=12)
    ax4.plot_wireframe(sphere_x, sphere_y, sphere_z, color='gray', alpha=0.2, linewidth=0.5)

# Set plot limits with larger margin to accommodate longer vectors
lim = 1.5
ax4.set_xlim([- lim, lim])
ax4.set_ylim([- lim, lim])
ax4.set_zlim([- lim, lim])

# Plot individual force vectors
for i, (vec, mag) in enumerate(zip(force_vectors, force_magnitudes)):
    # Normalize the direction vector
    normalized_vec = vec / np.linalg.norm(vec)
    
    # Calculate endpoint based on magnitude
    scaled_magnitude = 0.5 + (mag * magnitude_scale)
    endpoint = normalized_vec * scaled_magnitude
    
    # Get color from colormap
    color = cmap(norm(mag))
    
    # Add a smaller sphere at the endpoint (reduced size for clarity when many points)
    ax4.scatter(endpoint[0], endpoint[1], endpoint[2],
                color=color, s=20, alpha=0.7)

# Set plot properties
ax4.set_xlabel('X Direction')
ax4.set_ylabel('Y Direction')
ax4.set_zlabel('Z Direction')
ax4.set_title('Individual Force Vectors with Two STL Models')
ax4.set_box_aspect([1, 1, 1])

# Add colorbar for force magnitude
cbar4 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax4, pad=0.1)
cbar4.set_label('Force Magnitude (N)')

# Add legend for the two STL models
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=stl1_face_color, markersize=10),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=stl2_face_color, markersize=10)
]
ax4.legend(custom_lines, ['First STL Model', 'Second STL Model'], loc='upper right')

# Set equal aspect ratio for 3D plots
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_box_aspect([1, 1, 1])

# Update figure layout
fig = plt.gcf()
fig.set_size_inches(24, 12)

# Update layout
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust spacing between plots

# Save figure with high resolution
plt.savefig('biomechanical_analysis_with_mesh.png', dpi=300, bbox_inches='tight')

print("Analysis complete. Enhanced visualizations created with force vector mesh plot.")
plt.show()