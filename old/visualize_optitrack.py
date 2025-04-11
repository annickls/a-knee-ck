import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time

# Load JSON data
with open("optitrack_data.json", "r") as file:
    data = json.load(file)

# Prepare Matplotlib 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("OptiTrack Animated 3D Visualization")

# Prepare Open3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window()
cloud = o3d.geometry.PointCloud()
vis.add_geometry(cloud)

# Animate through timestamps
for frame in data:
    positions = []
    labels = []

    for rb in frame["rigid_bodies"]:
        x, y, z = rb["position"]["x"], rb["position"]["y"], rb["position"]["z"]
        positions.append([x, y, z])
        labels.append(rb["name"])

    positions = np.array(positions)

    # Clear previous Matplotlib plot
    ax.cla()
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c="r", marker="o")

    for i, label in enumerate(labels):
        ax.text(positions[i, 0], positions[i, 1], positions[i, 2], label, fontsize=12)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"OptiTrack Frame: {frame['frame']}")

    plt.pause(0.5)  # Pause to simulate real-time update

    # Update Open3D visualization
    cloud.points = o3d.utility.Vector3dVector(positions)
    vis.update_geometry(cloud)
    vis.poll_events()
    vis.update_renderer()

    time.sleep(0.5)  # Simulate real-time update delay

plt.show()
vis.destroy_window()
