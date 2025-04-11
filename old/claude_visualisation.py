import numpy as np
import json
import os
import time
import glob
from scipy.spatial.transform import Rotation
import pyvista as pv
import matplotlib.pyplot as plt

class KneeTracker:
    def __init__(self, ct_scan_model_path=None):
        """
        Initialize the knee tracker with paths to necessary files.
        
        Args:
            ct_scan_model_path: Path to the segmented CT scan model
        """
        self.femur_markers_ref = None  # Reference position of femur markers
        self.tibia_markers_ref = None  # Reference position of tibia markers
        
        # Initialize with placeholder models by default
        self.create_placeholder_models()
        
        # Load 3D models if provided
        if ct_scan_model_path:
            self.load_models(ct_scan_model_path)
    
    def create_placeholder_models(self):
        """
        Create simple placeholder models for visualization
        """
        # Create simple placeholder meshes for demonstration
        self.femur_model = pv.Cylinder(center=(0, 0, 0.05), direction=(0, 0, 1), 
                                     radius=0.03, height=0.1)
        self.tibia_model = pv.Cylinder(center=(0, 0, -0.05), direction=(0, 0, 1), 
                                     radius=0.025, height=0.1)
        print("Created placeholder models")
    
    def load_models(self, ct_scan_path):
        """
        Load 3D models from the segmented CT scan.
        In a real application, you'd load STL/OBJ/PLY files.
        """
        try:
            # For example, if you have STL files:
            self.femur_model = pv.read(os.path.join(ct_scan_path, "Right_femur.stl"))
            self.tibia_model = pv.read(os.path.join(ct_scan_path, "Right_tibia.stl"))
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            # Fall back to placeholder models
            self.create_placeholder_models()
    
    def set_reference_markers(self, femur_markers, tibia_markers):
        """
        Set the reference positions of the tracking markers.
        
        Args:
            femur_markers: Array of shape (5, 3) for femur marker coordinates
            tibia_markers: Array of shape (5, 3) for tibia marker coordinates
        """
        self.femur_markers_ref = np.array(femur_markers)
        self.tibia_markers_ref = np.array(tibia_markers)
        
        # Calculate the centroid of each marker set in reference pose
        self.femur_centroid_ref = np.mean(self.femur_markers_ref, axis=0)
        self.tibia_centroid_ref = np.mean(self.tibia_markers_ref, axis=0)
    
    def kabsch_algorithm(self, reference_points, current_points):
        """
        Implement the Kabsch algorithm to find rotation and translation.
        
        Args:
            reference_points: The reference configuration points
            current_points: The current configuration points
            
        Returns:
            rotation_matrix: 3x3 rotation matrix
            translation_vector: 3x1 translation vector
        """
        # Center the points
        ref_centroid = np.mean(reference_points, axis=0)
        current_centroid = np.mean(current_points, axis=0)
        
        ref_centered = reference_points - ref_centroid
        current_centered = current_points - current_centroid
        
        # Calculate the covariance matrix
        H = ref_centered.T @ current_centered
        
        # Singular Value Decomposition
        U, _, Vt = np.linalg.svd(H)
        
        # Ensure proper rotation (handle reflection case)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Calculate translation
        t = current_centroid - R @ ref_centroid
        
        return R, t
    
    def process_tracking_data(self, json_file):
        """
        Process tracking data from a JSON file.
        
        Args:
            json_file: Path to the JSON file containing marker coordinates
            
        Returns:
            femur_transform: 4x4 transformation matrix for femur
            tibia_transform: 4x4 transformation matrix for tibia
        """
        # Load tracking data from JSON
        with open(json_file, 'r') as f:
            tracking_data = json.load(f)
        
        # Extract marker positions
        # Assuming the JSON structure contains 'femur_markers' and 'tibia_markers'
        # Each with 5 points with x, y, z coordinates
        femur_markers_current = np.array(tracking_data['femur_markers'])
        tibia_markers_current = np.array(tracking_data['tibia_markers'])
        
        # Calculate transformation using Kabsch algorithm
        femur_R, femur_t = self.kabsch_algorithm(self.femur_markers_ref, femur_markers_current)
        tibia_R, tibia_t = self.kabsch_algorithm(self.tibia_markers_ref, tibia_markers_current)
        
        # Create 4x4 transformation matrices
        femur_transform = np.eye(4)
        femur_transform[:3, :3] = femur_R
        femur_transform[:3, 3] = femur_t
        
        tibia_transform = np.eye(4)
        tibia_transform[:3, :3] = tibia_R
        tibia_transform[:3, 3] = tibia_t
        
        return femur_transform, tibia_transform
    
    def calculate_joint_kinematics(self, femur_transform, tibia_transform):
        """
        Calculate knee joint kinematics parameters.
        
        Args:
            femur_transform: 4x4 transformation matrix for femur
            tibia_transform: 4x4 transformation matrix for tibia
            
        Returns:
            Dictionary containing knee joint parameters
        """
        # Calculate relative motion of tibia with respect to femur
        # Tfemur^-1 * Ttibia
        femur_transform_inv = np.linalg.inv(femur_transform)
        relative_transform = femur_transform_inv @ tibia_transform
        
        # Extract rotation matrix
        rotation_matrix = relative_transform[:3, :3]
        
        # Convert to Euler angles (in anatomical terms: flexion-extension, 
        # abduction-adduction, internal-external rotation)
        r = Rotation.from_matrix(rotation_matrix)
        angles_deg = r.as_euler('xyz', degrees=True)
        
        # Extract translation (anterior-posterior, medial-lateral, proximal-distal)
        translation = relative_transform[:3, 3]
        
        return {
            'flexion_deg': angles_deg[0],
            'abduction_deg': angles_deg[1],
            'rotation_deg': angles_deg[2],
            'anterior_mm': translation[0] * 1000,  # Convert to mm
            'medial_mm': translation[1] * 1000,
            'proximal_mm': translation[2] * 1000
        }
    
    def visualize_motion(self, json_directory, output_video=None, replay_speed=1.0):
        """
        Visualize the knee motion from a sequence of JSON files.
        
        Args:
            json_directory: Directory containing tracking JSON files
            output_video: Path to save the visualization as a video (optional)
            replay_speed: Speed multiplier for replay (default: 1.0)
        """
        # Check if reference markers are set
        if self.femur_markers_ref is None or self.tibia_markers_ref is None:
            raise ValueError("Reference markers not set. Call set_reference_markers first.")
        
        # Make sure we have models to visualize
        if self.femur_model is None or self.tibia_model is None:
            self.create_placeholder_models()
        
        # Get all JSON files in the directory, sorted by timestamp
        json_files = sorted(glob.glob(os.path.join(json_directory, "*.json")))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {json_directory}")
        
        # Set up visualization
        plotter = pv.Plotter(off_screen=output_video is not None)
        
        # Add a coordinate system for reference
        plotter.add_axes()
        
        # Add knee models
        femur_actor = plotter.add_mesh(self.femur_model, color='ivory', opacity=0.8)
        tibia_actor = plotter.add_mesh(self.tibia_model, color='lightblue', opacity=0.8)
        
        # Create PolyData objects for markers - PyVista requires this for points
        femur_points = pv.PolyData(self.femur_markers_ref)
        tibia_points = pv.PolyData(self.tibia_markers_ref)
        
        # Add markers
        femur_markers_actor = plotter.add_mesh(femur_points, color='red', point_size=10, render_points_as_spheres=True)
        tibia_markers_actor = plotter.add_mesh(tibia_points, color='blue', point_size=10, render_points_as_spheres=True)
        
        # Set up camera
        plotter.camera_position = [(0, -0.5, 0), (0, 0, 0), (0, 0, 1)]
        plotter.show(auto_close=False)
        
        # Create plot for kinematics
        plt.figure(figsize=(10, 6))
        flexion_data = []
        time_points = []
        
        # Process each frame
        for i, json_file in enumerate(json_files):
            # Process tracking data
            femur_transform, tibia_transform = self.process_tracking_data(json_file)
            
            # Update femur model
            femur_model_transformed = self.femur_model.copy()
            femur_model_transformed.transform(femur_transform)
            plotter.update_coordinates(femur_actor, femur_model_transformed.points, render=False)
            
            # Update tibia model
            tibia_model_transformed = self.tibia_model.copy()
            tibia_model_transformed.transform(tibia_transform)
            plotter.update_coordinates(tibia_actor, tibia_model_transformed.points, render=False)
            
            # Update marker positions
            with open(json_file, 'r') as f:
                tracking_data = json.load(f)
            
            femur_markers_current = np.array(tracking_data['femur_markers'])
            tibia_markers_current = np.array(tracking_data['tibia_markers'])
            
            # Create new PolyData for current marker positions
            new_femur_points = pv.PolyData(femur_markers_current)
            new_tibia_points = pv.PolyData(tibia_markers_current)
            
            # Remove previous marker actors and add new ones
            plotter.remove_actor(femur_markers_actor)
            plotter.remove_actor(tibia_markers_actor)
            femur_markers_actor = plotter.add_mesh(new_femur_points, color='red', 
                                                point_size=10, render_points_as_spheres=True, render=False)
            tibia_markers_actor = plotter.add_mesh(new_tibia_points, color='blue', 
                                               point_size=10, render_points_as_spheres=True, render=False)
            
            # Calculate and display kinematics
            kinematics = self.calculate_joint_kinematics(femur_transform, tibia_transform)
            
            # Update plot data
            flexion_data.append(kinematics['flexion_deg'])
            time_points.append(i)
            
            # Display frame info
            plotter.add_text(f"Frame: {i+1}/{len(json_files)}\n" + 
                           f"Flexion: {kinematics['flexion_deg']:.1f}°\n" + 
                           f"Abduction: {kinematics['abduction_deg']:.1f}°\n" + 
                           f"Rotation: {kinematics['rotation_deg']:.1f}°", 
                           position='upper_left', font_size=12, render=False)
            
            plotter.render()
            
            if output_video:
                plotter.screenshot(f"frame_{i:04d}.png")
            
            # Control playback speed
            time.sleep(0.1 / replay_speed)
        
        plotter.close()
        
        # Create and show the flexion angle plot
        plt.plot(time_points, flexion_data)
        plt.title('Knee Flexion Angle Over Time')
        plt.xlabel('Frame Number')
        plt.ylabel('Flexion Angle (degrees)')
        plt.grid(True)
        plt.savefig('flexion_curve.png')
        plt.show()
        
        # Combine frames into video if requested
        if output_video:
            import cv2
            frames = sorted(glob.glob("frame_*.png"))
            if frames:
                img = cv2.imread(frames[0])
                h, w, layers = img.shape
                
                video = cv2.VideoWriter(output_video, 
                                       cv2.VideoWriter_fourcc(*'XVID'), 
                                       10 * replay_speed, (w, h))
                
                for frame in frames:
                    video.write(cv2.imread(frame))
                
                video.release()
                
                # Clean up temporary files
                for frame in frames:
                    os.remove(frame)


# Example usage:
if __name__ == "__main__":
    # Create a knee tracker instance
    tracker = KneeTracker()
    
    # Create example reference marker positions
    # These would normally come from your CT scan or initial pose
    femur_markers_ref = [
        [0.00, 0.03, 0.10],
        [0.03, 0.00, 0.08],
        [-0.03, 0.00, 0.08],
        [0.00, -0.03, 0.10],
        [0.00, 0.00, 0.13]
    ]
    
    tibia_markers_ref = [
        [0.00, 0.02, -0.05],
        [0.02, 0.00, -0.07],
        [-0.02, 0.00, -0.07],
        [0.00, -0.02, -0.05],
        [0.00, 0.00, -0.10]
    ]
    
    # Set reference marker positions
    tracker.set_reference_markers(femur_markers_ref, tibia_markers_ref)
    
    # Example - Generate sample movement data
    # In real application, you'd use your OptiTrack JSON files
    def generate_sample_data(num_frames=50):
        os.makedirs("sample_data", exist_ok=True)
        
        for i in range(num_frames):
            # Simulate flexion movement (angle increases over time)
            angle_rad = (i / num_frames) * np.pi / 3  # 0 to 60 degrees
            
            # Calculate new positions for tibia markers (rotation around x-axis)
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
            
            # Apply rotation to tibia markers
            tibia_markers_current = np.array([rotation_matrix @ marker for marker in tibia_markers_ref])
            
            # Add small random noise to make it realistic
            noise = np.random.normal(0, 0.0005, tibia_markers_current.shape)
            tibia_markers_current += noise
            
            # Keep femur markers fixed with small noise
            femur_markers_current = np.array(femur_markers_ref) + np.random.normal(0, 0.0005, (5, 3))
            
            # Create and save the JSON file
            data = {
                "timestamp": i * 0.1,
                "femur_markers": femur_markers_current.tolist(),
                "tibia_markers": tibia_markers_current.tolist()
            }
            
            with open(f"sample_data/frame_{i:04d}.json", "w") as f:
                json.dump(data, f)
    
    # Generate sample data
    generate_sample_data(50)
    
    # Visualize the motion
    tracker.visualize_motion("sample_data", output_video="knee_motion.avi", replay_speed=0.5)