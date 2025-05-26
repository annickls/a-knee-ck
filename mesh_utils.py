import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pyqtgraph.opengl as gl
from stl import mesh
from pyqtgraph.Qt import QtGui
from OpenGL.GL import glBegin, glEnd, glVertex3f, glColor4f, GL_LINES, GL_LINE_SMOOTH, glEnable, glHint, GL_LINE_SMOOTH_HINT, GL_NICEST
import pyqtgraph.opengl as gl
import yaml
import constants
import numpy as np
import warnings

class MeshUtils:
    # Add class variables to store landmark positions for angle calculations
    tibia_landmarks = {}
    femur_landmarks = {}
    current_knee_angles = {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
    
    @staticmethod
    def load_stl_as_mesh(filename):
        """Load an STL file and return vertices and faces for PyQtGraph GLMeshItem"""
        try:
            stl_mesh = mesh.Mesh.from_file(filename)
            vertices = stl_mesh.vectors.reshape(-1, 3)
            faces = np.arange(len(vertices)).reshape(-1, 3)
            return vertices, faces
        except Exception as e:
            print(f"Error loading STL file {filename}: {e}")
    
    @staticmethod
    def quaternion_to_transform_matrix(quaternion, position=None):
        """Convert a quaternion and position to a 4x4 transformation matrix."""
        q = np.array(quaternion)
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        
        T = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y, 0],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x, 0],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y, 0],
            [0, 0, 0, 1]
        ])
        
        if position is not None:
            T[0:3, 3] = position
        
        return T
    
    @staticmethod
    def update_landmark_alex(self, landmark_position, landmark_name):
        """
        Update landmark position and calculate Grood and Suntay knee angles.
        
        Args:
            landmark_position: 3D position of the landmark (as already calculated and visualized)
            landmark_name: Name of the landmark (e.g., "tibia_medial", "femur_lateral", etc.)
        """
        # Store the landmark position directly (since it's already correctly calculated)
        landmark_data = {
            'position': np.array(landmark_position)
        }
        
        if landmark_name.startswith('tibia'):
            MeshUtils.tibia_landmarks[landmark_name] = landmark_data
        elif landmark_name.startswith('femur'):
            MeshUtils.femur_landmarks[landmark_name] = landmark_data
        else:
            print(f"Warning: Unknown landmark type {landmark_name}")
            return
        print("test")

        # Calculate knee angles if we have sufficient landmarks
        if MeshUtils._has_required_landmarks():
            angles = MeshUtils.calculate_grood_suntay_angles()
            MeshUtils.current_knee_angles = angles
            
            # Print or log the angles for debugging
            print(f"Knee Angles - Flexion: {angles['flexion']:.2f}°, "
                  f"Adduction: {angles['adduction']:.2f}°, "
                  f"Internal Rotation: {angles['rotation']:.2f}°")
    
    @staticmethod
    def _has_required_landmarks():
        """Check if we have the minimum required landmarks for angle calculation."""
        required_tibia = ['tibia_medial', 'tibia_lateral', 'tibia_proximal', 'tibia_distal']
        required_femur = ['femur_medial', 'femur_lateral', 'femur_proximal', 'femur_distal']
        
        tibia_available = all(landmark in MeshUtils.tibia_landmarks for landmark in required_tibia)
        femur_available = all(landmark in MeshUtils.femur_landmarks for landmark in required_femur)
        
        return tibia_available and femur_available
    
    @staticmethod
    def calculate_grood_suntay_angles():
        """
        Calculate knee angles using the Grood and Suntay method.
        
        Returns:
            dict: Dictionary containing flexion, adduction, and rotation angles in degrees
        """
        try:
            # Get landmark positions directly (already transformed and correct)
            tibia_medial = MeshUtils.tibia_landmarks['tibia_medial']['position']
            tibia_lateral = MeshUtils.tibia_landmarks['tibia_lateral']['position']
            tibia_proximal = MeshUtils.tibia_landmarks['tibia_proximal']['position']
            tibia_distal = MeshUtils.tibia_landmarks['tibia_distal']['position']
            
            femur_medial = MeshUtils.femur_landmarks['femur_medial']['position']
            femur_lateral = MeshUtils.femur_landmarks['femur_lateral']['position']
            femur_proximal = MeshUtils.femur_landmarks['femur_proximal']['position']
            femur_distal = MeshUtils.femur_landmarks['femur_distal']['position']
            
            # Debug: Print landmark positions to verify they're different
            print(f"Debug - Tibia medial: {tibia_medial}")
            print(f"Debug - Tibia lateral: {tibia_lateral}")
            print(f"Debug - Femur medial: {femur_medial}")
            print(f"Debug - Femur lateral: {femur_lateral}")
            
            # Define coordinate systems according to Grood and Suntay
            
            # Femoral coordinate system
            # e1f: femoral flexion-extension axis (lateral - medial direction)
            e1f = femur_lateral - femur_medial
            if np.linalg.norm(e1f) < 1e-10:
                print("Warning: Femur medial-lateral vector is too small")
                return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
            e1f = e1f / np.linalg.norm(e1f)
            
            # Temporary femoral long axis (proximal - distal direction)
            temp_femur = femur_proximal - femur_distal
            if np.linalg.norm(temp_femur) < 1e-10:
                print("Warning: Femur proximal-distal vector is too small")
                return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
            temp_femur = temp_femur / np.linalg.norm(temp_femur)
            
            # e3f: femoral anterior-posterior axis (perpendicular to e1f and temp_femur)
            e3f = np.cross(e1f, temp_femur)
            if np.linalg.norm(e3f) < 1e-10:
                print("Warning: Femur coordinate system is degenerate")
                return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
            e3f = e3f / np.linalg.norm(e3f)
            
            # e2f: femoral long axis (corrected, perpendicular to e3f and e1f)
            e2f = np.cross(e3f, e1f)
            e2f = e2f / np.linalg.norm(e2f)
            
            # Tibial coordinate system
            # e2t: tibial long axis (proximal - distal direction)
            e2t = tibia_proximal - tibia_distal
            if np.linalg.norm(e2t) < 1e-10:
                print("Warning: Tibia proximal-distal vector is too small")
                return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
            e2t = e2t / np.linalg.norm(e2t)
            
            # Temporary tibial medial-lateral axis
            temp_tibia = tibia_lateral - tibia_medial
            if np.linalg.norm(temp_tibia) < 1e-10:
                print("Warning: Tibia medial-lateral vector is too small")
                return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
            temp_tibia = temp_tibia / np.linalg.norm(temp_tibia)
            
            # e3t: tibial anterior-posterior axis
            e3t = np.cross(temp_tibia, e2t)
            if np.linalg.norm(e3t) < 1e-10:
                print("Warning: Tibia coordinate system is degenerate")
                return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
            e3t = e3t / np.linalg.norm(e3t)
            
            # e1t: tibial medial-lateral axis (corrected)
            e1t = np.cross(e2t, e3t)
            e1t = e1t / np.linalg.norm(e1t)
            
            # Calculate floating axis (common perpendicular to e1f and e2t)
            floating_axis = np.cross(e1f, e2t)
            if np.linalg.norm(floating_axis) < 1e-10:
                print("Warning: Floating axis is degenerate")
                return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
            floating_axis = floating_axis / np.linalg.norm(floating_axis)
            
            # Calculate Grood and Suntay angles using rotation matrix decomposition
            
            # Create rotation matrices for femur and tibia coordinate systems
            R_femur = np.column_stack([e1f, e2f, e3f])
            R_tibia = np.column_stack([e1t, e2t, e3t])
            
            # Relative rotation matrix from femur to tibia
            R_rel = R_tibia.T @ R_femur
            
            # Extract Grood and Suntay angles from rotation matrix
            # Following the ZXY Euler angle sequence used in Grood and Suntay
            
            # Flexion (rotation about femoral medial-lateral axis)
            flexion = np.arcsin(-R_rel[1, 2])
            flexion_deg = np.degrees(flexion)
            
            # Adduction (rotation about floating axis)
            cos_adduction = R_rel[2, 2] / np.cos(flexion)
            cos_adduction = np.clip(cos_adduction, -1.0, 1.0)
            adduction = np.arccos(cos_adduction)
            if R_rel[0, 2] < 0:
                adduction = -adduction
            adduction_deg = np.degrees(adduction)
            
            # Internal rotation (rotation about tibial long axis)
            cos_rotation = R_rel[1, 1] / np.cos(flexion)
            cos_rotation = np.clip(cos_rotation, -1.0, 1.0)
            rotation = np.arccos(cos_rotation)
            if R_rel[1, 0] < 0:
                rotation = -rotation
            rotation_deg = np.degrees(rotation)
            
            return {
                'flexion': flexion_deg,
                'adduction': adduction_deg,
                'rotation': rotation_deg
            }
            
        except Exception as e:
            print(f"Error calculating Grood and Suntay angles: {e}")
            import traceback
            traceback.print_exc()
            return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
    
    @staticmethod
    def get_current_knee_angles():
        """Get the current knee angles."""
        return MeshUtils.current_knee_angles.copy()
    
    @staticmethod
    def reset_landmarks():
        """Reset all stored landmarks."""
        MeshUtils.tibia_landmarks.clear()
        MeshUtils.femur_landmarks.clear()
        MeshUtils.current_knee_angles = {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
    
    @staticmethod
    def create_arrow(start_point, end_point, color=(1,0,0,1), arrow_size=15.0, shaft_width=2.0):
        """Create a 3D arrow from start_point to end_point"""
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        if length < 0.01:
            return None, None
            
        direction = direction / length
        shaft_length = length * 0.85
        shaft_end = start_point + direction * shaft_length
        shaft_points = np.array([start_point, shaft_end])
        
        shaft = gl.GLLinePlotItem(pos=shaft_points, color=color, width=shaft_width, antialias=True)
        
        try:
            md = gl.MeshData.cylinder(rows=10, cols=40, radius=[0, arrow_size], length=length*0.15)
            vertices = md.vertexes()
            faces = md.faces()
            
            z_axis = np.array([0, 0, -1])
            
            if np.allclose(direction, z_axis, rtol=1e-5, atol=1e-5):
                rotation_matrix = np.eye(3)
            elif np.allclose(direction, -z_axis, rtol=1e-5, atol=1e-5):
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ])
            else:
                rotation_axis = np.cross(z_axis, direction)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                
                angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
                
                K = np.array([
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]
                ])
                rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            
            transformed_vertices = np.dot(vertices, rotation_matrix.T)
            transformed_vertices += shaft_end
            
            head = gl.GLMeshItem(
                vertexes=transformed_vertices, 
                faces=faces, 
                smooth=False, 
                color=color, 
                shader='balloon'
            )

            return shaft, head
        except Exception as e:
            print(f"Error creating arrow head: {e}")
            return shaft, None
    
    def get_tibia_force_origin(tibia_position):
        """Get the specific point on the tibia where the force arrow should originate"""
        base_position = np.array(tibia_position)
        anatomical_offset = np.array([0, 0, 100])
        return base_position + anatomical_offset
    
    @staticmethod
    def kabsch(filePath, bone):
        """Calculate the optimal rigid transformation matrix from Q -> P using Kabsch algorithm"""
        with open(filePath, "r") as file:
            content = yaml.safe_load(file)

        def readYaml(marker):
            array = np.array([])
            for i in range(5):
                array = np.append(array, [content[marker][i]["x"], content[marker][i]["y"], content[marker][i]["z"]])
            array = array.reshape([5,3])
            return array

        bone_ref = readYaml(bone+"_ref")
        bone_slicer = readYaml(bone+"_slicer")

        q = bone_ref
        p = bone_slicer

        centroid_p = np.mean(p, axis=0)
        centroid_q = np.mean(q, axis=0)

        p_centered = p - centroid_p
        q_centered = q - centroid_q

        H = np.dot(p_centered.T, q_centered)

        U, _,  vt = np.linalg.svd(H)

        R = np.dot(vt.T, U.T)

        if np.linalg.det(R) < 0:
            vt[-1, :] *= -1
            R = np.dot(vt.T, U.T)

        t = centroid_q - centroid_p

        return t, R
    
    @staticmethod
    def update_mesh_with_data(mesh, position, quaternion):
        """Update a mesh with position and rotation data."""
        R_matrix = MeshUtils.quaternion_to_transform_matrix(quaternion)
        transform = R_matrix.copy()
        transform[0:3, 3] = position
        
        T_current = MeshUtils.quaternion_to_transform_matrix(quaternion, position*1000)
        transform = T_current
        
        mesh.setTransform(transform)