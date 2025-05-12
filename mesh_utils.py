
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

class MeshUtils:
    @staticmethod
    def load_stl_as_mesh(filename):
        """Load an STL file and return vertices and faces for PyQtGraph GLMeshItem"""
        try:
            stl_mesh = mesh.Mesh.from_file(filename) # Load the STL file
            vertices = stl_mesh.vectors.reshape(-1, 3) # Get vertices (each face has 3 vertices)
            faces = np.arange(len(vertices)).reshape(-1, 3) # Create faces array - each triplet of vertices forms a face
            return vertices, faces
        except Exception as e:
            print(f"Error loading STL file {filename}: {e}")
            # Return simple cube as fallback
            vertices = np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
            ])
            faces = np.array([
                [0, 1, 2], [0, 2, 3], [0, 1, 5], [0, 5, 4],
                [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6],
                [3, 0, 4], [3, 4, 7], [4, 5, 6], [4, 6, 7]
            ])
            return vertices, faces
    
    @staticmethod
    def quaternion_to_transform_matrix(quaternion, position=None):
        """
        Convert a quaternion and position to a 4x4 transformation matrix.
        Args:
            quaternion: A numpy array or list with 4 elements representing the quaternion [w, x, y, z]
            position: A numpy array or list with 3 elements representing the position [x, y, z]
                     If None, no translation is applied.
        Returns:
            A 4x4 numpy array representing the transformation matrix
        """
        # Normalize quaternion
        q = np.array(quaternion)
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        
        # Create 4x4 transformation matrix with rotation from quaternion
        T = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y, 0],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x, 0],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y, 0],
            [0, 0, 0, 1]
        ])
        
        # Apply translation if provided
        if position is not None:
            T[0:3, 3] = position
        
        return T
    
    @staticmethod
    def create_arrow(start_point, end_point, color=(1,0,0,1), arrow_size=15.0, shaft_width=2.0):
        """Create a 3D arrow from start_point to end_point"""
        # If the points are too close, just return None
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        if length < 0.01:
            return None, None
            
        # Normalize direction
        direction = direction / length
        
        # Create shaft points (reduce length to leave room for arrowhead)
        shaft_length = length * 0.85
        shaft_end = start_point + direction * shaft_length
        shaft_points = np.array([start_point, shaft_end])
        
        # Create shaft line
        shaft = gl.GLLinePlotItem(pos=shaft_points, color=color, width=shaft_width, antialias=True)
        
        try:
            # Create the cone for the arrowhead using the built-in function
            md = gl.MeshData.cylinder(rows=10, cols=40, radius=[0, arrow_size], length=length*0.15)
            
            # Get vertices and faces
            vertices = md.vertexes()
            faces = md.faces()
            
            # Create a rotation matrix to orient the arrow along the direction vector
            z_axis = np.array([0, 0, -1])
            
            # Handle special cases where cross product might fail
            if np.allclose(direction, z_axis, rtol=1e-5, atol=1e-5):
                # Direction is already aligned with z-axis
                rotation_matrix = np.eye(3)
            elif np.allclose(direction, -z_axis, rtol=1e-5, atol=1e-5):
                # Direction is opposite to z-axis
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ])
            else:
                # Normal case - calculate rotation axis and angle
                rotation_axis = np.cross(z_axis, direction)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                
                angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
                
                # Rodrigues rotation formula for rotation matrix
                K = np.array([
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]
                ])
                rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            
            # Transform vertices
            transformed_vertices = np.dot(vertices, rotation_matrix.T)
            transformed_vertices += shaft_end
            
            # Create mesh item with transformed vertices
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
            # If arrow head creation fails, return only the shaft
            return shaft, None
    
    def get_tibia_force_origin():
        """Get the specific point on the tibia where the force arrow should originate"""
        # platzhalter für später
        base_position = np.array([0, 0, 100])
        
        # Define anatomical offset - these values should be adjusted to match your specific model
        anatomical_offset = np.array([0, 0, 0])  # X, Y, Z offset in model coordinates
        
        # Return the origin point
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

        t = centroid_q - np.dot(centroid_p, R.T)

        return t, R
    
    @staticmethod
    def update_mesh_with_data(mesh, pivot_point, position, quaternion):
        """
        Update a mesh with position and rotation data around a specific pivot point.
        Args:
            mesh: The mesh to update
            pivot_point: The pivot point for rotation (numpy array)
            position: The final position (numpy array)
            quaternion: The rotation quaternion
        """
        # Get transformation matrix with rotation only, no translation
        R_matrix = MeshUtils.quaternion_to_transform_matrix(quaternion)
        
        # Create translation matrices
        T_to_origin = np.eye(4)
        T_to_origin[0:3, 3] = -pivot_point  # Move pivot point to origin 
        
        T_from_origin = np.eye(4)
        T_from_origin[0:3, 3] = pivot_point  # Move back from origin
        
        T_position = np.eye(4)
        T_position[0:3, 3] = position  # Final position
        
        # Apply combined transform: translate to origin, rotate, translate back, then to final position
        transform = np.dot(T_position, np.dot(T_from_origin, np.dot(R_matrix, T_to_origin)))
        
        # Update the mesh transformation
        mesh.setTransform(transform)