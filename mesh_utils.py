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