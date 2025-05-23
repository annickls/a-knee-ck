import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFrame, 
                            QProgressBar, QGridLayout, QSplitter, QTabWidget, QSlider, QGroupBox, QTextEdit)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
import matplotlib.cm as cm
import pyqtgraph.opengl as gl
from stl import mesh
from pyqtgraph.Qt import QtGui



import time
import datetime
from OpenGL.GL import glBegin, glEnd, glVertex3f, glColor4f, GL_LINES, GL_LINE_SMOOTH, glEnable, glHint, GL_LINE_SMOOTH_HINT, GL_NICEST
import pyqtgraph.opengl as gl
import constants
import pandas as pd
import numpy as np
import csv
from pathlib import Path


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

class BoneDataGenerator:
    def __init__(self):
        # Fixed femur position and orientation
        self.femur_position = np.array([0.0, 50.0, 20.0])
        self.femur_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        
        # Initial tibia position and orientation (will be updated during animation)
        self.tibia_position = np.array([0.0, 0.0, -100.0])
        self.tibia_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Define joint center (pivot point) relative to femur
        self.joint_center = np.array([-440.0, 70.0, 150.0])
        
        # Define rotation axis (medio-lateral axis for flexion/extension)
        self.rotation_axis = np.array([0.0, 1.0, 0.0])  # Y-axis for flexion/extension
        
        # Animation parameters
        self.time = 0
        self.freq = 0.2  # Hz
        self.max_angle = 90  # degrees
    
    def update(self, dt):
        """Update bone positions with tibia rotating around fixed femur"""
        self.time += dt
        
        # Calculate current flexion angle based on time
        flexion_angle = self.max_angle * (np.sin(2 * np.pi * self.freq * self.time) + 1) / 2
        angle_rad = np.radians(flexion_angle)
        
        # Calculate rotation quaternion for tibia (around X-axis)
        sin_half = np.sin(angle_rad/2)
        cos_half = np.cos(angle_rad/2)
        
        # Quaternion for rotation around Y-axis (medio-lateral for flexion/extension)
        self.tibia_quaternion = np.array([cos_half, 0.0, sin_half, 0.0])  # Y-axis rotation
        
        # Offset from joint center when fully extended
        offset_length = 100.0  # Length of tibia from joint center
        
        # Calculate new position based on rotation
        self.tibia_position = np.array([
            self.joint_center[0] + offset_length * np.sin(angle_rad),  # X changes with rotation
            self.joint_center[1],                                       # Y stays aligned with femur
            self.joint_center[2] - offset_length * np.cos(angle_rad)   # Z changes with rotation
        ])
        
        # Generate forces and torques based on current angle
        force_magnitude = 5.0 + 8.0 * abs(np.sin(angle_rad))
        force_dir = np.array([
            np.sin(angle_rad) * np.sin(self.time * 0.5),
            np.cos(angle_rad * 0.7),
            -np.sin(self.time * 0.3)
        ])
        force_dir = force_dir / np.linalg.norm(force_dir)  # Normalize
        force = force_dir * force_magnitude
        
        # Torque also changes with angle
        torque_magnitude = 1.0 + 1.5 * abs(np.cos(angle_rad))
        torque_dir = np.array([
            np.sin(self.time * 0.4),
            np.cos(angle_rad),
            np.sin(angle_rad * 0.5)
        ])
        torque_dir = torque_dir / np.linalg.norm(torque_dir)  # Normalize
        torque = torque_dir * torque_magnitude
        
        return {
            'femur_position': self.femur_position,
            'femur_quaternion': self.femur_quaternion,
            'tibia_position': self.tibia_position,
            'tibia_quaternion': self.tibia_quaternion,
            'force': force,
            'torque': torque
        }

class MplCanvas(FigureCanvas):
    """Matplotlib canvas class for embedding plots in Qt that can display either current or historical force/torque data"""
    def __init__(self, width=5, height=4, mode="current"):
        self.fig = Figure(figsize=(width, height))
        self.axes_force = self.fig.add_subplot(121, projection='3d')
        self.axes_torque = self.fig.add_subplot(122, projection='3d')
        self.mode = mode  # "current" or "history"
        
        # Set up axes once
        force_max = constants.FORCE_MAX #if mode == "current" else 11
        torque_max = constants.TORQUE_MAX
        
        # Force plot setup
        self.axes_force.set_xlim([-force_max, force_max])
        self.axes_force.set_ylim([-force_max, force_max])
        self.axes_force.set_zlim([-force_max, force_max])
        #self.axes_force.set_title('Current Force (N)' if mode == "current" else 'Force History (N)')
        self.axes_force.grid(False)
        self.axes_force.set_axis_off()
        
        # Torque plot setup
        self.axes_torque.set_xlim([-torque_max, torque_max])
        self.axes_torque.set_ylim([-torque_max, torque_max])
        self.axes_torque.set_zlim([-torque_max, torque_max])
        #self.axes_torque.set_title('Current Torque (Nm)' if mode == "current" else 'Torque History (Nm)')
        self.axes_torque.grid(False)
        self.axes_torque.set_axis_off()
        
        # Reference axes (drawn once)
        self.ref_axes_force = [
            self.axes_force.quiver(0, 0, 0, force_max*constants.AXIS_FACTOR, 0, 0, color='salmon', linewidth=constants.AXIS_LINEWIDTH, arrow_length_ratio=0.1),
            self.axes_force.quiver(0, 0, 0, 0, force_max*constants.AXIS_FACTOR, 0, color='limegreen', linewidth=constants.AXIS_LINEWIDTH, arrow_length_ratio=0.1),
            self.axes_force.quiver(0, 0, 0, 0, 0, force_max*constants.AXIS_FACTOR, color='deepskyblue', linewidth=constants.AXIS_LINEWIDTH, arrow_length_ratio=0.1)
        ]
        self.ref_axes_torque = [
            self.axes_torque.quiver(0, 0, 0, torque_max*constants.AXIS_FACTOR, 0, 0, color='salmon', linewidth=constants.AXIS_LINEWIDTH, arrow_length_ratio=0.1),
            self.axes_torque.quiver(0, 0, 0, 0, torque_max*constants.AXIS_FACTOR, 0, color='limegreen', linewidth=constants.AXIS_LINEWIDTH, arrow_length_ratio=0.1),
            self.axes_torque.quiver(0, 0, 0, 0, 0, torque_max*constants.AXIS_FACTOR, color='deepskyblue', linewidth=constants.AXIS_LINEWIDTH, arrow_length_ratio=0.1)
        ]
        
        # For history mode: maintain a list of arrows
        if mode == "history":
            self.force_arrows = []
            self.torque_arrows = []
        
        # Text elements for magnitudes and components
        self.force_mag_text = self.axes_force.text2D(0.32, 1.0, "", transform=self.axes_force.transAxes)
        self.torque_mag_text = self.axes_torque.text2D(0.4, 1.0, "", transform=self.axes_torque.transAxes)
        self.force_comp_text = self.axes_force.text2D(0.32, 0.95, "", transform=self.axes_force.transAxes, fontsize=8)
        self.torque_comp_text = self.axes_torque.text2D(0.4, 0.95, "", transform=self.axes_torque.transAxes, fontsize=8)
        
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

class ColoredGLAxisItem(gl.GLAxisItem):
    def __init__(self, size=(1,1,1)):
        gl.GLAxisItem.__init__(self)
        self.setSize(*size)
        
    def paint(self):
        self.setupGLState()
        
        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            
        glBegin(GL_LINES)

        # X axis (red)
        glColor4f(*constants.SALMON)
        glVertex3f(0, 0, 0)
        glVertex3f(self.size()[0], 0, 0)
        
        # Y axis (green)
        glColor4f(*constants.LIMEGREEN) 
        glVertex3f(0, 0, 0)
        glVertex3f(0, self.size()[1], 0)
        
        # Z axis (blue)
        glColor4f(*constants.DEEPSKYBLUE)  # deepskyblue
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, self.size()[2])
        
        glEnd()

class KneeFlexionExperiment(QMainWindow):
    def __init__(self):
        super().__init__()

        
        # Configuration
        self.setWindowTitle("Knee Test Bench with Force Visualization")
        self.setGeometry(100, 100, 1200, 800)

        
        # Initialize variables
        self.timercsv = QTimer()
        self.timercsv.timeout.connect(self.read_csv_data)
        self.monitoring = False
        self.csv_path = "data.csv"
        self.last_modified_time = 0
        self.last_size = 0

        
        # Experiment parameters
        self.current_angle_index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.rotation_complete)
        self.seconds_timer = QTimer()
        self.seconds_timer.timeout.connect(self.update_seconds_progress)
        
        # Timer for visualization updates (every 100ms)
        self.viz_timer = QTimer()
        self.viz_timer.timeout.connect(self.update_visualization_timer)
        self.viz_timer.setInterval(30)  # 100ms for smoother updates
        
        # History for visualization
        self.force_history = []
        self.torque_history = []
        self.current_data_index = 0
        
        # Experiment is running flag
        self.experiment_running = False
        
        # Load force/torque data
        self.load_force_torque_data()
        
        # Setup UI
        self.setup_ui()
        
        # Update visualization initially
        self.update_visualization(0)

        self.recording = False
        self.current_recording_data = []
        self.recording_start_time = None
        self.current_test_name = ""
        
        #self.force_x_data = []
        
        # Ensure directory exists for data files
        os.makedirs("recorded_data", exist_ok=True)



    #test csv
    def toggle_monitoring(self):
        if not self.monitoring:
            # Start monitoring real data
            self.monitoring = True
            self.start_buttoncsv.setText("Stop Real-Time Data")
            print("--- Real-Time Data Monitoring Started ---")
            
            # Initialize file stats
            csv_file = Path(self.csv_path)
            if csv_file.exists():
                self.last_modified_time = csv_file.stat().st_mtime
                self.last_size = csv_file.stat().st_size
                self.read_csv_data()  # Read initial data
            else:
                print(f"Error: {self.csv_path} not found!")
                self.toggle_monitoring()  # Stop monitoring
                return
                
            # Set experiment running flag to true to enable visualization updates
            self.experiment_running = True
                
            # Disable bone animation when using real data
            if hasattr(self, 'auto_bone_movement') and self.auto_bone_movement.isChecked():
                self.auto_bone_movement.click()  # Turn off automatic bone animation
                
            # Stop visualization timer which generates dummy data
            if hasattr(self, 'viz_timer') and self.viz_timer.isActive():
                self.viz_timer.stop()
                
            # Start timer to check for changes (check every 50ms for more responsive updates)
            self.timercsv.start(50)

            self.data_source_label.setText("Data Source: Real-Time CSV")
            self.data_source_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            # Stop monitoring
            self.monitoring = False
            self.timercsv.stop()
            self.start_buttoncsv.setText("Start Real-Time Data")
            print("--- Real-Time Data Monitoring Stopped ---")
            self.data_source_label.setText("Data Source: Simulation")
            self.data_source_label.setStyleSheet("color: black;")

    def read_csv_data(self):
        csv_file = Path(self.csv_path)
        
        if not csv_file.exists():
            print(f"Error: {self.csv_path} not found!")
            return
        
        current_modified_time = csv_file.stat().st_mtime
        current_size = csv_file.stat().st_size
        
        # Check if file has been modified
        if current_modified_time > self.last_modified_time or current_size != self.last_size:
            try:
                # Read the latest line from the CSV file
                with open(self.csv_path, 'r') as f:
                    # Read all lines and get the last non-empty one
                    lines = f.readlines()
                    if not lines:
                        return
                        
                    last_line = lines[-1].strip()
                    if not last_line:
                        return
                        
                    # Parse CSV data
                    parts = last_line.split(',')
                    
                    # Check if we have enough data
                    if len(parts) < 28:  # We need at least 28 elements based on your format
                        print(f"Warning: Incomplete data in CSV: {len(parts)} elements")
                        return
                    
                    # Extract data from CSV line
                    timestamp = float(parts[0])
                    
                    # Force and torque data
                    force = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    torque = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
                    
                    # Tibia position and quaternion (x,y,z, qx,qy,qz,qw)
                    tibia_position = np.array([float(parts[7]), float(parts[8]), float(parts[9])])
                    tibia_quaternion = np.array([float(parts[13]), float(parts[10]), float(parts[11]), float(parts[12])])
                    # Note the order change: CSV has qx,qy,qz,qw but your system expects qw,qx,qy,qz
                    
                    # Femur position and quaternion
                    femur_position = np.array([float(parts[14]), float(parts[15]), float(parts[16])])
                    femur_quaternion = np.array([float(parts[20]), float(parts[17]), float(parts[18]), float(parts[19])])
                    # Same reordering for quaternion components
                    
                    # Store force/torque in arrays
                    if len(self.forces) > 100:  # Keep only last 100 points
                        self.forces = np.vstack([self.forces[1:], force])
                        self.torques = np.vstack([self.torques[1:], torque])
                    else:
                        if len(self.forces) == 0:
                            self.forces = np.array([force])
                            self.torques = np.array([torque])
                        else:
                            self.forces = np.vstack([self.forces, force])
                            self.torques = np.vstack([self.torques, torque])
                    
                    self.current_data_index = len(self.forces) - 1
                    
                    # Update visualization based on current tab
                    current_tab = self.tabs.currentIndex()
                    
                    if current_tab == 0:  # Current Data tab
                        self.update_current_visualization(force, torque, flex)
                    elif current_tab == 1:  # History tab
                        # Add to history
                        self.force_history.append(force)
                        self.torque_history.append(torque)
                        
                        # Keep history to specified size
                        if len(self.force_history) > constants.HISTORY_SIZE:
                            self.force_history.pop(0)
                            self.torque_history.pop(0)
                            
                        self.update_history_visualization()
                    elif current_tab == 2:  # Bone visualization tab
                        # Update bone positions/orientations with real data
                        if hasattr(self, 'femur_mesh') and hasattr(self, 'femur_original_vertices'):
                            self.update_femur_with_data(femur_position, femur_quaternion)
                        
                        if hasattr(self, 'tibia_mesh') and hasattr(self, 'tibia_original_vertices'):
                            self.update_tibia_with_data(tibia_position, tibia_quaternion)
                        
                        # Update force visualization
                        self.update_bone_forces(self.current_data_index)
                    
                    # If recording is active, record this data point
                    if self.recording:
                        current_time = time.time() - self.recording_start_time
                        
                        # Use real bone data from CSV
                        data_point = [
                            current_time,
                            force[0], force[1], force[2],
                            torque[0], torque[1], torque[2],
                            femur_position[0], femur_position[1], femur_position[2],
                            femur_quaternion[0], femur_quaternion[1], femur_quaternion[2], femur_quaternion[3],
                            tibia_position[0], tibia_position[1], tibia_position[2],
                            tibia_quaternion[0], tibia_quaternion[1], tibia_quaternion[2], tibia_quaternion[3]
                        ]
                        
                        self.current_recording_data.append(data_point)
                
                # Update last modified time and size
                self.last_modified_time = current_modified_time
                self.last_size = current_size
                
            except Exception as e:
                print(f"Error processing CSV data: {str(e)}")
                import traceback
                traceback.print_exc()

    def update_bones_from_csv_data(self, femur_position, femur_quaternion, tibia_position, tibia_quaternion):
        """Update bone positions and orientations using real data from CSV"""
        # Update femur if it's loaded
        if hasattr(self, 'femur_mesh') and hasattr(self, 'femur_original_vertices'):
            self.update_femur_with_data(femur_position, femur_quaternion)
        
        # Update tibia if it's loaded
        if hasattr(self, 'tibia_mesh') and hasattr(self, 'tibia_original_vertices'):
            self.update_tibia_with_data(tibia_position, tibia_quaternion)
    


    def create_arrow(self, start_point, end_point, color=(1,0,0,1), arrow_size=15.0, shaft_width=2.0):
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

    def start_recording(self, test_name):
        """Start recording data for the current test"""
        self.recording = True
        self.current_recording_data = []
        self.recording_start_time = time.time()
        self.current_test_name = test_name
        print(f"Started recording data for {test_name}")

    def stop_recording(self):
        """Stop recording and save data to file"""
        if not self.recording:
            return
            
        self.recording = False
        
        # Create a filename with timestamp, angle, and test type
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        angle = constants.FLEXION_ANGLES[self.current_angle_index]
        filename = f"recorded_data/{timestamp}_{angle}deg_{self.current_test_name}.txt"
        
        # Write data to file
        with open(filename, 'w') as f:
            f.write("# Timestamp, Fx, Fy, Fz, Tx, Ty, Tz, FemurPosX, FemurPosY, FemurPosZ, FemurQuatW, FemurQuatX, FemurQuatY, FemurQuatZ, TibiaPosX, TibiaPosY, TibiaPosZ, TibiaQuatW, TibiaQuatX, TibiaQuatY, TibiaQuatZ\n")
            for data_point in self.current_recording_data:
                f.write(','.join(map(str, data_point)) + '\n')
        
        print(f"Saved {len(self.current_recording_data)} data points to {filename}")
        self.current_recording_data = []

    def load_force_torque_data(self):
        

        """Initialize force and torque data structures."""
        try:
            # Try to load data from file
            filename = constants.DATA_PREVIOUS_TEST
            print(f"Attempting to load data from {filename}")
            self.forces = []
            self.torques = []
            
            with open(filename, 'r') as file:
                for line in file:
                    values = [float(val.strip()) for val in line.strip().split(',')]
                    if len(values) >= 6:
                        self.forces.append(values[0:3])
                        self.torques.append(values[3:6])
            
            self.forces = np.array(self.forces)
            self.torques = np.array(self.torques)
            print(f"Successfully loaded {len(self.forces)} force/torque data points.")
        except Exception as e:
            print(f"Error loading force/torque data: {e}")
            # Just initialize empty arrays - we'll generate data dynamically
            self.forces = np.zeros((0, 3))
            self.torques = np.zeros((0, 3))
            print("Will generate force/torque data dynamically.")
    
    def on_tab_changed(self, index):
        # Update the appropriate visualization for the new tab
        if self.experiment_running and len(self.forces) > 0:
            if index == 0:  # Current Data tab
                force = self.forces[self.current_data_index].copy()
                torque = self.torques[self.current_data_index].copy()
                self.update_current_visualization(force, torque)
            elif index == 1:  # History tab
                self.update_history_visualization()
            elif index == 2:  # Bone visualization tab
                self.update_bone_forces(self.current_data_index)
            print(f"Tab changed to {index}, visualization updated")

    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QGridLayout()
        
        # Instruction label
        self.instruction_label = QLabel("Knee Test Bench")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(self.instruction_label, 0, 0, 1, 2)

        #csv test
         # Start button
        #self.start_buttoncsv = QPushButton("Start Reading")
        #self.start_buttoncsv.setFixedSize(150, 40)
        #self.start_buttoncsv.clicked.connect(self.toggle_monitoring)
        #main_layout.addWidget(self.start_buttoncsv, 0, 3)

        
        # Rotation timer progress bar
        rotation_progress_layout = QVBoxLayout()
        self.rotation_progress_label = QLabel("Please Flex the knee to the desired flexion angle, then hold the desired positions for the shown amount of time")
        self.rotation_progress_label.setAlignment(Qt.AlignCenter)
        rotation_progress_layout.addWidget(self.rotation_progress_label)
        self.rotation_progress = QProgressBar()
        self.rotation_progress.setRange(0, constants.HOLD_TIME)
        self.rotation_progress.setValue(constants.HOLD_TIME)
        self.rotation_progress.setTextVisible(True)
        self.rotation_progress.setFixedHeight(60)
        self.rotation_progress.setFormat("%v seconds remaining")
        rotation_progress_layout.addWidget(self.rotation_progress)
        main_layout.addLayout(rotation_progress_layout, 1, 0, 1, 2)
        
        # Create a splitter for the bottom section
        bottom_splitter = QSplitter(Qt.Horizontal)
        
        # Left part: Image display and visualization
        self.left_widget = QWidget()
        left_layout = QVBoxLayout()

         #Create the QTabWidget
        self.tabs = QTabWidget()

        # Create tab pages (as QWidget)
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
         
        # Add tabs to the tab widget
        self.tabs.addTab(self.tab1, "current data")
        self.tabs.addTab(self.tab2, "current + previous data")
        self.tabs.addTab(self.tab3, "bone visualization")

        # first tab
        tab1_layout = QVBoxLayout()
        # Add force/torque visualization
        viz_label_1 = QLabel("Force & Torque Visualization")
        viz_label_1.setAlignment(Qt.AlignCenter)
        viz_label_1.setFont(QFont("Arial", 12, QFont.Bold))
        tab1_layout.addWidget(viz_label_1)
        # Create matplotlib visualization
        self.canvas_current = MplCanvas(width=4, height=8, mode="current")
        tab1_layout.addWidget(self.canvas_current)
        self.tab1.setLayout(tab1_layout)

        # second tab
        tab2_layout = QVBoxLayout()
        # Add force/torque visualization
        viz_label_2 = QLabel("Force & Torque Visualization")
        viz_label_2.setAlignment(Qt.AlignCenter)
        viz_label_2.setFont(QFont("Arial", 12, QFont.Bold))
        tab2_layout.addWidget(viz_label_2)
        # Create matplotlib visualization
        self.canvas_history = MplCanvas(width=4, height=8, mode="history")
        tab2_layout.addWidget(self.canvas_history)
        self.tab2.setLayout(tab2_layout)
        
         # bone tab
        tab3_layout = QVBoxLayout()
        # Create 3D GL View Widget for bone visualization
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=1000, elevation=30, azimuth=-55)
        self.gl_view.setMinimumHeight(400)
        # Add axes for reference
        self.axes = ColoredGLAxisItem(size=(100, 100, 100)) #defined colors
        self.gl_view.addItem(self.axes)
        # buttons for loading bones
        bone_load_layout = QHBoxLayout()
        self.load_femur_button = QPushButton("Load Femur")
        self.load_femur_button.clicked.connect(self.load_femur)
        bone_load_layout.addWidget(self.load_femur_button)
        self.load_tibia_button = QPushButton("Load Tibia")
        self.load_tibia_button.clicked.connect(self.load_tibia)
        bone_load_layout.addWidget(self.load_tibia_button)
        # set background color
        self.gl_view.setBackgroundColor(QtGui.QColor(255, 255, 255))
         # Add force visualization objects
        self.force_arrow_shaft = None
        self.force_arrow_head = None
        # Connect tab change signal
        self.tabs.currentChanged.connect(self.on_tab_changed)
        tab3_layout.addWidget(self.gl_view)
        tab3_layout.addLayout(bone_load_layout)
        self.tab3.setLayout(tab3_layout)
        # Timer for bone animation updates
        self.bone_timer = QTimer()
        self.bone_timer.timeout.connect(self.update_bones)
        self.bone_timer.setInterval(25)  # 25ms for 40 fps
        # Initialize bone data generator
        self.bone_data_generator = BoneDataGenerator()
        # Add a toggle for automatic bone movement
        self.auto_bone_movement = QPushButton("Start Bone Animation")
        self.auto_bone_movement.setCheckable(True)
        self.auto_bone_movement.clicked.connect(self.toggle_bone_animation)
        tab3_layout.addWidget(self.auto_bone_movement)

        left_layout.addWidget(self.tabs)
        self.left_widget.setLayout(left_layout)
        bottom_splitter.addWidget(self.left_widget)
        
        
        # Right part: Control buttons and image
        right_widget = QWidget()
        right_layout = QGridLayout()
        
        # Start Experiment Button
        self.start_button = QPushButton("Start Experiment")
        self.start_button.clicked.connect(self.start_experiment)
        self.start_button.setFixedHeight(constants.BUTTON_HEIGHT)
        
        # Next Angle Button
        self.next_button = QPushButton("Next Angle")
        self.next_button.clicked.connect(self.next_angle)
        self.next_button.setEnabled(False)
        self.next_button.setFixedHeight(constants.BUTTON_HEIGHT)

        # Next Angle Label
        self.next_label = QLabel("test1")
        font = self.next_label.font()
        font.setPointSize(12)
        self.next_label.setFont(font)

        # Rotate Button
        self.rotate_button = QPushButton("Hold Flexion for 5 s")
        self.rotate_button.clicked.connect(self.start_rotation)
        self.rotate_button.setEnabled(False)
        self.rotate_button.setFixedHeight(constants.BUTTON_HEIGHT)

        # Varus Button
        self.varus_button = QPushButton("Apply Varus Load for 5 s")
        self.varus_button.clicked.connect(self.start_varus)
        self.varus_button.setEnabled(False)
        self.varus_button.setFixedHeight(constants.BUTTON_HEIGHT)

        # Valgus Button
        self.valgus_button = QPushButton("Apply Valgus Load for 5 s")
        self.valgus_button.clicked.connect(self.start_valgus)
        self.valgus_button.setEnabled(False)
        self.valgus_button.setFixedHeight(constants.BUTTON_HEIGHT)

        # IR Button
        self.internal_rot_button = QPushButton("Rotate Tibia internally for 5 s")
        self.internal_rot_button.clicked.connect(self.start_internal_rot)
        self.internal_rot_button.setEnabled(False)
        self.internal_rot_button.setFixedHeight(constants.BUTTON_HEIGHT)

        # ER Button
        self.external_rot_button = QPushButton("Rotate Tibia externally for 5 s")
        self.external_rot_button.clicked.connect(self.start_external_rot)
        self.external_rot_button.setEnabled(False)
        self.external_rot_button.setFixedHeight(constants.BUTTON_HEIGHT)

        # Lachmann Test Button - New addition
        self.lachmann_button = QPushButton("Perform Lachmann Test for 8 s")
        self.lachmann_button.clicked.connect(self.start_lachmann)
        self.lachmann_button.setEnabled(False)
        self.lachmann_button.setFixedHeight(constants.BUTTON_HEIGHT)

        record_data_label = QLabel("Record Data")
        record_data_label.setAlignment(Qt.AlignCenter)
        
        # Image frame
        self.image_frame = QFrame()
        self.image_frame.setLineWidth(2)
        self.image_frame.setMinimumSize(300, 250)
        image_layout = QVBoxLayout()
        self.image_label = QLabel()
        #self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label, alignment=Qt.AlignHCenter | Qt.AlignTop)
        self.image_frame.setLayout(image_layout)
        
        # Layout arrangement
        subsub_layout = QHBoxLayout()
        subsub_layout.addWidget(self.start_button)
        subsub_layout.addWidget(self.next_button)

        right_layout.addLayout(subsub_layout, 0, 0)
        right_layout.addWidget(self.next_label, 0, 1)
        right_layout.addWidget(self.image_frame, 1, 1, 5, 1)
        right_layout.addWidget(record_data_label, 1, 0, 2, 1)
        right_layout.addWidget(self.rotate_button, 3, 0)
        right_layout.addWidget(self.varus_button, 4, 0)
        right_layout.addWidget(self.valgus_button, 5, 0)
        right_layout.addWidget(self.internal_rot_button, 6, 0)
        right_layout.addWidget(self.external_rot_button, 7, 0)
        right_layout.addWidget(self.lachmann_button, 8, 0)
        
        right_widget.setLayout(right_layout)
        bottom_splitter.addWidget(right_widget)
        
        # Add the splitter to the main layout
        main_layout.addWidget(bottom_splitter, 2, 0, 1, 2)
        
        # Overall progress bar
        overall_progress_layout = QVBoxLayout()
        overall_progress_label = QLabel("Overall Experiment Progress:")
        overall_progress_label.setAlignment(Qt.AlignBottom)
        overall_progress_layout.addWidget(overall_progress_label)
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, len(constants.FLEXION_ANGLES))
        self.overall_progress.setValue(0)
        self.overall_progress.setTextVisible(True)
        self.overall_progress.setFixedHeight(60)
        self.overall_progress.setFormat("%v/%m angles completed")
        self.overall_progress.setStyleSheet("QProgressBar {border: 1px solid grey; border-radius: 3px; text-align: center;}"
                                           "QProgressBar::chunk {background-color: #4CAF50; width: 10px;}") # Set color for overall progress bar
        overall_progress_layout.addWidget(self.overall_progress)
        main_layout.addLayout(overall_progress_layout, 3, 0, 1, 2)


        # Set main layout
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Initial display
        self.update_display()
        
        # Current test type
        self.current_test_type = 'none'


    def toggle_bone_animation(self):
        if self.auto_bone_movement.isChecked():
            self.auto_bone_movement.setText("Stop Bone Animation")
            # Start animation timer
            self.bone_timer.start()
            # Update forces right away
            self.update_bone_forces(self.current_data_index)
        else:
            self.auto_bone_movement.setText("Start Bone Animation")
            # Stop animation timer
            self.bone_timer.stop()

    def update_bones(self):
        # Only update if the bone tab is active
        if self.tabs.currentIndex() != 2:
            return
            
        # Get updated bone position, quaternion, force and torque data
        bone_data = self.bone_data_generator.update(0.02)  # 50ms = 0.02s
        
        # Update bone positions/orientations
        if hasattr(self, 'femur_mesh') and hasattr(self, 'femur_original_vertices'):
            # Update femur
            self.update_femur_with_data(
                bone_data['femur_position'], 
                bone_data['femur_quaternion']
            )
        
        if hasattr(self, 'tibia_mesh') and hasattr(self, 'tibia_original_vertices'):
            # Update tibia
            self.update_tibia_with_data(
                bone_data['tibia_position'], 
                bone_data['tibia_quaternion']
            )
        
        # Update forces and torques if bone animation is active
        if self.auto_bone_movement.isChecked():
            # Get current force/torque and update the visualization
            force = bone_data['force']
            torque = bone_data['torque']
            
            # Store current data point
            if len(self.forces) > 100:  # Keep only last 100 points in memory
                self.forces = np.vstack([self.forces[1:], force])
                self.torques = np.vstack([self.torques[1:], torque])
            else:
                if len(self.forces) == 0:
                    self.forces = np.array([force])
                    self.torques = np.array([torque])
                else:
                    self.forces = np.vstack([self.forces, force])
                    self.torques = np.vstack([self.torques, torque])
                    
            self.current_data_index = len(self.forces) - 1
            
            # Update force visualization on bone
            self.update_bone_forces(self.current_data_index)

    def update_femur_with_data(self, position, quaternion):
        # Get 4x4 transformation matrix
        transform = quaternion_to_transform_matrix(quaternion, position)
        
        # Update the mesh transformation
        self.femur_mesh.setTransform(transform)

    def update_tibia_with_data(self, position, quaternion):
        pivot_point = np.array([0, 0, 0])  # Define the specific pivot point relative to the tibia's local coordinates
        
        # Get transformation matrix with rotation only, no translation
        R_matrix = quaternion_to_transform_matrix(quaternion)
        
        # Create translation matrices
        T_to_origin = np.eye(4)
        T_to_origin[0:3, 3] = -pivot_point  # Move pivot point to origin 
        
        T_from_origin = np.eye(4)
        T_from_origin[0:3, 3] = pivot_point  # Move back from origin
        
        T_position = np.eye(4)
        T_position[0:3, 3] = position  # Final position
        
        # Apply combined transform: translate to origin, rotate, translate back, then to final position
        transform = np.dot(T_position, np.dot(T_from_origin, np.dot(R_matrix, T_to_origin)))
        
        # Apply the transform to the mesh
        self.tibia_mesh.setTransform(transform)
        
    def update_visualization_timer(self):
        
        """Called by timer to update visualization"""
        if self.experiment_running:
            # Get bone positions and force/torque data
            bone_data = self.bone_data_generator.update(0.03)  # Update with small time step
            
            # Extract force/torque data
            force = bone_data['force']
            torque = bone_data['torque']
            
            # Store current data point for visualization
            if len(self.forces) > 100:  # Keep only last 100 points in memory
                self.forces = np.vstack([self.forces[1:], force])
                self.torques = np.vstack([self.torques[1:], torque])
            else:
                if len(self.forces) == 0:
                    self.forces = np.array([force])
                    self.torques = np.array([torque])
                else:
                    self.forces = np.vstack([self.forces, force])
                    self.torques = np.vstack([self.torques, torque])
            
            self.current_data_index = len(self.forces) - 1
            
            # Check which tab is currently active and update the relevant visualization
            current_tab = self.tabs.currentIndex()
            
            if current_tab == 0:  # Current Data tab
                self.update_current_visualization(force, torque)
            elif current_tab == 1:  # History tab
                # Add to history
                self.force_history.append(force)
                self.torque_history.append(torque)
                
                # Keep history to specified size
                if len(self.force_history) > constants.HISTORY_SIZE:
                    self.force_history.pop(0)
                    self.torque_history.pop(0)
                    
                self.update_history_visualization()
            elif current_tab == 2:  # Bone visualization tab
                self.update_bone_forces(self.current_data_index)

            # Record data if recording is active
            if self.recording:
                current_time = time.time() - self.recording_start_time
                
                # Get bone data
                femur_pos = bone_data['femur_position']
                femur_quat = bone_data['femur_quaternion']
                tibia_pos = bone_data['tibia_position']
                tibia_quat = bone_data['tibia_quaternion']
                
                # Combine all data into one record
                data_point = [
                    current_time,
                    force[0], force[1], force[2],
                    torque[0], torque[1], torque[2],
                    femur_pos[0], femur_pos[1], femur_pos[2],
                    femur_quat[0], femur_quat[1], femur_quat[2], femur_quat[3],
                    tibia_pos[0], tibia_pos[1], tibia_pos[2],
                    tibia_quat[0], tibia_quat[1], tibia_quat[2], tibia_quat[3]
                ]
                
                self.current_recording_data.append(data_point)
        
    def update_visualization(self, data_index=0):
        """Update only the active visualization tab"""
        current_tab = self.tabs.currentIndex()
        
        if not self.experiment_running or len(self.forces) == 0:
            return
            
        idx = data_index % len(self.forces)
        force = self.forces[idx].copy()
        torque = self.torques[idx].copy()
        
        update_methods = {
            0: self.update_current_visualization,
            1: self.update_history_visualization,
            2: self.update_bone_forces
        }
        
        if current_tab in update_methods:
            if current_tab == 1:
                update_methods[current_tab]()
            else:
                update_methods[current_tab](force, torque)

    def update_current_visualization(self, force, torque):
        """Update the force/torque visualization with only the current data."""
        # Force arrow
        force_mag = np.sqrt(np.sum(force**2))
        if force_mag > 0.01:
            # Remove old arrow from the plot
            if hasattr(self, 'force_arrow_plt'):
                self.force_arrow_plt.remove()
                
            # Create a new arrow
            self.force_arrow_plt = self.canvas_current.axes_force.quiver(
                0, 0, 0, 
                force[0], force[1], force[2],
                color='blue', 
                linewidth=1,
                normalize=False,
                arrow_length_ratio=0.1
            )
        
        # Torque arrow
        torque_mag = np.sqrt(np.sum(torque**2))
        if torque_mag > 0.01:
            # Remove old arrow from the plot
            if hasattr(self, 'torque_arrow_plt'):
                self.torque_arrow_plt.remove()
                
            # Create a new arrow
            self.torque_arrow_plt = self.canvas_current.axes_torque.quiver(
                0, 0, 0, 
                torque[0], torque[1], torque[2],
                color='red', 
                linewidth=1,
                normalize=False,
                arrow_length_ratio=0.1
            )
        
        # Update text elements
        self.canvas_current.force_mag_text.set_text(f"Current Force: {round(force_mag)}N")
        self.canvas_current.torque_mag_text.set_text(f"Current Torque: {round(torque_mag)}Nm")
        self.canvas_current.force_comp_text.set_text(f"Fx: {round(force[0])}, Fy: {round(force[1])}, Fz: {round(force[2])}")
        self.canvas_current.torque_comp_text.set_text(f"Tx: {round(torque[0])}, Ty: {round(torque[1])}, Tz: {round(torque[2])}")
        
        # Redraw the canvas
        self.canvas_current.draw()
    
    def update_history_visualization(self):
        """Update the force/torque visualization with history data."""
        # Check if we have data to visualize
        if not self.force_history or not self.torque_history:
            return
        
        # Determine how many arrows should be displayed (all entries in history)
        history_length = len(self.force_history)
        
        # If we already have the maximum number of arrows displayed,
        # remove the oldest one to make room for the newest
        if len(self.canvas_history.force_arrows) >= history_length:
            if self.canvas_history.force_arrows:
                oldest_force_arrow = self.canvas_history.force_arrows.pop(0)
                oldest_force_arrow.remove()
            
            if self.canvas_history.torque_arrows:
                oldest_torque_arrow = self.canvas_history.torque_arrows.pop(0)
                oldest_torque_arrow.remove()
        
        # If we're just starting or reset, we need to draw all arrows
        if len(self.canvas_history.force_arrows) == 0:
            # Plot history with color gradient (older = more transparent)
            cmap_force = plt.get_cmap('Blues')
            cmap_torque = plt.get_cmap('PuRd')
            
            # Draw all arrows in history
            for i, (hist_force, hist_torque) in enumerate(zip(self.force_history, self.torque_history)):
                # Calculate color and alpha based on position in history
                alpha = 0.3 + 0.7 * (i / max(1, history_length - 1))
                color_idx = i / max(1, history_length - 1)
                
                # Force arrow
                force_mag = np.sqrt(np.sum(hist_force**2))
                color_force = cmap_force(color_idx)
                color_force = (*color_force[:3], alpha)
                
                # Only draw if magnitude is not zero
                if force_mag > 0.01:
                    arrow = self.canvas_history.axes_force.quiver(
                        0, 0, 0, 
                        hist_force[0], hist_force[1], hist_force[2],
                        color=color_force, 
                        linewidth=1,
                        normalize=False,
                        arrow_length_ratio=0.1
                    )
                    self.canvas_history.force_arrows.append(arrow)
                else:
                    # Add placeholder if magnitude is too small
                    self.canvas_history.force_arrows.append(None)
                
                # Torque arrow
                torque_mag = np.sqrt(np.sum(hist_torque**2))
                color_torque = cmap_torque(color_idx)
                color_torque = (*color_torque[:3], alpha)
                
                # Only draw if magnitude is not zero
                if torque_mag > 0.01:
                    arrow = self.canvas_history.axes_torque.quiver(
                        0, 0, 0, 
                        hist_torque[0], hist_torque[1], hist_torque[2],
                        color=color_torque, 
                        linewidth=1,
                        normalize=False,
                        arrow_length_ratio=0.1
                    )
                    self.canvas_history.torque_arrows.append(arrow)
                else:
                    # Add placeholder if magnitude is too small
                    self.canvas_history.torque_arrows.append(None)
        else:
            # Just add the newest arrow
            cmap_force = plt.get_cmap('Blues')
            cmap_torque = plt.get_cmap('PuRd')
            
            # Newest data point
            newest_force = self.force_history[-1]
            newest_torque = self.torque_history[-1]
            
            # Calculate color for newest arrow (full opacity)
            alpha = 1.0
            color_idx = 1.0  # Newest = full color
            
            # Force arrow
            force_mag = np.sqrt(np.sum(newest_force**2))
            color_force = cmap_force(color_idx)
            color_force = (*color_force[:3], alpha)
            
            # Only draw if magnitude is not zero
            if force_mag > 0.01:
                new_force_arrow = self.canvas_history.axes_force.quiver(
                    0, 0, 0, 
                    newest_force[0], newest_force[1], newest_force[2],
                    color=color_force, 
                    linewidth=1,
                    normalize=False,
                    arrow_length_ratio=0.1
                )
                self.canvas_history.force_arrows.append(new_force_arrow)
            else:
                # Add placeholder if magnitude is too small
                self.canvas_history.force_arrows.append(None)
            
            # Torque arrow
            torque_mag = np.sqrt(np.sum(newest_torque**2))
            color_torque = cmap_torque(color_idx)
            color_torque = (*color_torque[:3], alpha)
            
            # Only draw if magnitude is not zero
            if torque_mag > 0.01:
                new_torque_arrow = self.canvas_history.axes_torque.quiver(
                    0, 0, 0, 
                    newest_torque[0], newest_torque[1], newest_torque[2],
                    color=color_torque, 
                    linewidth=1,
                    normalize=False,
                    arrow_length_ratio=0.1
                )
                self.canvas_history.torque_arrows.append(new_torque_arrow)
            else:
                # Add placeholder if magnitude is too small
                self.canvas_history.torque_arrows.append(None)
        
        # Update the colors of all arrows to maintain the gradient effect
        for i, (force_arrow, torque_arrow) in enumerate(zip(
                self.canvas_history.force_arrows, 
                self.canvas_history.torque_arrows)):
            
            # Calculate new color and alpha based on updated position in history
            alpha = 0.3 + 0.7 * (i / max(1, len(self.canvas_history.force_arrows) - 1))
            color_idx = i / max(1, len(self.canvas_history.force_arrows) - 1)
            
            # Update force arrow color if it exists
            if force_arrow is not None:
                color_force = cmap_force(color_idx)
                color_force = (*color_force[:3], alpha)
                force_arrow.set_color(color_force)
            
            # Update torque arrow color if it exists
            if torque_arrow is not None:
                color_torque = cmap_torque(color_idx)
                color_torque = (*color_torque[:3], alpha)
                torque_arrow.set_color(color_torque)
        
        # Display magnitudes of the current force/torque
        current_force = self.force_history[-1]
        current_torque = self.torque_history[-1]
        force_mag = np.sqrt(np.sum(current_force**2))
        torque_mag = np.sqrt(np.sum(current_torque**2))
        
        self.canvas_history.force_mag_text.set_text(f"Force Mag: {round(force_mag)}N")
        self.canvas_history.torque_mag_text.set_text(f"Torque Mag: {round(torque_mag)}Nm")
        self.canvas_history.force_comp_text.set_text(
            f"Fx: {round(current_force[0])}, Fy: {round(current_force[1])}, Fz: {round(current_force[2])}"
        )
        self.canvas_history.torque_comp_text.set_text(
            f"Tx: {round(current_torque[0])}, Ty: {round(current_torque[1])}, Tz: {round(current_torque[2])}"
        )
        
        # Redraw the canvas
        self.canvas_history.draw()

    def update_display(self):
        current_angle = constants.FLEXION_ANGLES[self.current_angle_index]
        self.next_label.setText(f"Please flex knee to {current_angle} degrees")
        self.next_label.setAlignment(Qt.AlignCenter)
        # Update overall progress
        self.overall_progress.setValue(self.current_angle_index)
            
        # Load the appropriate image
        try:
            pixmap = QPixmap(f"KW{current_angle}.jpg")
            if pixmap.isNull():
                    self.image_label.setText(f"Image for {current_angle}° not found")
            else:
                # Scale the image to fit the frame while maintaining aspect ratio
                pixmap = pixmap.scaled(self.image_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
        except Exception as e:
            self.image_label.setText(f"Error loading image: {str(e)}")

    def update_bone_forces(self, data_index=0):
        """Update the force/torque visualization in 3D bone view"""
        # Skip if not on the bone visualization tab
        if self.tabs.currentIndex() != 2:
            return
            
        # Get current data point
        idx = data_index % len(self.forces)
        force = self.forces[idx].copy()
        
        # Scale forces for better visualization
        scale_factor = 20.0
        force_scaled = force * scale_factor

        # Set the position of the force arrow - attach to tibia at specific point
        tibia_pos = self.get_tibia_force_origin()
        tibia_pos[0] -= 20
        tibia_pos[2] -= 40
        
        # Calculate end point for the arrow
        end_point = tibia_pos + force_scaled
        
        # First, remove old arrows if they exist
        if hasattr(self, 'force_arrow_shaft') and self.force_arrow_shaft is not None:
            self.gl_view.removeItem(self.force_arrow_shaft)
        if hasattr(self, 'force_arrow_head') and self.force_arrow_head is not None:
            self.gl_view.removeItem(self.force_arrow_head)
        
        # Create new arrows
        self.force_arrow_shaft, self.force_arrow_head = self.create_arrow(
            tibia_pos, end_point, color=(1, 0, 0, 1), arrow_size=constants.ARROW_SIZE, shaft_width=constants.SHAFT_WIDTH
        )
        
        # Add new arrows to view
        if self.force_arrow_shaft is not None:
            self.gl_view.addItem(self.force_arrow_shaft)
        if self.force_arrow_head is not None:
            self.gl_view.addItem(self.force_arrow_head)

    def get_tibia_force_origin(self):
        """Get the specific point on the tibia where the force arrow should originate"""
        # Get base position from bone data generator
        base_position = self.bone_data_generator.tibia_position.copy()
        
        # Define anatomical offset - these values should be adjusted to match your specific model
        anatomical_offset = np.array([190, -20, 0])  # X, Y, Z offset in model coordinates
        
        # Return the origin point
        return base_position + anatomical_offset

    def get_tibia_center(self):
        """Get the current center of the tibia for attaching forces"""
        return self.bone_data_generator.tibia_position
           
    def start_experiment(self):
        self.current_angle_index = 0
        self.current_angle = constants.FLEXION_ANGLES[self.current_angle_index]
        self.overall_progress.setValue(0)
        self.next_label.setText(f"Please flex knee to {self.current_angle} degrees")
        self.rotation_progress_label.show()
        self.rotation_progress.show()

        # Reset progress bar range to match rotation time (5 seconds)
        self.rotation_progress.setRange(0, constants.HOLD_TIME)
        self.rotation_progress.setValue(constants.HOLD_TIME)
        self.rotation_progress.setFormat("%v seconds remaining")
    
        # Reset current test type
        self.current_test_type = 'none'
        
        # Reset visualization history
        self.force_history = []
        self.torque_history = []
        self.current_data_index = 0
        
        try:
            pixmap = QPixmap(f"KW{self.current_angle}.jpg")
            if pixmap.isNull():
                self.image_label.setText(f"Image for {self.current_angle}° not found")
            else:
                pixmap = pixmap.scaled(self.image_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation) # Scale the image 
                self.image_label.setPixmap(pixmap)
        except Exception as e:
            self.image_label.setText(f"Error loading image: {str(e)}")
        
        # Set experiment running flag
        self.experiment_running = True
        
        # Enable only needed buttons
        self.next_label.show()
        self.start_button.setEnabled(False)
        self.rotate_button.setEnabled(True)
        
        # Start visualization timer immediately and keep it running throughout the experiment
        if not self.viz_timer.isActive():
            self.viz_timer.start()

        # Update visualization initially
        self.update_visualization(0)
        
        # Also update bone forces explicitly
        self.update_bone_forces(0)
    
    def next_angle(self):
        self.current_angle_index += 1
        self.update_display()
        self.next_button.setEnabled(False)
        self.rotate_button.setEnabled(True)

    def start_rotation(self):
        self.rotate_button.setEnabled(False) # Disable rotate button
        self.varus_button.setEnabled(True) 
        self.remaining_time = constants.HOLD_TIME
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.next_button.setEnabled(False)
        self.start_recording(f"neutral") # Start recording data
        
    def start_varus(self):
        self.varus_button.setEnabled(False) # Disable varus button
        self.remaining_time = constants.HOLD_TIME
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  
        self.valgus_button.setEnabled(True)
        self.start_recording(f"var") # Start recording data

    def start_valgus(self):
        self.valgus_button.setEnabled(False) # Disable valgus button
        self.remaining_time = constants.HOLD_TIME
        self.rotation_progress.setValue(constants.HOLD_TIME)
        self.seconds_timer.start(1000)  
        self.internal_rot_button.setEnabled(True)
        self.start_recording(f"val") # Start recording data

    def start_internal_rot(self):
        self.internal_rot_button.setEnabled(False) # Disable internal rotation button
        self.remaining_time = constants.HOLD_TIME
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.external_rot_button.setEnabled(True)
        self.start_recording(f"int")# Start recording data

    def start_external_rot(self):
        self.external_rot_button.setEnabled(False) # Disable external rotation button
        self.remaining_time = constants.HOLD_TIME
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.start_recording(f"ext") # Start recording data

        # Enable appropriate next button based on where we are in the test
        if self.current_angle_index >= (len(constants.FLEXION_ANGLES) - 1):
            self.lachmann_button.setEnabled(True) # last angle, enable Lachmann test button
            self.next_button.setEnabled(False)
        else:
            self.next_button.setEnabled(True) # not last angle: enable next button
            self.lachmann_button.setEnabled(False)

    def start_lachmann(self):  
        self.lachmann_button.setEnabled(False)
        self.image_label.clear()
        self.next_label.hide()
        
        self.rotation_progress_label.setText("Performing Lachmann Test")
        self.rotation_progress_label.show()
        self.remaining_time = constants.LACHMANN_TIME # Set timer for Lachmann test
        self.rotation_progress.setValue(self.remaining_time)
        self.rotation_progress.setRange(0, constants.LACHMANN_TIME)
        self.rotation_progress.setFormat("%v seconds remaining")
        self.seconds_timer.start(1000)  # Start the timer and update every second
        self.start_recording("lachmann") # Start recording data
        self.current_test_type = 'lachmann' # Set flag to indicate we're in Lachmann test
    
    def update_seconds_progress(self):
        self.remaining_time -= 1
        self.rotation_progress.setValue(self.remaining_time)
        
        if self.remaining_time <= 0:
            self.seconds_timer.stop()
            self.rotation_complete()
    
    def rotation_complete(self):
        self.timer.stop()
        self.seconds_timer.stop()
        self.rotation_progress.setValue(0)

        # Stop recording data if active
        if self.recording:
            self.stop_recording()

        # Check if we just completed a Lachmann test
        if self.current_test_type == 'lachmann':
            # Reset the flag
            self.current_test_type = 'none'
        
            #self.instruction_label.setText("Experiment Complete!")
            self.overall_progress.setValue(len(constants.FLEXION_ANGLES))
            self.image_label.clear()
            self.start_button.setEnabled(True) # Enable start button again

            # Hide instructions
            self.next_label.hide()
            self.rotation_progress_label.hide()
            self.rotation_progress.hide()
        
            # Stop visualization timer
            if self.viz_timer.isActive():
                self.viz_timer.stop()
        
            # Reset experiment running flag
            self.experiment_running = False

        elif self.current_angle_index >= (len(constants.FLEXION_ANGLES) - 1) and self.external_rot_button.isEnabled() == False:
            self.next_button.setEnabled(False) # End of regular experiment - enable Lachmann test

    def load_femur(self):
        try:
            # Load femur STL
            femur_vertices, femur_faces = load_stl_as_mesh(constants.FEMUR)
            self.femur_original_vertices = femur_vertices.copy()
            #femur_vertices = femur_vertices * 0.9 #Scale down the vertices
            
            # Store vertices in a numpy array for faster operations
            self.femur_verts = np.array(femur_vertices, dtype=np.float32)
            self.femur_faces = np.array(femur_faces, dtype=np.uint32)
            
            # Create a SINGLE mesh item to reuse
            self.femur_mesh = gl.GLMeshItem(
                vertexes=self.femur_verts,
                faces=self.femur_faces,
                smooth=True, 
                drawEdges=False,
                color = QtGui.QColor(112, 128, 144)
            )
            self.gl_view.addItem(self.femur_mesh)
            
            # Set up transform matrix (initialize once)
            self.femur_transform = np.identity(4, dtype=np.float32)
            
            # Disable load button
            self.load_femur_button.setEnabled(False)
            self.load_femur_button.setText("Femur Loaded")
        except Exception as e:
            print(f"Error loading femur: {e}")
            self.load_femur_button.setText("Error")


    def load_tibia(self):
        try:
            # Load tibia STL
            tibia_vertices, tibia_faces = load_stl_as_mesh(constants.TIBIA)
            self.tibia_original_vertices = tibia_vertices.copy()
            
            # Create mesh item but don't apply any transformations yet
            self.tibia_mesh = gl.GLMeshItem(
                vertexes=tibia_vertices, 
                faces=tibia_faces,
                smooth=True, 
                drawEdges=False,
                color = QtGui.QColor(47, 79, 79)
            )
            self.gl_view.addItem(self.tibia_mesh)
            
            # Disable load button
            self.load_tibia_button.setEnabled(False)
            self.load_tibia_button.setText("Tibia Loaded")
        except Exception as e:
            print(f"Error loading tibia: {e}")
            self.load_tibia_button.setText("Error")
        
    

if __name__ == "__main__":
    try:  
        app = QApplication(sys.argv)
        window = KneeFlexionExperiment()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()