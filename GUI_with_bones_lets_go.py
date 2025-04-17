import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFrame, 
                            QProgressBar, QGridLayout, QSplitter, QTabWidget, QSlider, QGroupBox)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
import matplotlib.cm as cm
import pyqtgraph.opengl as gl
from stl import mesh
from pyqtgraph.Qt import QtGui
import os
import time
import datetime


def load_stl_as_mesh(filename):
    """Load an STL file and return vertices and faces for PyQtGraph GLMeshItem"""
    try:
        # Load the STL file
        stl_mesh = mesh.Mesh.from_file(filename)
        
        # Get vertices (each face has 3 vertices)
        vertices = stl_mesh.vectors.reshape(-1, 3)
        
        # Create faces array - each triplet of vertices forms a face
        faces = np.arange(len(vertices)).reshape(-1, 3)
        
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

def apply_transform_quaternion(vertices, position=(0, 0, 0), quaternion=(1, 0, 0, 0)):
    """
    Apply transformation using position and quaternion
    
    Args:
        vertices: Original vertices
        position: (x, y, z) translation
        quaternion: (w, x, y, z) quaternion
    """
    # Normalize quaternion
    q = np.array(quaternion)
    q = q / np.sqrt(np.sum(q * q))
    w, x, y, z = q
    
    # Convert quaternion to rotation matrix
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    # Apply transformation
    transformed = (vertices @ R.T) + np.array(position)
    return transformed

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
        
        return {
            'femur_position': self.femur_position,
            'femur_quaternion': self.femur_quaternion,
            'tibia_position': self.tibia_position,
            'tibia_quaternion': self.tibia_quaternion
        }


class MplCanvas(FigureCanvas):
    """Matplotlib canvas class for embedding plots in Qt"""
    def __init__(self, width=5, height=4):
        self.fig = Figure(figsize=(width, height))
        self.axes_force = self.fig.add_subplot(121, projection='3d')
        self.axes_torque = self.fig.add_subplot(122, projection='3d')
        
        # Set up axes once
        force_max = 10
        torque_max = 2
        
        # Force plot setup
        self.axes_force.set_xlim([-force_max, force_max])
        self.axes_force.set_ylim([-force_max, force_max])
        self.axes_force.set_zlim([-force_max, force_max])
        self.axes_force.set_title('Force History (N)')
        self.axes_force.set_xlabel('X')
        self.axes_force.set_ylabel('Y')
        self.axes_force.set_zlabel('Z')
        self.axes_force.grid(False)
        self.axes_force.set_axis_off()
        
        # Torque plot setup
        self.axes_torque.set_xlim([-torque_max, torque_max])
        self.axes_torque.set_ylim([-torque_max, torque_max])
        self.axes_torque.set_zlim([-torque_max, torque_max])
        self.axes_torque.set_title('Torque History (Nm)')
        self.axes_torque.set_xlabel('X')
        self.axes_torque.set_ylabel('Y')
        self.axes_torque.set_zlabel('Z')
        self.axes_torque.grid(False)
        self.axes_torque.set_axis_off()
        
        # Reference axes (drawn once)
        self.ref_axes_force = [
            self.axes_force.quiver(0, 0, 0, force_max*0.2, 0, 0, color='red', linewidth=1, arrow_length_ratio=0.1),
            self.axes_force.quiver(0, 0, 0, 0, force_max*0.2, 0, color='green', linewidth=1, arrow_length_ratio=0.1),
            self.axes_force.quiver(0, 0, 0, 0, 0, force_max*0.2, color='blue', linewidth=1, arrow_length_ratio=0.1)
        ]
        self.ref_axes_torque = [
            self.axes_torque.quiver(0, 0, 0, torque_max*0.2, 0, 0, color='red', linewidth=1, arrow_length_ratio=0.1),
            self.axes_torque.quiver(0, 0, 0, 0, torque_max*0.2, 0, color='green', linewidth=1, arrow_length_ratio=0.1),
            self.axes_torque.quiver(0, 0, 0, 0, 0, torque_max*0.2, color='blue', linewidth=1, arrow_length_ratio=0.1)
        ]
        
        # Labels for the reference axes
        self.axes_force.text(force_max*0.22, 0, 0, "X", color='red')
        self.axes_force.text(0, force_max*0.22, 0, "Y", color='green')
        self.axes_force.text(0, 0, force_max*0.22, "Z", color='blue')
        
        self.axes_torque.text(torque_max*0.22, 0, 0, "X", color='red')
        self.axes_torque.text(0, torque_max*0.22, 0, "Y", color='green')
        self.axes_torque.text(0, 0, torque_max*0.22, "Z", color='blue')
        
        # For history, we'll maintain a list of arrows
        self.force_arrows = []
        self.torque_arrows = []
        
        # Text elements for magnitudes and components
        self.force_mag_text = self.axes_force.text2D(0.05, 0.95, "", transform=self.axes_force.transAxes)
        self.torque_mag_text = self.axes_torque.text2D(0.05, 0.95, "", transform=self.axes_torque.transAxes)
        self.force_comp_text = self.axes_force.text2D(0.05, 0.90, "", transform=self.axes_force.transAxes, fontsize=8)
        self.torque_comp_text = self.axes_torque.text2D(0.05, 0.90, "", transform=self.axes_torque.transAxes, fontsize=8)
        
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()


class MplCurrentCanvas(FigureCanvas):
    """Matplotlib canvas class for displaying only current force/torque"""
    def __init__(self, width=5, height=4):
        self.fig = Figure(figsize=(width, height))
        self.axes_force = self.fig.add_subplot(121, projection='3d')
        self.axes_torque = self.fig.add_subplot(122, projection='3d')
        
        # Set up axes once
        force_max = 12
        torque_max = 2
        
        # Force plot setup
        self.axes_force.set_xlim([-force_max, force_max])
        self.axes_force.set_ylim([-force_max, force_max])
        self.axes_force.set_zlim([-force_max, force_max])
        self.axes_force.set_title('Current Force (N)')
        self.axes_force.set_xlabel('X')
        self.axes_force.set_ylabel('Y')
        self.axes_force.set_zlabel('Z')
        self.axes_force.grid(False)
        self.axes_force.set_axis_off()
        
        # Torque plot setup
        self.axes_torque.set_xlim([-torque_max, torque_max])
        self.axes_torque.set_ylim([-torque_max, torque_max])
        self.axes_torque.set_zlim([-torque_max, torque_max])
        self.axes_torque.set_title('Current Torque (Nm)')
        self.axes_torque.set_xlabel('X')
        self.axes_torque.set_ylabel('Y')
        self.axes_torque.set_zlabel('Z')
        self.axes_torque.grid(False)
        self.axes_torque.set_axis_off()
        
        # Reference axes (drawn once)
        from mpl_toolkits.mplot3d import Axes3D
        self.ref_axes_force = [
            self.axes_force.quiver(0, 0, 0, force_max*0.2, 0, 0, color='red', linewidth=1, arrow_length_ratio=0.1),
            self.axes_force.quiver(0, 0, 0, 0, force_max*0.2, 0, color='green', linewidth=1, arrow_length_ratio=0.1),
            self.axes_force.quiver(0, 0, 0, 0, 0, force_max*0.2, color='blue', linewidth=1, arrow_length_ratio=0.1)
        ]
        self.ref_axes_torque = [
            self.axes_torque.quiver(0, 0, 0, torque_max*0.2, 0, 0, color='red', linewidth=1, arrow_length_ratio=0.1),
            self.axes_torque.quiver(0, 0, 0, 0, torque_max*0.2, 0, color='green', linewidth=1, arrow_length_ratio=0.1),
            self.axes_torque.quiver(0, 0, 0, 0, 0, torque_max*0.2, color='blue', linewidth=1, arrow_length_ratio=0.1)
        ]
        
        # Labels for the reference axes
        self.axes_force.text(force_max*0.22, 0, 0, "X", color='red')
        self.axes_force.text(0, force_max*0.22, 0, "Y", color='green')
        self.axes_force.text(0, 0, force_max*0.22, "Z", color='blue')
        
        self.axes_torque.text(torque_max*0.22, 0, 0, "X", color='red')
        self.axes_torque.text(0, torque_max*0.22, 0, "Y", color='green')
        self.axes_torque.text(0, 0, torque_max*0.22, "Z", color='blue')
        
        # Initialize force and torque arrows (to be updated later)
        self.force_arrow = self.axes_force.quiver(0, 0, 0, 1, 0, 0, color='blue', linewidth=1, arrow_length_ratio=0.1)
        self.torque_arrow = self.axes_torque.quiver(0, 0, 0, 1, 0, 0, color='red', linewidth=1, arrow_length_ratio=0.1)
        
        # Text elements for magnitudes and components
        self.force_mag_text = self.axes_force.text2D(0.05, 0.95, "", transform=self.axes_force.transAxes)
        self.torque_mag_text = self.axes_torque.text2D(0.05, 0.95, "", transform=self.axes_torque.transAxes)
        self.force_comp_text = self.axes_force.text2D(0.05, 0.90, "", transform=self.axes_force.transAxes, fontsize=8)
        self.torque_comp_text = self.axes_torque.text2D(0.05, 0.90, "", transform=self.axes_torque.transAxes, fontsize=8)
        
        super(MplCurrentCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

class KneeFlexionExperiment(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Configuration
        self.setWindowTitle("Knee Test Bench with Force Visualization")
        self.setGeometry(100, 100, 1200, 800)

        
        # Experiment parameters
        self.flexion_angles = [0, 30, 60, 90, 120]
        self.current_angle_index = 0
        self.rotation_time = 5  # seconds
        self.lachmann_time = 8  # seconds for Lachmann test
        self.timer = QTimer()
        self.timer.timeout.connect(self.rotation_complete)
        self.seconds_timer = QTimer()
        self.seconds_timer.timeout.connect(self.update_seconds_progress)
        
        # Timer for visualization updates (every 100ms)
        self.viz_timer = QTimer()
        self.viz_timer.timeout.connect(self.update_visualization_timer)
        self.viz_timer.setInterval(30)  # 100ms for smoother updates

        
        # History for visualization
        self.history_size = 50
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

    
    def load_force_torque_data(self):
        
        """Load force and torque data from the txt file."""
        filename = "print_data.F_sensor_temp_data_79.txt"
        self.forces = []
        self.torques = []
    
        try:
            print(f"Attempting to load data from {filename}")
            with open(filename, 'r') as file:
                for line in file:
                    # Remove any whitespace and split by comma
                    values = [float(val.strip()) for val in line.strip().split(',')]
                    if len(values) >= 6:  # Ensure we have at least 6 values
                        self.forces.append(values[0:3])
                        self.torques.append(values[3:6])
            
            # Convert to numpy arrays after the file is processed
            self.forces = np.array(self.forces)
            self.torques = np.array(self.torques)

            print(f"Successfully loaded {len(self.forces)} force/torque data points.")
            # Ensure we have at least some data
            if len(self.forces) == 0:
                raise ValueError("No valid data points found in file")
                
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            # Create dummy data if file not found or empty

            self.forces = np.zeros((200,3))
            self.forces[0] = np.random.uniform(-13, 13, size=3)
            for i in range(1, 200):
                delta = np.random.uniform(-0.8, 0.8, size=3)
                self.forces[i] = self.forces[i - 1] + delta
                self.forces[i] = np.clip(self.forces[i], -13, 13)

            self.torques = np.zeros((200,3))
            self.torques[0] = np.random.uniform(-1.5, 1.5, size=3)
            for i in range(1, 200):
                delta = np.random.uniform(-0.2, 0.2, size=3)
                self.torques[i] = self.torques[i - 1] + delta
                self.torques[i] = np.clip(self.torques[i], -1.5, 1.5)

            print("Using random dummy data instead.")


        except Exception as e:
            print(f"An error occurred: {e}")
            # Create dummy data if error
            self.forces = np.random.rand(100, 3) * 10 - 5
            self.torques = np.random.rand(100, 3) * 2 - 1
            print("Using random dummy data instead.")
    
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
        
        # Rotation timer progress bar
        rotation_progress_layout = QVBoxLayout()
        self.rotation_progress_label = QLabel("Please Flex the knee to the desired flexion angle, then hold the desired positions for the shown amount of time")
        self.rotation_progress_label.setAlignment(Qt.AlignCenter)
        rotation_progress_layout.addWidget(self.rotation_progress_label)
        
        self.rotation_progress = QProgressBar()
        self.rotation_progress.setRange(0, self.rotation_time)
        self.rotation_progress.setValue(self.rotation_time)
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
        self.tabs.addTab(self.tab1, "Current Data")
        self.tabs.addTab(self.tab2, "Current + Previous Data")
        self.tabs.addTab(self.tab3, "bone viz")


        # Add content to the first tab
        tab1_layout = QVBoxLayout()
        #tab1_layout.addWidget(QLabel("This is Tab 1"))

          # Add force/torque visualization
        viz_label_1 = QLabel("Force & Torque Visualization")
        viz_label_1.setAlignment(Qt.AlignCenter)
        viz_label_1.setFont(QFont("Arial", 12, QFont.Bold))
        #left_layout.addWidget(viz_label)
        tab1_layout.addWidget(viz_label_1)
        
        # Create matplotlib visualization
        self.canvas_current = MplCurrentCanvas(width=4, height=4)
        #left_layout.addWidget(self.canvas)
        tab1_layout.addWidget(self.canvas_current)

        self.tab1.setLayout(tab1_layout)

        # Add content to the second tab
        tab2_layout = QVBoxLayout()
        
        # Add force/torque visualization
        viz_label_2 = QLabel("Force & Torque Visualization")
        viz_label_2.setAlignment(Qt.AlignCenter)
        viz_label_2.setFont(QFont("Arial", 12, QFont.Bold))
        tab2_layout.addWidget(viz_label_2)
        
        # Create matplotlib visualization
        self.canvas_history = MplCanvas(width=4, height=4)
        tab2_layout.addWidget(self.canvas_history)
        
        self.tab2.setLayout(tab2_layout)
        
         # test bone tab
        tab3_layout = QVBoxLayout()
        
        # Create 3D GL View Widget for bone visualization
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=900, elevation=30, azimuth=-55)
        self.gl_view.setMinimumHeight(400)

        # Add axes for reference
        self.axes = gl.GLAxisItem()
        self.axes.setSize(100, 100, 100)
        self.gl_view.addItem(self.axes)

        # Separate buttons for loading bones
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
        self.force_arrow = gl.GLLinePlotItem(width=2, color=(1, 0, 0, 1))  # Blue for force
        self.torque_arrow = gl.GLLinePlotItem(width=2, color=(1, 0, 0, 1))  # Red for torque
        
        # Initialize the arrows with dummy data so they're visible
        initial_pos = np.array([[0, 0, 0], [200, 200, 200]])
        self.force_arrow.setData(pos=initial_pos, color=(1, 0, 0, 1), width=3, antialias=True)
        self.force_arrow.setGLOptions('opaque') 
        self.force_arrow.setDepthValue(-10)

        self.gl_view.addItem(self.force_arrow)

        # Connect tab change signal
        self.tabs.currentChanged.connect(self.on_tab_changed)
        tab3_layout.addWidget(self.gl_view)
        tab3_layout.addLayout(bone_load_layout)
        self.tab3.setLayout(tab3_layout)


        # Timer for bone animation updates
        self.bone_timer = QTimer()
        self.bone_timer.timeout.connect(self.update_bones)
        self.bone_timer.setInterval(25)  # 50ms for 20 fps
        
        # Initialize bone data generator
        self.bone_data_generator = BoneDataGenerator()
        
        # Add a toggle for automatic bone movement
        self.auto_bone_movement = QPushButton("Start Bone Animation")
        self.auto_bone_movement.setCheckable(True)
        self.auto_bone_movement.clicked.connect(self.toggle_bone_animation)
        
        # Add this button to your UI, for example in tab3_layout
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
        self.start_button.setFixedHeight(60)
        
        # Next Angle Button
        self.next_button = QPushButton("Next Angle")
        self.next_button.clicked.connect(self.next_angle)
        self.next_button.setEnabled(False)
        self.next_button.setFixedHeight(60)

        # Next Angle Label
        self.next_label = QLabel("test1")
        font = self.next_label.font()
        font.setPointSize(12)
        self.next_label.setFont(font)
        self.next_label.setAlignment(Qt.AlignTop)

        # Rotate Button
        self.rotate_button = QPushButton("Hold Flexion for 5 s")
        self.rotate_button.clicked.connect(self.start_rotation)
        self.rotate_button.setEnabled(False)
        self.rotate_button.setFixedHeight(60)

        # Varus Button
        self.varus_button = QPushButton("Apply Varus Load for 5 s")
        self.varus_button.clicked.connect(self.start_varus)
        self.varus_button.setEnabled(False)
        self.varus_button.setFixedHeight(60)

        # Valgus Button
        self.valgus_button = QPushButton("Apply Valgus Load for 5 s")
        self.valgus_button.clicked.connect(self.start_valgus)
        self.valgus_button.setEnabled(False)
        self.valgus_button.setFixedHeight(60)

        # IR Button
        self.internal_rot_button = QPushButton("Rotate Tibia internally for 5 s")
        self.internal_rot_button.clicked.connect(self.start_internal_rot)
        self.internal_rot_button.setEnabled(False)
        self.internal_rot_button.setFixedHeight(60)

        # ER Button
        self.external_rot_button = QPushButton("Rotate Tibia externally for 5 s")
        self.external_rot_button.clicked.connect(self.start_external_rot)
        self.external_rot_button.setEnabled(False)
        self.external_rot_button.setFixedHeight(60)

        # Lachmann Test Button - New addition
        self.lachmann_button = QPushButton("Perform Lachmann Test for 8 s")
        self.lachmann_button.clicked.connect(self.start_lachmann)
        self.lachmann_button.setEnabled(False)
        self.lachmann_button.setFixedHeight(60)


        record_data_label = QLabel("Record Data")
        record_data_label.setAlignment(Qt.AlignCenter)
        
        # Image frame
        self.image_frame = QFrame()
        self.image_frame.setLineWidth(2)
        self.image_frame.setMinimumSize(300, 250)
        
        image_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label)
        self.image_frame.setLayout(image_layout)
        
        # Layout arrangement
        subsub_layout = QHBoxLayout()
        subsub_layout.addWidget(self.start_button)
        subsub_layout.addWidget(self.next_button)

        right_layout.addLayout(subsub_layout, 0, 0)
        right_layout.addWidget(self.next_label, 0, 1)
        right_layout.addWidget(self.image_frame, 2, 1, 5, 1)
        right_layout.addWidget(record_data_label, 1, 0, 2, 1)
        right_layout.addWidget(self.rotate_button, 3, 0)
        right_layout.addWidget(self.varus_button, 4, 0)
        right_layout.addWidget(self.valgus_button, 5, 0)
        right_layout.addWidget(self.internal_rot_button, 6, 0)
        right_layout.addWidget(self.external_rot_button, 7, 0)
        right_layout.addWidget(self.lachmann_button, 8, 0)  # Add the Lachmann test button
        
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
        self.overall_progress.setRange(0, len(self.flexion_angles))
        self.overall_progress.setValue(0)
        self.overall_progress.setTextVisible(True)
        self.overall_progress.setFixedHeight(60)
        self.overall_progress.setFormat("%v/%m angles completed")
        
        # Set green color for the overall progress bar
        self.overall_progress.setStyleSheet("QProgressBar {border: 1px solid grey; border-radius: 3px; text-align: center;}"
                                           "QProgressBar::chunk {background-color: #4CAF50; width: 10px;}")
        
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
        # Only update the bones if the bone tab is active
        if self.tabs.currentIndex() != 2:
            return
            
        # Get updated bone position and quaternion data
        bone_data = self.bone_data_generator.update(0.02)  # 50ms = 0.05s
        
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

        # Also update forces if experiment is running
        if self.experiment_running:
            self.update_bone_forces(self.current_data_index)

    def update_femur_with_data(self, position, quaternion):
        # Don't remove and recreate mesh - just update its transformation
        
        # Convert quaternion to rotation matrix
        q = np.array(quaternion)
        q = q / np.sqrt(np.sum(q * q))
        w, x, y, z = q
        
        # Create rotation matrix
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y, 0],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x, 0],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y, 0],
            [0, 0, 0, 1]
        ])
        
        # Apply translation
        T = np.eye(4)
        T[0:3, 3] = position
        
        # Apply combined transform
        transform = np.dot(T, R)
        
        # Update the mesh transformation
        self.femur_mesh.setTransform(transform)

    def update_tibia_with_data(self, position, quaternion):
        pivot_point = np.array([100, 0, 0])  # Define the specific pivot point relative to the tibia's local coordinates
        
        # Convert quaternion to rotation matrix
        q = np.array(quaternion)
        q = q / np.sqrt(np.sum(q * q))
        w, x, y, z = q
        
        # Create rotation matrix
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y, 0],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x, 0],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y, 0],
            [0, 0, 0, 1]
        ])
        
        # Create translation matrices
        T_to_origin = np.eye(4)
        T_to_origin[0:3, 3] = -pivot_point  # Move pivot point to origin
        
        T_from_origin = np.eye(4)
        T_from_origin[0:3, 3] = pivot_point  # Move back from origin
        
        T_position = np.eye(4)
        T_position[0:3, 3] = position  # Final position
        
        # Apply combined transform: translate to origin, rotate, translate back, then to final position
        transform = np.dot(T_position, np.dot(T_from_origin, np.dot(R, T_to_origin)))
        
        # Apply the transform to the mesh
        self.tibia_mesh.setTransform(transform)
        
    def update_visualization_timer(self):
        """Called by timer to update visualization"""
        if self.experiment_running:
            self.current_data_index = (self.current_data_index + 1) % len(self.forces)
            
            # Check which tab is currently active and only update the relevant visualization
            current_tab = self.tabs.currentIndex()
            
            if current_tab == 0:  # Current Data tab
                force = self.forces[self.current_data_index].copy()
                torque = self.torques[self.current_data_index].copy()
                self.update_current_visualization(force, torque)
            elif current_tab == 1:  # History tab
                # Add to history
                force = self.forces[self.current_data_index].copy()
                torque = self.torques[self.current_data_index].copy()
                self.force_history.append(force)
                self.torque_history.append(torque)
                
                # Keep history to specified size
                if len(self.force_history) > self.history_size:
                    self.force_history.pop(0)
                    self.torque_history.pop(0)
                    
                self.update_history_visualization()
            elif current_tab == 2:  # Bone visualization tab
                self.update_bone_forces(self.current_data_index)
        

    
    def update_visualization(self, data_index=0):
        """Update the appropriate visualization based on current tab."""
        # Make sure we have data
        if len(self.forces) == 0 or len(self.torques) == 0:
            return
            
        # Get current data point
        idx = data_index % len(self.forces)
        force = self.forces[idx].copy()
        torque = self.torques[idx].copy()
        
        # Check which tab is currently active
        current_tab = self.tabs.currentIndex()
        
        if current_tab == 0:  # Current Data tab
            self.update_current_visualization(force, torque)
        elif current_tab == 1:  # History tab
            # Add to history
            self.force_history.append(force)
            self.torque_history.append(torque)
            
            # Keep history to specified size
            if len(self.force_history) > self.history_size:
                self.force_history.pop(0)
                self.torque_history.pop(0)
                
            self.update_history_visualization()
        elif current_tab == 2:  # Bone visualization tab
            self.update_bone_forces(data_index)

        
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
        self.canvas_current.force_mag_text.set_text(f"Force Mag: {force_mag:.2f}N")
        self.canvas_current.torque_mag_text.set_text(f"Torque Mag: {torque_mag:.2f}Nm")
        self.canvas_current.force_comp_text.set_text(f"Fx: {force[0]:.2f}, Fy: {force[1]:.2f}, Fz: {force[2]:.2f}")
        self.canvas_current.torque_comp_text.set_text(f"Tx: {torque[0]:.2f}, Ty: {torque[1]:.2f}, Tz: {torque[2]:.2f}")
        
        # Redraw the canvas
        self.canvas_current.draw()
    
    def update_history_visualization(self):
        """Update the force/torque visualization with history data."""
        # We need to remove old arrows first
        for arrow in self.canvas_history.force_arrows:
            arrow.remove()
        for arrow in self.canvas_history.torque_arrows:
            arrow.remove()
            
        # Clear the arrow lists
        self.canvas_history.force_arrows = []
        self.canvas_history.torque_arrows = []
        
        # Plot history with color gradient (older = more transparent)
        cmap_force = plt.get_cmap('Blues')
        cmap_torque = plt.get_cmap('PuRd')
        
        # Calculate max magnitudes for scaling
        force_mags = np.sqrt(np.sum(np.array(self.force_history)**2, axis=1))
        torque_mags = np.sqrt(np.sum(np.array(self.torque_history)**2, axis=1))
        
        max_force_mag = np.max(force_mags) if len(force_mags) > 0 else 1
        max_torque_mag = np.max(torque_mags) if len(torque_mags) > 0 else 1
        
        # Plot arrows with more visible formatting
        for i, (hist_force, hist_torque) in enumerate(zip(self.force_history, self.torque_history)):
            # Calculate color and alpha based on position in history
            alpha = 0.3 + 0.7 * (i / max(1, len(self.force_history) - 1))
            color_idx = i / max(1, len(self.force_history) - 1)
            
            # Force arrow
            force_mag = np.sqrt(np.sum(hist_force**2))
            width_scale = max(0.5, 0.5 + 2.5 * (force_mag / max_force_mag) if max_force_mag > 0 else 0.5)
            
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
        
        # Display magnitudes of the current force/torque
        current_force = self.force_history[-1] if self.force_history else None
        current_torque = self.torque_history[-1] if self.torque_history else None
        
        if current_force is not None and current_torque is not None:
            force_mag = np.sqrt(np.sum(current_force**2))
            torque_mag = np.sqrt(np.sum(current_torque**2))
            
            # Update text instead of recreating
            self.canvas_history.force_mag_text.set_text(f"Force Mag: {force_mag:.2f}N")
            self.canvas_history.torque_mag_text.set_text(f"Torque Mag: {torque_mag:.2f}Nm")
            self.canvas_history.force_comp_text.set_text(
                f"Fx: {current_force[0]:.2f}, Fy: {current_force[1]:.2f}, Fz: {current_force[2]:.2f}"
            )
            self.canvas_history.torque_comp_text.set_text(
                f"Tx: {current_torque[0]:.2f}, Ty: {current_torque[1]:.2f}, Tz: {current_torque[2]:.2f}"
            )
        
        # Redraw the canvas
        self.canvas_history.draw()

        
    def update_display(self):
        current_angle = self.flexion_angles[self.current_angle_index]
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

    
    def reset_buttons_and_labels(self):
        # Disable all buttons
        self.start_button.setEnabled(True)
        self.next_button.setEnabled(False)
        self.rotate_button.setEnabled(False)
        self.varus_button.setEnabled(False)
        self.valgus_button.setEnabled(False)
        self.internal_rot_button.setEnabled(False)
        self.external_rot_button.setEnabled(False)
        
        # Stop visualization timer
        if self.viz_timer.isActive():
            self.viz_timer.stop()
            
        # Reset experiment running flag
        self.experiment_running = False


    def update_bone_forces(self, data_index=0):
        """Update the force/torque visualization in 3D bone view"""
        # Skip if we're not on the bone visualization tab
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
        tibia_pos[2] -=40
        
        # Create line paths for the arrows
        force_path = np.array([
            tibia_pos,
            tibia_pos + force_scaled
        ])
        
        # Update the arrows - pyqtgraph already uses efficient updates here
        self.force_arrow.setData(pos=force_path, color=(1, 0, 0, 1), width=2, antialias=True)

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
        self.current_angle = self.flexion_angles[self.current_angle_index]
        self.overall_progress.setValue(0)
        self.next_label.setText(f"Please flex knee to {self.current_angle} degrees")
        self.rotation_progress_label.show()
        self.rotation_progress.show()

        # Reset progress bar range to match rotation time (5 seconds)
        self.rotation_progress.setRange(0, self.rotation_time)
        self.rotation_progress.setValue(self.rotation_time)
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
                # Scale the image to fit the frame while maintaining aspect ratio
                pixmap = pixmap.scaled(self.image_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
        except Exception as e:
            self.image_label.setText(f"Error loading image: {str(e)}")
        
        # Set experiment running flag
        self.experiment_running = True
        
        # Enable only needed buttons
        self.start_button.setEnabled(False)
        self.next_label.show()
        self.next_button.setEnabled(False)
        self.rotate_button.setEnabled(True)
        self.lachmann_button.setEnabled(False)
        
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
        
        # Reset button states
        self.next_button.setEnabled(False)
        self.rotate_button.setEnabled(True)
        self.varus_button.setEnabled(False)
        self.valgus_button.setEnabled(False)
        self.internal_rot_button.setEnabled(False)
        self.external_rot_button.setEnabled(False)
        self.lachmann_button.setEnabled(False)

    def start_rotation(self):
        # Disable rotate button
        self.rotate_button.setEnabled(False)
        
        # Enable varus button
        self.varus_button.setEnabled(True)
        
        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.next_button.setEnabled(False)
        
    def start_varus(self):
        # Disable varus button
        self.varus_button.setEnabled(False)
        
        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.valgus_button.setEnabled(True)

    def start_valgus(self):
        # Disable valgus button
        self.valgus_button.setEnabled(False)
        
        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.internal_rot_button.setEnabled(True)

    def start_internal_rot(self):
        # Disable internal rotation button
        self.internal_rot_button.setEnabled(False)
        
        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.external_rot_button.setEnabled(True)

    def start_external_rot(self):
        # Disable external rotation button
        self.external_rot_button.setEnabled(False)
        
        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second

        # Enable appropriate next button based on where we are in the test
        if self.current_angle_index >= (len(self.flexion_angles) - 1):
            # We're at the last angle, enable Lachmann test button
            self.lachmann_button.setEnabled(True)
            self.next_button.setEnabled(False)
        else:
            # We're not at the last angle, enable next button
            self.next_button.setEnabled(True)
            self.lachmann_button.setEnabled(False)


    def start_lachmann(self):  # Fixed spelling from previous question
        # Disable Lachmann button
        self.lachmann_button.setEnabled(False)
        self.image_label.clear()
        self.next_label.hide()
        
        # Make sure progress bar is visible
        self.rotation_progress.show()
        self.rotation_progress_label.setText("Performing Lachmann Test")
        self.rotation_progress_label.show()
        
        # Set timer for Lachmann test (10s)
        self.remaining_time = self.lachmann_time
        self.rotation_progress.setValue(self.remaining_time)
        self.rotation_progress.setRange(0, self.lachmann_time)  # Update range for 10s
        self.rotation_progress.setFormat("%v seconds remaining")
        
        # Start the timer
        self.seconds_timer.start(1000)  # Update every second
        
        # Set flag to indicate we're in Lachmann test
        self.current_test_type = 'lachmann'

    
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
    
        # Check if we just completed a Lachmann test
        if self.current_test_type == 'lachmann':
            # Reset the flag
            self.current_test_type = 'none'
        
            # Now we can show experiment complete
            self.instruction_label.setText("Experiment Complete!")
            self.overall_progress.setValue(len(self.flexion_angles))
            self.image_label.clear()
        
            # Disable all buttons except start
            self.start_button.setEnabled(True)
            self.next_button.setEnabled(False)
            self.rotate_button.setEnabled(False)
            self.varus_button.setEnabled(False)
            self.valgus_button.setEnabled(False)
            self.internal_rot_button.setEnabled(False)
            self.external_rot_button.setEnabled(False)
            self.lachmann_button.setEnabled(False)
        
            # Hide instructions
            self.next_label.hide()
            self.rotation_progress_label.hide()
            self.rotation_progress.hide()
        
            # Stop visualization timer
            if self.viz_timer.isActive():
                self.viz_timer.stop()
        
            # Reset experiment running flag
            self.experiment_running = False
        elif self.current_angle_index >= (len(self.flexion_angles) - 1) and self.external_rot_button.isEnabled() == False:
            self.next_button.setEnabled(False) # End of regular experiment - enable Lachmann test


    def load_femur(self):
        try:
            # Load femur STL
            femur_vertices, femur_faces = load_stl_as_mesh("femur_simplified.stl")
            self.femur_original_vertices = femur_vertices.copy()
            
            # Scale down the vertices (do this once)
            #femur_vertices = femur_vertices * 0.9
            
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
            tibia_vertices, tibia_faces = load_stl_as_mesh("tibia_simplified.stl")
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