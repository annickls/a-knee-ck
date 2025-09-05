import sys
import numpy as np
import pandas as pd
import csv
import warnings
#from stl import mesh
import os
import glob
import time
import datetime
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFrame, 
                            QProgressBar, QGridLayout, QSplitter, QTabWidget, 
                            QSlider, QGroupBox, QTextEdit, QDialog, QDialogButtonBox)


# custom
import constants
from pathlib import Path
from plot_config1 import MplCanvas, ColoredGLAxisItem, OptimizedVarusValgusPlot
from mesh_utils import MeshUtils
from update_visualization import UpdateVisualization



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
        self.csv_path = constants.DATA_CSV
        self.last_modified_time = 0
        self.last_size = 0
        
        # Experiment parameters
        self.current_angle_index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.rotation_complete)
        self.seconds_timer = QTimer()
        self.seconds_timer.timeout.connect(self.update_seconds_progress)
        
        # Timer for visualization updates
        self.viz_timer = QTimer()
        self.viz_timer.timeout.connect(self.update_visualization_timer)
        self.viz_timer.setInterval(20)  # 20ms for smoother updates
        
        # History for visualization
        self.force_history = []
        self.torque_history = []
        self.current_data_index = 0
        
        # Experiment is running flag
        self.experiment_running = False
        
        # Initialize empty force/torque arrays
        self.forces = np.zeros((0, 3))
        self.torques = np.zeros((0, 3))
        
        # Setup UI
        self.setup_ui()
        
        self.recording = False
        self.current_recording_data = []
        self.recording_start_time = None
        self.current_test_name = ""

        self.axes_valgus_on = 0
        self.diagram_mode = "rotation"  # Can be "varus_valgus" or "rotation"
        self.diagram_start_mode = "stop"

        
        # Ensure directory exists for data files
        os.makedirs("recorded_data", exist_ok=True)
        
    def toggle_monitoring(self):
        if not self.monitoring:
        # Start monitoring real data
            self.monitoring = True
            self.start_buttoncsv.setText("Stop Real-Time Data")
            print("--- Real-Time Data Monitoring Started ---")
            
            # Find latest csv file
            root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            pattern = os.path.join(root_folder, "A-KNEE-CK", "data.csv")

            try: 
                self.csv_path = max(glob.glob(pattern), key=os.path.getmtime)
            except: 
                print(f"No file corresponding to the pattern: {pattern}")
                return

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
                
            # Start timer to check for changes (check every 20ms for more responsive updates)
            self.timercsv.start(20)
            
            # Start visualization timer
            self.viz_timer.start()

        else:
            # Stop monitoring
            self.monitoring = False
            self.timercsv.stop()
            self.viz_timer.stop()
            self.start_buttoncsv.setText("Start Recieving Data")
            print("--- Real-Time Data Recieving Stopped ---")
            self.experiment_running = False  # Disable updates when not monitoring

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
                with open(self.csv_path, 'rb') as f:
                    try:  # catch OSError in case of a one line file 
                        f.seek(-2, os.SEEK_END)
                        while f.read(1) != b'\n':
                            f.seek(-2, os.SEEK_CUR)
                    except OSError:
                        f.seek(0)
                    last_line = f.readline().decode().strip()
                        
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

                    #FT position and quaternion
                    FT_position = np.array([float(parts[21]), float(parts[22]), float(parts[23])])
                    FT_quaternion = np.array([float(parts[27]), float(parts[24]), float(parts[25]), float(parts[26])])
                    
                    # Store positions and quaternions for other methods to use
                    self.last_femur_position = femur_position
                    self.last_femur_quaternion = femur_quaternion
                    self.last_tibia_position = tibia_position
                    self.last_tibia_quaternion = tibia_quaternion
                    self.last_FT_position = FT_position
                    self.last_FT_quaternion = FT_quaternion

                    # Calculate the distance between sensor and tibia center in sensor CoSy
                    # Get Trafo of sensor and tibia CoSy
                    R_sensor = MeshUtils.quaternion_to_transform_matrix(FT_quaternion)[:3,:3]
                    R_tibia = MeshUtils.quaternion_to_transform_matrix(tibia_quaternion)[:3,:3]
                    # Calculate distances in sensor CoSy
                    sensor2tibia_sensor = R_sensor.T @ (tibia_position-FT_position)
                    tibia2center_sensor = R_sensor.T @ R_tibia @ self.distance_tibia_center
                    sensor2center_sensor = sensor2tibia_sensor+tibia2center_sensor

                    # print(f"Sensor OptiCoSy: {np.round(FT_position,3)}")
                    # print(f"Marker OptiCoSy: {np.round(tibia_position,3)}")                    
                    # print(f"Sensor zu Marker OptiCoSy: {np.round(tibia_position-FT_position,3)}")
                    # print(f"Sensor zu Marker SensorCoSy: {np.round(sensor2tibia_sensor,3)}")
                    # print(f"Marker zu Tibiazentrum: {np.round(tibia2center_sensor,3)}")

                    # calculate real torques in the knee joint from forces and torques
                    tjx = torque[0] - force[2] * sensor2center_sensor[1] + force[1] * sensor2center_sensor[2]
                    tjy = torque[1] - force[0] * sensor2center_sensor[2] - force[2] * sensor2center_sensor[0]
                    tjz = torque[2] + force[1] * sensor2center_sensor[0] + force[0] * sensor2center_sensor[1]

                    torque[0] = tjx
                    torque[1] = tjy
                    torque[2] = tjz

                    # # Set some values to 0 for debuggung purpose
                    # torque[0] = 0
                    # torque[1] = 0
                    # torque[2] = 10

                    # Transform calculated force and torque back to init CoSy for visualization
                    force = R_sensor @ force
                    torque = R_sensor @ torque

                    # print(f"Force: {np.round(force,3)}")
                    # print(f"Torque pure: {np.round(np.array([float(parts[4]), float(parts[5]), float(parts[6])]),3)}")
                    # print(f"Sensor zu Tibia zentrum: {np.round(sensor2center_sensor, 2)}")
                    # print(f"Torque: {np.round(torque,3)}\n")
                    
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

                    # Validation for post-experiment visualization of torque along the anatomical axes
                    index_vector_start = 42
                    self.force_AP = np.array([float(parts[index_vector_start]), float(parts[index_vector_start+1]), float(parts[index_vector_start+2])])
                    self.force_medLat = np.array([float(parts[index_vector_start+3]), float(parts[index_vector_start+4]), float(parts[index_vector_start+5])])
                    self.force_proxDist = np.array([float(parts[index_vector_start+6]), float(parts[index_vector_start+7]), float(parts[index_vector_start+8])])
                    
                    # Update visualization based on current tab
                    current_tab = self.tabs.currentIndex()
                    
                    if current_tab == 0:  # Current Data tab
                        UpdateVisualization.update_current_visualization(self, force, torque)
                    elif current_tab == 1:  # History tab
                        UpdateVisualization.update_history_visualization(self)

                    elif current_tab == 2:  # Bone visualization tab
                        # Update bone positions/orientations with real data
                        if hasattr(self, 'femur_mesh') and hasattr(self, 'femur_original_vertices'):
                            MeshUtils.update_mesh_with_data(self.femur_mesh, femur_position, femur_quaternion)

                            UpdateVisualization.update_landmark_alex(self, femur_position*1000, femur_quaternion, "femur_medial")
                            UpdateVisualization.update_landmark_alex(self, femur_position*1000, femur_quaternion, "femur_lateral")
                            UpdateVisualization.update_landmark_alex(self, femur_position*1000, femur_quaternion, "femur_proximal")
                            UpdateVisualization.update_landmark_alex(self, femur_position*1000, femur_quaternion, "femur_distal")

                            #UpdateVisualization.update_landmark_alex(self, femur_position*1000, femur_quaternion, "femur_center_axis_medial")
                            #UpdateVisualization.update_landmark_alex(self, femur_position*1000, femur_quaternion, "femur_center_axis_lateral")
                            #UpdateVisualization.update_landmark_alex(self, femur_position*1000, femur_quaternion, "femur_center_medial_1")
                            #UpdateVisualization.update_landmark_alex(self, femur_position*1000, femur_quaternion, "femur_center_medial_2")
                            #UpdateVisualization.update_landmark_alex(self, femur_position*1000, femur_quaternion, "femur_center_lateral_1")
                            #UpdateVisualization.update_landmark_alex(self, femur_position*1000, femur_quaternion, "femur_center_lateral_2")
                            #UpdateVisualization.update_landmark_alex(self, femur_position*1000, femur_quaternion, "femur_sphere_center_medial")
                            #UpdateVisualization.update_landmark_alex(self, femur_position*1000, femur_quaternion, "femur_sphere_center_lateral")
                        
                        if hasattr(self, 'tibia_mesh') and hasattr(self, 'tibia_original_vertices'):
                            MeshUtils.update_mesh_with_data(self.tibia_mesh, tibia_position, tibia_quaternion)

                            UpdateVisualization.update_landmark_alex(self, tibia_position*1000, tibia_quaternion, "tibia_medial")
                            UpdateVisualization.update_landmark_alex(self, tibia_position*1000, tibia_quaternion, "tibia_lateral")
                            UpdateVisualization.update_landmark_alex(self, tibia_position*1000, tibia_quaternion, "tibia_proximal")
                            UpdateVisualization.update_landmark_alex(self, tibia_position*1000, tibia_quaternion, "tibia_distal")

                            #visualize landmarks for debugging
                            #UpdateVisualization.update_landmark_alex(self, tibia_position*1000, tibia_quaternion, "tibia_m1")
                            #UpdateVisualization.update_landmark_alex(self, tibia_position*1000, tibia_quaternion, "tibia_m2")
                            #UpdateVisualization.update_landmark_alex(self, tibia_position*1000, tibia_quaternion, "tibia_m3")
                            #UpdateVisualization.update_landmark_alex(self, tibia_position*1000, tibia_quaternion, "tibia_m4")

                            
                            if self.diagram_start_mode == "start":
                                # Calculate flexion/I-E/Varus-Valgus and joint gap
                                angles_new = UpdateVisualization.get_current_knee_angles()

                                # Set angles and newly calculated joint gaps
                                flexion_angle = angles_new['flexion']
                                lateral_joint_gap = angles_new['lateral_tibia_femur']
                                medial_joint_gap = angles_new['medial_tibia_femur']
                                internal_rotation_angle = angles_new['rotation'] 
                                adduction_angle = angles_new['adduction'] # test for adduction angles
                                medial_translation = angles_new['medial'] 
                                anterior_translation = angles_new['anterior'] 
                                

                                if self.diagram_mode == "varus_valgus":
                                    
                                    
                                    self.canvas_varus_valgus.update_varus_valgus_plot(
                                        flexion_angle, 
                                        lateral_joint_gap,
                                        internal_rotation_angle,
                                        adduction_angle,
                                        anterior_translation,
                                        medial_translation,
                                        self.diagram_mode, 
                                        self.diagram_point_mode)
                                    
                                    self.canvas_varus_valgus.update_varus_valgus_plot(
                                        flexion_angle, 
                                        -medial_joint_gap, 
                                        internal_rotation_angle,
                                        adduction_angle,
                                        anterior_translation,
                                        medial_translation,
                                        self.diagram_mode, 
                                        self.diagram_point_mode)

                                    
                                elif self.diagram_mode == 'rotation':  # rotation mode
                                    self.canvas_varus_valgus.update_varus_valgus_plot(
                                        flexion_angle, 
                                        internal_rotation_angle,
                                        internal_rotation_angle,
                                        adduction_angle,
                                        anterior_translation,
                                        medial_translation,
                                        self.diagram_mode, 
                                        self.diagram_point_mode)
                                elif self.diagram_mode == 'adduction':
                                    self.canvas_varus_valgus.update_varus_valgus_plot(
                                        flexion_angle, 
                                        adduction_angle,
                                        internal_rotation_angle,
                                        adduction_angle,
                                        anterior_translation,
                                        medial_translation,
                                        self.diagram_mode, 
                                        self.diagram_point_mode)
                                    
                                elif self.diagram_mode == 'anterior':
                                    
                                    self.canvas_varus_valgus.update_varus_valgus_plot(
                                        flexion_angle, 
                                        anterior_translation,
                                        internal_rotation_angle,
                                        adduction_angle,
                                        anterior_translation,
                                        medial_translation,
                                        self.diagram_mode, 
                                        self.diagram_point_mode)
                                elif self.diagram_mode == 'medial':
                                    
                                    self.canvas_varus_valgus.update_varus_valgus_plot(
                                        flexion_angle, 
                                        medial_translation,
                                        internal_rotation_angle,
                                        adduction_angle,
                                        anterior_translation,
                                        medial_translation,
                                        self.diagram_mode, 
                                        self.diagram_point_mode)
                            else:
                                test = 0
                            
                        # Update force visualization
                        UpdateVisualization.update_bone_forces(self, self.current_data_index)

                    
                    
                    # If recording is active, record this data point
                    if self.recording:
                        current_time = time.time() - self.recording_start_time
                        angles = UpdateVisualization.get_current_knee_angles()
                        # Use real bone data from CSV
                        data_point = [
                            current_time,
                            force[0], force[1], force[2],
                            torque[0], torque[1], torque[2],
                            femur_position[0], femur_position[1], femur_position[2],
                            femur_quaternion[0], femur_quaternion[1], femur_quaternion[2], femur_quaternion[3],
                            tibia_position[0], tibia_position[1], tibia_position[2],
                            tibia_quaternion[0], tibia_quaternion[1], tibia_quaternion[2], tibia_quaternion[3],
                            angles['flexion'], angles['adduction'], angles['rotation'],
                            angles['anterior_posterior'], angles['medial_lateral'], angles['proximal_distal'],
                            angles['medial_tibia_femur'], angles['lateral_tibia_femur'],
                            FT_position[0], FT_position[1], FT_position[2],
                            FT_quaternion[0], FT_quaternion[1], FT_quaternion[2], FT_quaternion[3]
                        ]
                        
                        self.current_recording_data.append(data_point)
                
                # Update last modified time and size
                self.last_modified_time = current_modified_time
                self.last_size = current_size
                
            except Exception as e:
                print(f"Error processing CSV data: {str(e)}")
                import traceback
                traceback.print_exc()
    
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
        filename = f"recorded_data/{timestamp}_{self.current_test_name}_{angle}.csv"

        # Get current knee angles
        angles = UpdateVisualization.get_current_knee_angles()
        

        # Create angle filename (same as main file but with _angles suffix)
        #relevant_filename = f"recorded_data/{timestamp}_{angle}deg_{self.current_test_name}_relevant.txt"

        # Write both files simultaneously
        #with open(filename, 'w') as main_file, open(relevant_filename, 'w') as angle_file:
        with open(filename, 'w') as main_file:    
            # Write main data
            #main_file.write("# Timestamp, Fx, Fy, Fz, Tx, Ty, Tz, FemurPosX, FemurPosY, FemurPosZ, FemurQuatW, FemurQuatX, FemurQuatY, FemurQuatZ, TibiaPosX, TibiaPosY, TibiaPosZ, TibiaQuatW, TibiaQuatX, TibiaQuatY, TibiaQuatZ, Flexion, Adduction, Rotation, Anterior_Posterior, Medial_Lateral, Proximal_Distal, Medial_Joint_Gap, Lateral_Joint_Gap, FTPosX, FTPosY, FTPosZ, TibiaPosZ, FTQuatW, FTQuatX, FTQuatY, FTQuatZ\n")
            headers = [
                "Timestamp", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz",
                "FemurPosX", "FemurPosY", "FemurPosZ", "FemurQuatW", "FemurQuatX", "FemurQuatY", "FemurQuatZ",
                "TibiaPosX", "TibiaPosY", "TibiaPosZ", "TibiaQuatW", "TibiaQuatX", "TibiaQuatY", "TibiaQuatZ",
                "Flexion", "Adduction", "Rotation", "Anterior_Posterior", "Medial_Lateral", "Proximal_Distal",
                "Medial_Joint_Gap", "Lateral_Joint_Gap", "FTPosX", "FTPosY", "FTPosZ", "FTPosZ",
                "FTQuatW", "FTQuatX", "FTQuatY", "FTQuatZ"
            ]
            main_file.write("# " + ", ".join(headers) + "\n")
            for data_point in self.current_recording_data:
                main_file.write(','.join(map(str, data_point)) + '\n')
            
            # Write angle data with timestamp and torques
            #angle_file.write("# Timestamp, Flexion, Adduction, Rotation, Translation_ap, Translation_ml, Translation_pd, Tx, Ty, Tz, Fx, Fy, Fz, test\n")
            #for data_point in self.current_recording_data:
            #    timestamp = data_point[0]  
            #    tx, ty, tz = data_point[4], data_point[5], data_point[6]  
            #    fx, fy, fz = data_point[1], data_point[2], data_point[3]  
            #    flexion = data_point[21]
                
                #angle_file.write(f"{timestamp}, {angles['flexion']}, {angles['adduction']}, {angles['rotation']}, {angles['anterior_posterior']}, {angles['medial_lateral']}, {angles['proximal_distal']}, {tx}, {ty}, {tz}, {fx}, {fy}, {fz},{flexion}\n")
        #print(f"Saved {len(self.current_recording_data)} data points to {filename}")
        print(f"Saved {len(self.current_recording_data)} data points with relevant data to {filename}")

        # Clear the recording data
        self.current_recording_data = []
    
    def on_tab_changed(self, index):
        # Update the appropriate visualization for the new tab
        if self.experiment_running and len(self.forces) > 0:
            if index == 0:  # Current Data tab
                force = self.forces[self.current_data_index].copy()
                torque = self.torques[self.current_data_index].copy()
                UpdateVisualization.update_current_visualization(self, force, torque)
            elif index == 1:  # History tab
                UpdateVisualization.update_history_visualization(self)
            elif index == 2:  # Bone visualization tab
                UpdateVisualization.update_bone_forces(self, self.current_data_index)
            print(f"Tab changed to {index}, visualization updated")

    def on_plot_tab_changed(self, index):
        print("plot tab changed")

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
        self.tabs.addTab(self.tab2, "point diagram")
        self.tabs.addTab(self.tab3, "bone visualization")

        #subtabs for contour plot
        self.plot_tabs = QTabWidget()
        self.tab_live = QWidget()
        self.tab_contour = QWidget()

        self.plot_tabs.addTab(self.tab_live, "live points")
        self.plot_tabs.addTab(self.tab_contour, "contour plot")

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
        viz_label_2 = QLabel("Filtered Point Diagram with contour lines")
        viz_label_2.setAlignment(Qt.AlignCenter)
        viz_label_2.setFont(QFont("Arial", 12, QFont.Bold))
        tab2_layout.addWidget(viz_label_2)
        
         # bone tab
        # Modified third tab layout with dynamic diagram
        tab3_layout = QVBoxLayout()

        # Create horizontal layout for bone visualization and dynamic diagram
        bone_and_diagram_layout = QHBoxLayout()

        # Left side - Bone visualization section
        bone_viz_layout = QVBoxLayout()

        # Create your GLViewWidget
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=constants.DISTANCE_BONE_VIZ, elevation=10, azimuth=90) #55 ausgangsposition
        self.gl_view.setMinimumHeight(650)
        # Add axes for reference
        self.axes = ColoredGLAxisItem(size=(100, 100, 100))  # defined colors
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
        gl_legend = self.setup_legend_widget()

        # Add text display for joint angles and translations
        angles_translations_layout = QHBoxLayout()
        self.joint_angles_text = QLabel("Joint Angles: \n Not calculated yet")
        self.joint_angles_text.setFont(QFont("Arial", 11))
        self.joint_angles_text.setAlignment(Qt.AlignLeft)
        angles_translations_layout.addWidget(self.joint_angles_text)

        # Add text display for joint translations
        self.joint_translations_text = QLabel("Joint Translations: \n Not calculated yet")
        self.joint_translations_text.setFont(QFont("Arial", 11))
        self.joint_translations_text.setAlignment(Qt.AlignLeft)
        angles_translations_layout.addWidget(self.joint_translations_text)
        

        bone_viz_layout.addLayout(angles_translations_layout)
        bone_viz_layout.addWidget(gl_legend)
        bone_viz_layout.addWidget(self.gl_view)
        bone_viz_layout.addLayout(bone_load_layout)

        # Right side - Dynamic diagram section
        diagram_layout = QVBoxLayout()
        tab_live_layout = QVBoxLayout()
        self.canvas_varus_valgus = OptimizedVarusValgusPlot(self, width=600, height=700)
        tab_live_layout.addWidget(self.canvas_varus_valgus)
        self.tab_live.setLayout(tab_live_layout)

        # second tab
        tab_contour_plot_layout = QVBoxLayout()
     
        # Create matplotlib visualization
        self.canvas_contour_plot = MplCanvas(width=4, height=8, mode="varus_valgus")
        self.canvas_contour_plot.ax.invert_yaxis()
        tab_contour_plot_layout.addWidget(self.canvas_contour_plot)

        contour_buttons_sublayout = QHBoxLayout()
          # Add contour calculation button
        self.contour_button = QPushButton("Calculate Contour Plot")
        self.contour_button.clicked.connect(self.calculate_and_plot_contours)
        self.contour_button.setMinimumHeight(constants.BUTTON_HEIGHT_3)
        self.contour_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        contour_buttons_sublayout.addWidget(self.contour_button)
        
        self.save_plot_button = QPushButton("Save Plot")
        self.save_plot_button.clicked.connect(self.save_current_plot)
        contour_buttons_sublayout.addWidget(self.save_plot_button)

        tab_contour_plot_layout.addLayout(contour_buttons_sublayout)

        self.tab_contour.setLayout(tab_contour_plot_layout)


        self.plot_tabs.currentChanged.connect(self.on_plot_tab_changed)
        diagram_layout.addWidget(self.plot_tabs)



        diagram_buttons_layout = QHBoxLayout()

        self.diagram_axes_rotation_button = QPushButton("rotation [°]")
        self.diagram_axes_rotation_button.setFixedHeight(constants.BUTTON_HEIGHT_3)
        self.diagram_axes_rotation_button.clicked.connect(self.toggle_diagram_axes_rotation)
        diagram_buttons_layout.addWidget(self.diagram_axes_rotation_button)

        self.diagram_axes_adduction_button = QPushButton("var/val [°]")
        self.diagram_axes_adduction_button.setFixedHeight(constants.BUTTON_HEIGHT_3)
        self.diagram_axes_adduction_button.clicked.connect(self.toggle_diagram_axes_adduction)
        diagram_buttons_layout.addWidget(self.diagram_axes_adduction_button)

        self.diagram_axes_joint_gaps_button = QPushButton("joint gaps (var/val)")
        self.diagram_axes_joint_gaps_button.setFixedHeight(constants.BUTTON_HEIGHT_3)
        self.diagram_axes_joint_gaps_button.clicked.connect(self.toggle_diagram_axes_joint_gaps)
        diagram_buttons_layout.addWidget(self.diagram_axes_joint_gaps_button)


        diagram_buttons_translations_layout = QHBoxLayout()

        self.diagram_axes_anterior_button = QPushButton("anterior/posterior translation")
        self.diagram_axes_anterior_button.setFixedHeight(constants.BUTTON_HEIGHT_3)
        self.diagram_axes_anterior_button.clicked.connect(self.toggle_diagram_axes_anterior)
        diagram_buttons_translations_layout.addWidget(self.diagram_axes_anterior_button)

        self.diagram_axes_medial_button = QPushButton("medial/lateral translation")
        self.diagram_axes_medial_button.setFixedHeight(constants.BUTTON_HEIGHT_3)
        self.diagram_axes_medial_button.clicked.connect(self.toggle_diagram_axes_medial)
        diagram_buttons_translations_layout.addWidget(self.diagram_axes_medial_button)
        
        

        diagram_layout.addLayout(diagram_buttons_layout)
        diagram_layout.addLayout(diagram_buttons_translations_layout)
        #diagram_layout.addWidget(self.contour_button)

       
        

        # Add both sections to horizontal layout
        bone_and_diagram_layout.addLayout(bone_viz_layout, 2)  # Give bone viz more space (ratio 2:1)
        bone_and_diagram_layout.addLayout(diagram_layout, 1)

        

        # Connect tab change signal
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Add all components to main tab layout
        #tab3_layout.addLayout(angles_translations_layout)
        tab3_layout.addLayout(bone_and_diagram_layout)
        self.tab3.setLayout(tab3_layout)

        

        left_layout.addWidget(self.tabs)
        self.left_widget.setLayout(left_layout)
        bottom_splitter.addWidget(self.left_widget)
        
        # Right part: Control buttons
        right_widget = QWidget()
        right_layout = QGridLayout()
        
        # Start Experiment Button
        self.start_button = QPushButton("Start Experiment")
        self.start_button.clicked.connect(self.start_experiment)
        self.start_button.setFixedHeight(constants.BUTTON_HEIGHT)
        
        # Next Angle Button
        self.next_button = QPushButton("Next Round")
        self.next_button.clicked.connect(self.next_angle)
        self.next_button.setEnabled(False)
        self.next_button.setFixedHeight(constants.BUTTON_HEIGHT)

        # neutral axis button
        self.neutral_axis_button = QPushButton("Neutral Axis")
        self.neutral_axis_button.clicked.connect(self.neutral_axis)
        self.neutral_axis_button.setEnabled(False)
        self.neutral_axis_button.setFixedHeight(constants.BUTTON_HEIGHT)

        # Varus Valgus Button 
        self.varus_button = QPushButton("Apply Varus/Valgus Load")
        self.varus_button.clicked.connect(self.start_varus)
        self.varus_button.setEnabled(False)
        self.varus_button.setFixedHeight(constants.BUTTON_HEIGHT)

        # Rotation Button (weird naming because of quick changes)
        self.valgus_button = QPushButton("Rotate Tibia int/ext")
        self.valgus_button.clicked.connect(self.start_valgus)
        self.valgus_button.setEnabled(False)
        self.valgus_button.setFixedHeight(constants.BUTTON_HEIGHT)

        # anterior translation Button
        self.internal_rot_button = QPushButton("Translate Tibia anterior/posterior")
        self.internal_rot_button.clicked.connect(self.start_internal_rot)
        self.internal_rot_button.setEnabled(False)
        self.internal_rot_button.setFixedHeight(constants.BUTTON_HEIGHT)

        # medial translation Button
        self.external_rot_button = QPushButton("Translate Tibia medial/lateral")
        self.external_rot_button.clicked.connect(self.start_external_rot)
        self.external_rot_button.setEnabled(False)
        self.external_rot_button.setFixedHeight(constants.BUTTON_HEIGHT)

        # Lachmann Test Button - New addition
        self.lachmann_button = QPushButton("Perform Lachmann Test")
        self.lachmann_button.clicked.connect(self.start_lachmann)
        self.lachmann_button.setEnabled(False)
        self.lachmann_button.setFixedHeight(constants.BUTTON_HEIGHT)

        record_data_label = QLabel("Record Data")
        record_data_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

         # Start reading csv button
        self.start_buttoncsv = QPushButton("Start Reading")
        self.start_buttoncsv.setFixedHeight(constants.BUTTON_HEIGHT_2)
        self.start_buttoncsv.clicked.connect(self.toggle_monitoring)


        # start stop plotting
        self.diagram_start_stop_button = QPushButton("start plot")
        self.diagram_start_stop_button.setFixedHeight(constants.BUTTON_HEIGHT_2)
        self.diagram_start_mode = "stop"
        self.diagram_start_stop_button.clicked.connect(self.start_stop_diagram)

        self.diagram_toggle_bar_point_button = QPushButton("show bars")
        self.diagram_toggle_bar_point_button.setFixedHeight(constants.BUTTON_HEIGHT_2)
        self.diagram_point_mode = "points"
        self.diagram_toggle_bar_point_button.clicked.connect(self.toggle_bar_point_diagram)

        # button to clear plot
        self.diagram_clear_button = QPushButton("clear plot")
        self.diagram_clear_button.setFixedHeight(constants.BUTTON_HEIGHT_2)
        self.diagram_clear_button.clicked.connect(self.clear_diagram)
        
        self.offset_button = QPushButton("zero angles and translations")
        self.offset_button.setFixedHeight(constants.BUTTON_HEIGHT_2)
        self.offset_button.clicked.connect(self.zero_angles)


        #button to record data
        self.record_individual_button = QPushButton("Record Data")
        self.record_individual_button.setFixedHeight(constants.BUTTON_HEIGHT_2)
        self.record_individual_button.clicked.connect(self.record_individual)
        
        # Layout arrangement
        subsub_layout = QHBoxLayout()
        subsub_layout.addWidget(self.start_button)
        subsub_layout.addWidget(self.next_button)
        right_layout.addLayout(subsub_layout, 0, 0)
        right_layout.addWidget(record_data_label, 2, 0, 2, 1)
        right_layout.addWidget(self.neutral_axis_button, 3, 0)
        right_layout.addWidget(self.varus_button, 4, 0)
        right_layout.addWidget(self.valgus_button, 5, 0)
        right_layout.addWidget(self.internal_rot_button, 6, 0)
        right_layout.addWidget(self.external_rot_button, 7, 0)
        right_layout.addWidget(self.lachmann_button, 8, 0)
        right_layout.addWidget(self.start_buttoncsv, 9,0, 2, 1)
        right_layout.addWidget(self.record_individual_button, 10,0, 2, 1)
        right_layout.addWidget(self.diagram_start_stop_button, 12,0, 2, 1)
        right_layout.addWidget(self.diagram_clear_button, 13,0, 2, 1)
        right_layout.addWidget(self.diagram_toggle_bar_point_button, 14,0, 2, 1)
        right_layout.addWidget(self.offset_button, 15,0, 2, 1)
        
        
        
        right_widget.setLayout(right_layout)
        bottom_splitter.addWidget(right_widget)
        bottom_splitter.setSizes([1500, 100])  # adjust sizes for left and right part
        
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
        #main_layout.addLayout(overall_progress_layout, 3, 0, 1, 2)

        # Set main layout
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Initial display
        UpdateVisualization.update_display(self)
        
        # Current test type
        self.current_test_type = 'none'

    def update_visualization_timer(self):
        """Called by timer to update visualization"""
        if self.experiment_running and len(self.forces) > 0:
            # Just update the appropriate visualization based on active tab
            current_tab = self.tabs.currentIndex()
            
            if current_tab == 0:  # Current Data tab
                force = self.forces[self.current_data_index].copy()
                torque = self.torques[self.current_data_index].copy()
                UpdateVisualization.update_current_visualization(self, force, torque)
            elif current_tab == 1:  # History tab
                UpdateVisualization.update_history_visualization(self)
            elif current_tab == 2:  # Bone visualization tab
                UpdateVisualization.update_bone_forces(self, self.current_data_index)
           
            
            # Record data if recording is active
            if self.recording:
                current_time = time.time() - self.recording_start_time
                
                """# Use real CSV data for recording
                force = self.forces[self.current_data_index].copy()
                torque = self.torques[self.current_data_index].copy()
                
                angles = UpdateVisualization.get_current_knee_angles()
                # Make sure these variables are defined in your read_csv_data method
                if hasattr(self, 'last_femur_position') and hasattr(self, 'last_femur_quaternion') and \
                hasattr(self, 'last_tibia_position') and hasattr(self, 'last_tibia_quaternion'):
                    
                    # Combine all data into one record
                    data_point = [
                        current_time,
                        force[0], force[1], force[2],
                        torque[0], torque[1], torque[2],
                        self.last_femur_position[0], self.last_femur_position[1], self.last_femur_position[2],
                        self.last_femur_quaternion[0], self.last_femur_quaternion[1], 
                        self.last_femur_quaternion[2], self.last_femur_quaternion[3],
                        self.last_tibia_position[0], self.last_tibia_position[1], self.last_tibia_position[2],
                        self.last_tibia_quaternion[0], self.last_tibia_quaternion[1], 
                        self.last_tibia_quaternion[2], self.last_tibia_quaternion[3],
                        angles['flexion'], angles['adduction'], angles['rotation'],
                        angles['anterior_posterior'], angles['medial_lateral'], angles['proximal_distal'],
                        angles['medial_tibia_femur'], angles['lateral_tibia_femur']
                    ]
                    
                    self.current_recording_data.append(data_point)"""
        
    def update_visualization(self, data_index=0):
        """Update only the active visualization tab"""
        current_tab = self.tabs.currentIndex()
        
        if not self.experiment_running or len(self.forces) == 0:
            return
            
        idx = data_index % len(self.forces)
        force = self.forces[idx].copy()
        torque = self.torques[idx].copy()
        
        update_methods = {
            0: UpdateVisualization.update_current_visualization(self, force, torque),
            1: UpdateVisualization.update_history_visualization(self),
            2: UpdateVisualization.update_bone_forces(self, self.current_data_index)
        }
        
        if current_tab in update_methods:
            if current_tab == 1:
                update_methods[current_tab]()
            else:
                update_methods[current_tab](force, torque)

    def start_experiment(self):
        self.current_angle_index = 0
        self.current_angle = constants.FLEXION_ANGLES[self.current_angle_index]
        self.overall_progress.setValue(0)
        #self.rotation_progress_label.show()
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
        
        # Set experiment running flag
        self.experiment_running = True
        
        # Enable only needed buttons
        #self.next_label.show()
        self.start_button.setEnabled(False)
        self.neutral_axis_button.setEnabled(True)
        
        # Start visualization timer immediately and keep it running throughout the experiment
        if not self.viz_timer.isActive():
            self.viz_timer.start()

        # Update visualization initially
        self.update_visualization(0)
        
        # Also update bone forces explicitly
        UpdateVisualization.update_bone_forces(self, 0)

    
    def next_angle(self):
        self.current_angle_index += 1
        UpdateVisualization.update_display(self)
        self.next_button.setEnabled(False)
        self.neutral_axis_button.setEnabled(True)

    def neutral_axis(self):
        self.neutral_axis_button.setEnabled(False)
        self.remaining_time = constants.HOLD_TIME
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  
        self.varus_button.setEnabled(True)
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
        self.start_recording(f"rot") # Start recording data

    def start_internal_rot(self):
        self.internal_rot_button.setEnabled(False) # Disable internal rotation button
        self.remaining_time = constants.HOLD_TIME
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.external_rot_button.setEnabled(True)
        self.start_recording(f"anterior")# Start recording data

    def start_external_rot(self):
        self.external_rot_button.setEnabled(False) # Disable external rotation button
        self.remaining_time = constants.HOLD_TIME
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.start_recording(f"medial") # Start recording data

        self.lachmann_button.setEnabled(True)

    def start_lachmann(self):  
        self.lachmann_button.setEnabled(False)
        #self.rotation_progress_label.setText("Performing Lachmann Test")
        #self.rotation_progress_label.show()
        self.remaining_time = constants.LACHMANN_TIME # Set timer for Lachmann test
        self.rotation_progress.setValue(self.remaining_time)
        self.rotation_progress.setRange(0, constants.LACHMANN_TIME)
        self.rotation_progress.setFormat("%v seconds remaining")
        self.seconds_timer.start(1000)  # Start the timer and update every second
        self.start_recording(f"lachmann") # Start recording data
        self.current_test_type = 'none' # Set flag to indicate we're in Lachmann test
        self.next_button.setEnabled(True) 
    
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
            self.start_button.setEnabled(True) # Enable start button again

            # Hide instructions
            #self.rotation_progress_label.hide()
            self.rotation_progress.hide()
        
            # Stop visualization timer
            if self.viz_timer.isActive():
                self.viz_timer.stop()
        
            # Reset experiment running flag
            self.experiment_running = False

        elif self.current_angle_index >= (len(constants.FLEXION_ANGLES) - 1) and self.external_rot_button.isEnabled() == False:
            self.next_button.setEnabled(False) # End of regular experiment - enable Lachmann test

    def record_individual(self):
        self.remaining_time = constants.HOLD_INDIVIDUAL
        self.rotation_progress.setValue(constants.HOLD_INDIVIDUAL)
        self.seconds_timer.start(1000)  
        self.start_recording(f"individual") # Start recording data

    def load_femur(self):
        try:
            # Load femur STL
            femur_vertices, femur_faces = MeshUtils.load_stl_as_mesh(constants.FEMUR)
            warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
            self.femur_original_vertices = femur_vertices.copy()
            
            # Store vertices in a numpy array for faster operations
            femur_vertices = np.array(femur_vertices, dtype=np.float32)
            femur_faces = np.array(femur_faces, dtype=np.uint32)
            
            # Check for and fix invalid vertices
            # Replace NaN values with zeros
            femur_vertices = np.nan_to_num(femur_vertices)

            
            #--------------------------------------
            #          Kabsch
            #--------------------------------------

            # Run kabsch algorithm
            current_folder = os.path.dirname(os.path.abspath(__file__))
            yaml_path = os.path.join(current_folder, "data_for_gui/marker_coordinates.yaml")
            translation, rotation = MeshUtils.kabsch(yaml_path, "femur")
            femur_vertices_centered = femur_vertices + translation
            femur_vertices_transformed = (rotation@(femur_vertices_centered.T)).T

            # Build KD-Tree from vertices
            face_centroids = femur_vertices[femur_faces].mean(axis=1)
            self.femur_kdtree = cKDTree(face_centroids)
            # Add kabsch to class cause we need it for the gap measurements
            self.femur_kabsch_rot = rotation
            self.femur_kabsch_trans = translation

            # Create mesh item with the repositioned and rotated vertices
            # Set up the mesh with proper shading
            self.femur_mesh = gl.GLMeshItem(
                vertexes=femur_vertices_transformed,
                faces=femur_faces,
                smooth=True,
                drawEdges=False,
                color=(112, 128, 144, 255),
                computeNormals=True,
                shader='shaded',
                glOptions='opaque'
            )

            # Add the mesh to your GLViewWidget
            self.gl_view.addItem(self.femur_mesh)

            # Configure the main camera view
            #self.gl_view.setCameraPosition(distance=constants.DISTANCE_BONE_VIZ, elevation=30, azimuth=45)


            # Configure lighting direction - this is the key part
            # This positions the light coming from the opposite side
            # (Negative values place the light on the opposite axis)
            self.gl_view.opts['lightPosition'] = np.array([-10, -10, -500])  # x, y, z coordinates

            # You can also adjust these lighting parameters for better contrast
            self.gl_view.opts['ambient'] = 0.3     # Amount of ambient light (0-1)
            self.gl_view.opts['diffuse'] = 0.8     # Amount of diffuse light (0-1)
            self.gl_view.opts['specular'] = 0.2    # Amount of specular light (0-1)
            self.gl_view.opts['shininess'] = 50    # Controls the sharpness of specular highlights
            
            # Set up transform matrix (initialize once)
            self.femur_transform = np.identity(4, dtype=np.float32)
            
            # Disable load button
            self.load_femur_button.setEnabled(False)
            self.load_femur_button.setText("Femur Loaded")

            femur_medial = constants.FEMUR_MEDIAL
            femur_lateral = constants.FEMUR_LATERAL
            femur_proximal = constants.FEMUR_PROXIMAL
            femur_distal = constants.FEMUR_DISTAL
            #femur_center_axis_medial = constants.CENTER_AXIS_MEDIAL
            #femur_center_axis_lateral = constants.CENTER_AXIS_LATERAL
            #femur_center_medial_1 = constants.SURFACE_MEDIAL_1
            #femur_center_medial_2 = constants.SURFACE_MEDIAL_2
            #femur_center_lateral_1 = constants.SURFACE_LATERAL_1
            #femur_center_lateral_2 = constants.SURFACE_LATERAL_2
            #femur_sphere_center_medial = constants.TEST_POINT_MEDIAL
            #femur_sphere_center_lateral = constants.TEST_POINT_LATERAL

            femur_medial_rot = rotation@(femur_medial+translation)
            femur_lateral_rot = rotation@(femur_lateral+translation)
            femur_proximal_rot = rotation@(femur_proximal+translation)
            femur_distal_rot = rotation@(femur_distal+translation)
            #femur_center_axis_medial_rot = rotation@(femur_center_axis_medial+translation)
            #femur_center_axis_lateral_rot = rotation@(femur_center_axis_lateral+translation)
            #femur_center_medial_1_rot = rotation@(femur_center_medial_1+translation)
            #femur_center_medial_2_rot = rotation@(femur_center_medial_2+translation)
            #femur_center_lateral_1_rot = rotation@(femur_center_lateral_1+translation)
            #femur_center_lateral_2_rot = rotation@(femur_center_lateral_2+translation)
            #femur_sphere_center_medial_rot = rotation@(femur_sphere_center_medial+translation)
            #femur_sphere_center_lateral_rot = rotation@(femur_sphere_center_lateral+translation)


            UpdateVisualization.add_landmark(self, femur_medial_rot, "femur_medial")
            UpdateVisualization.add_landmark(self, femur_lateral_rot, "femur_lateral")
            UpdateVisualization.add_landmark(self, femur_proximal_rot, "femur_proximal")
            UpdateVisualization.add_landmark(self, femur_distal_rot, "femur_distal")
            #UpdateVisualization.add_landmark(self, femur_center_axis_medial_rot, "femur_center_axis_medial")
            #UpdateVisualization.add_landmark(self, femur_center_axis_lateral_rot, "femur_center_axis_lateral")
            #UpdateVisualization.add_landmark(self, femur_center_medial_1_rot, "femur_center_medial_1")
            #UpdateVisualization.add_landmark(self, femur_center_medial_2_rot, "femur_center_medial_2")
            #UpdateVisualization.add_landmark(self, femur_center_lateral_1_rot, "femur_center_lateral_1")
            #UpdateVisualization.add_landmark(self, femur_center_lateral_2_rot, "femur_center_lateral_2")
            #UpdateVisualization.add_landmark(self, femur_sphere_center_medial_rot, "femur_sphere_center_medial")
            #UpdateVisualization.add_landmark(self, femur_sphere_center_lateral_rot, "femur_sphere_center_lateral")
         
        except Exception as e:
            print(f"Error loading femur: {e}")
            import traceback
            traceback.print_exc()
            self.load_femur_button.setText("Error")
  
    def load_tibia(self):
        try:
            # Load tibia STL
            tibia_vertices, tibia_faces = MeshUtils.load_stl_as_mesh(constants.TIBIA)
            warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
            self.tibia_original_vertices = tibia_vertices.copy()
            
            # Store vertices in a numpy array for faster operations
            tibia_vertices = np.array(tibia_vertices, dtype=np.float32)
            tibia_faces = np.array(tibia_faces, dtype=np.uint32)
            
            # Check for and fix invalid vertices
            # Replace NaN values with zeros
            tibia_vertices = np.nan_to_num(tibia_vertices)
            
          
            #--------------------------------------
            #          Kabsch
            #--------------------------------------

            # Run kabsch algorithm
            current_folder = os.path.dirname(os.path.abspath(__file__))
            yaml_path = os.path.join(current_folder, "data_for_gui/marker_coordinates.yaml")
            translation, rotation = MeshUtils.kabsch(yaml_path, "tibia")
            tibia_vertices_centered = tibia_vertices + translation
            tibia_vertices_transformed = (rotation@(tibia_vertices_centered.T)).T


            # Create mesh item with the repositioned and rotated vertices
            self.tibia_mesh = gl.GLMeshItem(
                vertexes=tibia_vertices_transformed,
                faces=tibia_faces,
                smooth=True,
                drawEdges=False,
                #color = QtGui.QColor(47, 79, 79),
                color=(112, 128, 144, 255),
                computeNormals=True,
                shader='shaded',
                glOptions='opaque'
            )

            self.gl_view.opts['lightPosition'] = np.array([-10, -10, -500])  # x, y, z coordinates

            # You can also adjust these lighting parameters for better contrast
            self.gl_view.opts['ambient'] = 0.3     # Amount of ambient light (0-1)
            self.gl_view.opts['diffuse'] = 0.8     # Amount of diffuse light (0-1)
            self.gl_view.opts['specular'] = 0.2  # Amount of specular light (0-1)
            self.gl_view.opts['shininess'] = 50    # Controls the sharpness of specular highlights
            self.gl_view.addItem(self.tibia_mesh)
            
            # Store for later use
            self.tibia_verts = tibia_vertices_transformed
            self.tibia_faces = tibia_faces
            
            # Set up transform matrix (initialize once)
            self.tibia_transform = np.identity(4, dtype=np.float32)
            
            # Disable load button
            self.load_tibia_button.setEnabled(False)
            self.load_tibia_button.setText("Tibia Loaded")

            # ---------------------------
            # -   Add landmark to tibia -
            # ---------------------------
            tibia_medial = constants.TIBIA_MEDIAL
            tibia_lateral = constants.TIBIA_LATERAL
            tibia_proximal = constants.TIBIA_PROXIMAL
            tibia_distal = constants.TIBIA_DISTAL
            tibia_marker = constants.TIBIA_MARKER

            tibia_medial_rot = rotation@(tibia_medial+translation)
            tibia_lateral_rot = rotation@(tibia_lateral+translation)
            tibia_proximal_rot = rotation@(tibia_proximal+translation)
            tibia_distal_rot = rotation@(tibia_distal+translation)
            tibia_marker_rot = rotation@(tibia_marker+translation)

            self.distance_tibia_center = (tibia_proximal_rot-tibia_marker_rot)*0.001

            UpdateVisualization.add_landmark(self, tibia_medial_rot, "tibia_medial")
            UpdateVisualization.add_landmark(self, tibia_lateral_rot, "tibia_lateral")
            UpdateVisualization.add_landmark(self, tibia_proximal_rot, "tibia_proximal")
            UpdateVisualization.add_landmark(self, tibia_distal_rot, "tibia_distal")
            UpdateVisualization.add_landmark(self, tibia_marker_rot, "tibia_marker")



            #visualise marker points for debugging
            #tibia_m1 = np.array([-87.40117250193568, -90.80779189255344, 1575.7205254081575])
            #tibia_m2 = np.array([-111.04134830095568, -114.69156189192014, 1559.338514868094])
            #tibia_m3 = np.array([-124.53185834797662, -88.77439542502907, 1557.3575856843993])
            #tibia_m4 = np.array([-106.98374014215688, -72.95723968988962, 1555.5494236207694])
            #tibia_m1_rot = rotation@(tibia_m1+translation)
            #tibia_m2_rot = rotation@(tibia_m2+translation)
            #tibia_m3_rot = rotation@(tibia_m3+translation)
            #tibia_m4_rot = rotation@(tibia_m4+translation)
            #UpdateVisualization.add_landmark(self, tibia_m1_rot, "tibia_m1")
            #UpdateVisualization.add_landmark(self, tibia_m2_rot, "tibia_m2")
            #UpdateVisualization.add_landmark(self, tibia_m3_rot, "tibia_m3")
            #UpdateVisualization.add_landmark(self, tibia_m4_rot, "tibia_m4")
            #UpdateVisualization.add_landmark(self, tibia_ref_rot, "tibia_ref")

            # ---------------------------
            # -     Add CoSy to tibia   - 
            # ---------------------------
            #tibia_ursprung = np.array([-108.3848216194612,-90.25476224637612,1557.4634567569026])
            #tibia_ursprung_rot = rotation@(tibia_ursprung+translation)
            #UpdateVisualization.add_coordinate_axes(self, tibia_ursprung_rot, rotation, "tibia_ursprung")

        except Exception as e:
            print(f"Error loading tibia: {e}")
            import traceback
            traceback.print_exc()
            self.load_tibia_button.setText("Error")

    def setup_legend_widget(self):
        """Create a separate widget for the legend"""
        legend_widget = QWidget()
        legend_layout = QVBoxLayout()
        legend_layout.setSpacing(5)
        legend_layout.setContentsMargins(5, 5, 5, 5)
        
        # Combine color and text in single labels
        force_label = QLabel("● Force")
        force_label.setStyleSheet("color: red; font-size: 14px;")
        
        torque_label = QLabel("● Torque")
        torque_label.setStyleSheet("color: deepskyblue; font-size: 14px;")
        
        legend_layout.addWidget(force_label)
        legend_layout.addWidget(torque_label)
        legend_widget.setLayout(legend_layout)
        
        return legend_widget

    def toggle_diagram_axes_rotation(self):
        self.diagram_mode = "rotation"
        print(f"Diagram mode switched to: {self.diagram_mode}")

    def toggle_diagram_axes_adduction(self):
        self.diagram_mode = "adduction"
        print(f"Diagram mode switched to: {self.diagram_mode}")

    def toggle_diagram_axes_joint_gaps(self):
        self.diagram_mode = "varus_valgus"
        print(f"Diagram mode switched to: {self.diagram_mode}") 

    def toggle_diagram_axes_anterior(self):
        print("here you would see ap translation")
        self.diagram_mode = "anterior"
        print(f"Diagram mode switched to: {self.diagram_mode}") 

    def toggle_diagram_axes_medial(self):
        print("here you would see ml translation")
        self.diagram_mode = "medial"
        print(f"Diagram mode switched to: {self.diagram_mode}") 
        
    def start_stop_diagram(self):
        if self.diagram_start_mode == "stop":
            self.diagram_start_mode = "start"
            self.diagram_start_stop_button.setText("stop plot")  # Update button text   
        else:
            self.diagram_start_mode = "stop"
            self.diagram_start_stop_button.setText("start plot")  # Update button text

    def toggle_bar_point_diagram(self):
        if self.diagram_point_mode == "points":
            self.diagram_point_mode = "bars"
            self.diagram_toggle_bar_point_button.setText("show points")  # Update button text   
        else:
            self.diagram_point_mode = "points"
            self.diagram_toggle_bar_point_button.setText("show bars")  # Update button text

    def clear_diagram(self):
        self.canvas_varus_valgus.clear_data()

    def zero_angles(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recorded_data/{timestamp}_offsets.csv"
        
        # Get current angles from the correct location
        current_angles = UpdateVisualization.current_knee_angles
        
        # Open file in write mode and write the variable
        with open(filename, 'w') as file:
            file.write(f"rotation angle: {current_angles['rotation']}\n")
            file.write(f"adduction angle: {current_angles['adduction']}\n")
            file.write(f"anterior translation: {current_angles['anterior_posterior']}\n")
            file.write(f"medial translation: {current_angles['medial_lateral']}\n")
            file.write(f"proximal translation: {current_angles['proximal_distal']}")
        
        # Set the offsets using the current values
        UpdateVisualization.offset_rotation = current_angles['rotation']
        UpdateVisualization.offset_adduction = current_angles['adduction']
        UpdateVisualization.offset_anterior = current_angles['anterior_posterior']
        UpdateVisualization.offset_medial = current_angles['medial_lateral']
        UpdateVisualization.offset_proximal = current_angles['proximal_distal']
        

    def calculate_and_plot_contours(self):
        """
        Calculate and display the contour plot in the varus_valgus canvas.
        This method integrates your existing contour calculation code.
        """
        
        try:
            #recorded_data_folder = r'/home/annick/a-knee-ck/recorded_data'
            recorded_data_folder = constants.RECORDED
            
            # Find all CSV files in the folder
            csv_files = []
            for file in os.listdir(recorded_data_folder):
                if file.endswith('.csv'):
                    file_path = os.path.join(recorded_data_folder, file)
                    # Get file modification time
                    mtime = os.path.getmtime(file_path)
                    csv_files.append((file_path, mtime))
            
            if not csv_files:
                print(f"No CSV files found in folder: {recorded_data_folder}")
                return
            
            # Sort by modification time (newest first) and get the newest file
            csv_files.sort(key=lambda x: x[1], reverse=True)
            file_path = csv_files[0][0]
            
            print(f"Using newest CSV file: {os.path.basename(file_path)}")

            # Load data
            df = pd.read_csv(file_path, header=0)

            print(f"Successfully loaded data from: {file_path}")
            print(f"Data shape: {df.shape}")


            # Debug: Check the dataframe itself
            #print(f"DataFrame shape: {df.shape}")
            #print(f"DataFrame info:")
            #print(df.info())

            # Clean column names
            df.columns = df.columns.str.strip()


            # Extract force data using cleaned column headers
            fx = df['Fx']  # Using the second set of force columns
            fy = df['Fy']
            fz = df['Fz']
            

            # Extract torque data using column headers
            tx = df['Tx']  # Using the second set of torque columns
            ty = df['Ty']
            tz = df['Tz']
            

            # Extract kinematic data
            flexion = df['Flexion']
            rotation = df['Rotation']
            adduction = df['Adduction']

            # Extract gap measurements
            medial_joint_gap = df['Medial_Joint_Gap']
            lateral_joint_gap = df['Lateral_Joint_Gap']

            # Extract position measurements
            anterior = df['Anterior_Posterior']
            medial = df['Medial_Lateral']


            # Extract tibia position data
            tib_position = np.zeros((len(df), 3))
            tib_position[:, 0] = df['TibiaPosX']  # Using the second set of tibia columns
            tib_position[:, 1] = df['TibiaPosY']
            tib_position[:, 2] = df['TibiaPosZ']

            # Extract tibia quaternion data
            tib_quaternion = np.zeros((len(df), 4))
            tib_quaternion[:, 0] = df['TibiaQuatW']  # w component first
            tib_quaternion[:, 1] = df['TibiaQuatX']  # x component
            tib_quaternion[:, 2] = df['TibiaQuatY']  # y component
            tib_quaternion[:, 3] = df['TibiaQuatZ']  # z component

            """# Extract force/torque sensor position data
            ft_position = np.zeros((len(df), 3))
            ft_position[:, 0] = df['sensor_x']  # Using the second set of sensor columns
            ft_position[:, 1] = df['sensor_y']
            ft_position[:, 2] = df['sensor_z']

            # Extract force/torque sensor quaternion data
            ft_quaternion = np.zeros((len(df), 4))
            ft_quaternion[:, 0] = df['sensor_qw']  # w component first
            ft_quaternion[:, 1] = df['sensor_qx']  # x component
            ft_quaternion[:, 2] = df['sensor_qy']  # y component
            ft_quaternion[:, 3] = df['sensor_qz']  # z component"""
            
            
            fjx = fx
            fjy = fy
            fjz = fz
            tjx = tx
            tjy = ty
            tjz = tz

            # rotate forces back to sensor joint coosys (for now tibia cosys because of data)
            for i in range(len(fx)):
                rotation_tibia_plot = MeshUtils.quaternion_to_transform_matrix(tib_quaternion[i])[:3,:3]
                forces_plot = np.array([fx[i], fy[i], fz[i]])
                forces_tibia_coord = rotation_tibia_plot.T @ forces_plot
                torques_plot = np.array([tx[i], ty[i], tz[i]])
                torques_tibia_coord = rotation_tibia_plot.T @ torques_plot
                fjx[i] = forces_tibia_coord[0]
                fjy[i] = forces_tibia_coord[1]
                fjz[i] = forces_tibia_coord[2]
                tjx[i] = torques_tibia_coord[0]
                tjy[i] = torques_tibia_coord[1]
                tjz[i] = torques_tibia_coord[2]
            
            tjy=tjz # correspondng torque for adduction
            placeholder = fjz
            fjx = placeholder
            fjz = fjx


            
            
            # Configuration parameters (you can make these class attributes for easy modification)
            
            flexion_bin_size = constants.FLEXION_BIN_SIZE
            INTERPOLATION_KIND = constants.INTERPOLATION_KIND 
            SMOOTHING_FACTOR = constants.SMOOTHING_FACTOR 
            MIN_POINTS_FOR_SMOOTHING = constants.MIN_POINTS_FOR_SMOOTHING
            MOVING_AVERAGE_WINDOW = constants.MOVING_AVERAGE_WINDOW
            MOVING_AVERAGE_METHOD = constants.MOVING_AVERAGE_METHOD
            APPLY_MOVING_AVERAGE = constants.APPLY_MOVING_AVERAGE
            WEIGHT_TYPE = constants.WEIGHT_TYPE
            SIGMA_FACTOR = constants.SIGMA_FACTOR
            
            # Create bins for tjx and tjy (torque)
            tjx_min = tjx.min()
            tjx_max = tjx.max()
            tjy_min = tjy.min()
            tjy_max = tjy.max()
            fjx_min = fjx.min()
            fjx_max = fjx.max()
            fjy_min = fjy.min()
            fjy_max = fjy.max()


            if self.diagram_mode == 'rotation':    
                #bin_size = 0.2
                desired_bins = constants.BINS_ROT
                tjx_range_temp = tjx_max - tjx_min
                bin_size_temp = tjx_range_temp / desired_bins
                bin_size = round(bin_size_temp, 1)
            elif self.diagram_mode == 'varus_valgus':
                #bin_size = 0.5
                desired_bins = constants.BINS_VAR
                tjy_range_temp = tjy_max - tjy_min
                bin_size_temp = tjy_range_temp / desired_bins
                bin_size = round(bin_size_temp, 1)
            elif self.diagram_mode == 'adduction':
                desired_bins = constants.BINS_ADD
                tjy_range_temp = tjy_max - tjy_min
                bin_size_temp = tjy_range_temp / desired_bins
                bin_size = round(bin_size_temp, 1)
            elif self.diagram_mode == 'anterior':
                desired_bins = constants.BINS_ANT
                fjx_range_temp = fjx_max - fjx_min
                bin_size_temp = fjx_range_temp / desired_bins
                bin_size = round(bin_size_temp, 1)
            elif self.diagram_mode == 'medial':
                desired_bins = constants.BINS_ANT
                fjy_range_temp = fjy_max - fjy_min
                bin_size_temp = fjy_range_temp / desired_bins
                bin_size = round(bin_size_temp, 1)


            print(bin_size)
            
            tjx_bins = np.arange(tjx_min, tjx_max + bin_size, bin_size)
            tjy_bins = np.arange(tjy_min, tjy_max + bin_size, bin_size)
            fjx_bins = np.arange(fjx_min, fjx_max + bin_size, bin_size)
            fjy_bins = np.arange(fjy_min, fjy_max + bin_size, bin_size)

            
            # Create bins for flexion angles
            flexion_min = flexion.min()
            flexion_max = flexion.max()
            flexion_bins = np.arange(flexion_min, flexion_max + flexion_bin_size, flexion_bin_size)
            
            # Assign bin indices
            tjx_bin_indices = pd.cut(tjx, tjx_bins, include_lowest=True, labels=False)
            flexion_bin_indices = pd.cut(flexion, flexion_bins, include_lowest=True, labels=False)
            tjy_bin_indices = pd.cut(tjy, tjy_bins, include_lowest=True, labels=False)
            fjx_bin_indices = pd.cut(fjx, fjx_bins, include_lowest=True, labels=False)
            fjy_bin_indices = pd.cut(fjy, fjy_bins, include_lowest=True, labels=False)

            
            # Create bin centers
            bin_centers_tjx = (tjx_bins[:-1] + tjx_bins[1:]) / 2
            tjx_bin_centers = bin_centers_tjx[tjx_bin_indices.astype(int)]
         
            
            flexion_bin_centers = (flexion_bins[:-1] + flexion_bins[1:]) / 2
            flexion_bin_centers_mapped = flexion_bin_centers[flexion_bin_indices.astype(int)]

            bin_centers_tjy = (tjy_bins[:-1] + tjy_bins[1:]) / 2
            tjy_bin_centers = bin_centers_tjy[tjy_bin_indices.astype(int)]

            bin_centers_fjx = (fjx_bins[:-1] + fjx_bins[1:]) / 2
            fjx_bin_centers = bin_centers_fjx[fjx_bin_indices.astype(int)]

            bin_centers_fjy = (fjy_bins[:-1] + fjy_bins[1:]) / 2
            fjy_bin_centers = bin_centers_fjy[fjy_bin_indices.astype(int)]

           
            
            # Create DataFrame with bin indices and centers
            data_df = pd.DataFrame({
                'tjx_bin': tjx_bin_indices,
                'tjy_bin': tjy_bin_indices,
                'fjx_bin': fjx_bin_indices,
                'fjy_bin': fjy_bin_indices,
                'flexion_bin': flexion_bin_indices,
                'rotation': rotation,
                'medial_joint_gap': medial_joint_gap,
                'lateral_joint_gap': lateral_joint_gap,
                'adduction': adduction,
                'anterior': anterior,
                'medial': medial,
                'flexion': flexion,
                'tjx': tjx,
                'tjy': tjy,
                'fjx': fjx,
                'fjy': fjy,
                'tjx_bin_center': tjx_bin_centers,
                'tjy_bin_center': tjy_bin_centers,
                'fjx_bin_center': fjx_bin_centers,
                'fjy_bin_center': fjy_bin_centers,
                'flexion_bin_center': flexion_bin_centers_mapped
            })

                
            # Remove rows with NaN bin indices
            data_df = data_df.dropna()
            
            # Calculate weighted averages using your existing function
            weighted_groups = []
            
            if self.diagram_mode == 'rotation':
                for (tjx_bin_idx, flexion_bin_idx), group in data_df.groupby(['tjx_bin', 'flexion_bin']):
                    if len(group) < 1:
                        continue
                    
                    # Calculate weights based on distance from torque bin center
                    tjx_weights = self.calculate_bin_weights(
                        group['tjx'].values, 
                        group['tjx_bin_center'].values,
                        WEIGHT_TYPE, 
                        SIGMA_FACTOR,
                        bin_size
                    )
                    
                    # Calculate weighted averages
                    total_weight = np.sum(tjx_weights)
                    if total_weight > 0:
                        weighted_rotation = np.sum(group['rotation'].values * tjx_weights) / total_weight
                        weighted_flexion = np.sum(group['flexion'].values * tjx_weights) / total_weight
                        weighted_tjx = np.sum(group['tjx'].values * tjx_weights) / total_weight
                        
                        weighted_groups.append({
                            'tjx_bin': tjx_bin_idx,
                            'flexion_bin': flexion_bin_idx,
                            'rotation': weighted_rotation,
                            'flexion': weighted_flexion,
                            'tjx': weighted_tjx,
                            'n_points': len(group),
                            'effective_n': total_weight
                        })
            elif self.diagram_mode == "varus_valgus":
                for (tjy_bin_idx, flexion_bin_idx), group in data_df.groupby(['tjy_bin', 'flexion_bin']):
                    if len(group) < 1:
                        continue
                    
                    # Calculate weights based on distance from torque bin center
                    tjy_weights = self.calculate_bin_weights(
                        group['tjy'].values, 
                        group['tjy_bin_center'].values,
                        WEIGHT_TYPE, 
                        SIGMA_FACTOR,
                        bin_size
                    )
                    
                    # Calculate weighted averages
                    total_weight = np.sum(tjy_weights)
                    if total_weight > 0:
                        weighted_medial_joint_gap = np.sum(group['medial_joint_gap'].values * tjy_weights) / total_weight
                        weighted_lateral_joint_gap = np.sum(group['lateral_joint_gap'].values * tjy_weights) / total_weight
                        weighted_flexion = np.sum(group['flexion'].values * tjy_weights) / total_weight
                        weighted_tjy = np.sum(group['tjy'].values * tjy_weights) / total_weight
                        
                        weighted_groups.append({
                            'tjy_bin': tjy_bin_idx,
                            'flexion_bin': flexion_bin_idx,
                            'medial_joint_gap': weighted_medial_joint_gap,
                            'lateral_joint_gap': weighted_lateral_joint_gap,
                            'flexion': weighted_flexion,
                            'tjy': weighted_tjy,
                            'n_points': len(group),
                            'effective_n': total_weight
                        })
            elif self.diagram_mode == "adduction":
                for (tjy_bin_idx, flexion_bin_idx), group in data_df.groupby(['tjy_bin', 'flexion_bin']):
                    if len(group) < 1:
                        continue
                    
                    # Calculate weights based on distance from torque bin center
                    tjy_weights = self.calculate_bin_weights(
                        group['tjy'].values, 
                        group['tjy_bin_center'].values,
                        WEIGHT_TYPE, 
                        SIGMA_FACTOR,
                        bin_size
                    )
                    
                    # Calculate weighted averages
                    total_weight = np.sum(tjy_weights)
                    if total_weight > 0:
                        weighted_adduction = np.sum(group['adduction'].values * tjy_weights) / total_weight
                        weighted_flexion = np.sum(group['flexion'].values * tjy_weights) / total_weight
                        weighted_tjy = np.sum(group['tjy'].values * tjy_weights) / total_weight
                        
                        weighted_groups.append({
                            'tjy_bin': tjy_bin_idx,
                            'flexion_bin': flexion_bin_idx,
                            'adduction': weighted_adduction,
                            'flexion': weighted_flexion,
                            'tjy': weighted_tjy,
                            'n_points': len(group),
                            'effective_n': total_weight
                        })
            elif self.diagram_mode == "anterior":
                for (fjx_bin_idx, flexion_bin_idx), group in data_df.groupby(['fjx_bin', 'flexion_bin']):
                        if len(group) < 1:
                            continue
                        
                        # Calculate weights based on distance from torque bin center
                        fjx_weights = self.calculate_bin_weights(
                            group['fjx'].values, 
                            group['fjx_bin_center'].values,
                            WEIGHT_TYPE, 
                            SIGMA_FACTOR,
                            bin_size
                        )
                        
                        # Calculate weighted averages
                        total_weight = np.sum(fjx_weights)
                        if total_weight > 0:
                            weighted_anterior = np.sum(group['anterior'].values * fjx_weights) / total_weight
                            weighted_flexion = np.sum(group['flexion'].values * fjx_weights) / total_weight
                            weighted_fjx = np.sum(group['fjx'].values * fjx_weights) / total_weight
                            
                            weighted_groups.append({
                                'fjx_bin': fjx_bin_idx,
                                'flexion_bin': flexion_bin_idx,
                                'anterior': weighted_anterior,
                                'flexion': weighted_flexion,
                                'fjx': weighted_fjx,
                                'n_points': len(group),
                                'effective_n': total_weight
                            })
            elif self.diagram_mode == "medial":
                for (fjy_bin_idx, flexion_bin_idx), group in data_df.groupby(['fjy_bin', 'flexion_bin']):
                        if len(group) < 1:
                            continue
                        
                        # Calculate weights based on distance from torque bin center
                        fjy_weights = self.calculate_bin_weights(
                            group['fjy'].values, 
                            group['fjy_bin_center'].values,
                            WEIGHT_TYPE, 
                            SIGMA_FACTOR,
                            bin_size
                        )
                        
                        # Calculate weighted averages
                        total_weight = np.sum(fjy_weights)
                        if total_weight > 0:
                            weighted_medial = np.sum(group['medial'].values * fjy_weights) / total_weight
                            weighted_flexion = np.sum(group['flexion'].values * fjy_weights) / total_weight
                            weighted_fjy = np.sum(group['fjy'].values * fjy_weights) / total_weight
                            
                            weighted_groups.append({
                                'fjy_bin': fjy_bin_idx,
                                'flexion_bin': flexion_bin_idx,
                                'medial': weighted_medial,
                                'flexion': weighted_flexion,
                                'fjy': weighted_fjy,
                                'n_points': len(group),
                                'effective_n': total_weight
                            })
            
            # Convert to DataFrame
            grouped_data = pd.DataFrame(weighted_groups)
            
            # Clear the existing plot
            self.canvas_contour_plot.ax.clear()
            
            # Create colormap
            n_tjx_bins = len(tjx_bins) - 1
            n_tjy_bins = len(tjy_bins) - 1
            n_fjx_bins = len(fjx_bins) - 1
            n_fjy_bins = len(fjy_bins) - 1
            if self.diagram_mode == 'rotation':
                colors = plt.cm.viridis(np.linspace(0, 1, n_tjx_bins))
                
            elif self.diagram_mode == 'varus_valgus' or self.diagram_mode == "adduction":
                colors = plt.cm.viridis(np.linspace(0, 1, n_tjy_bins))

            elif self.diagram_mode == 'anterior':
                colors = plt.cm.viridis(np.linspace(0, 1, n_fjx_bins))

            elif self.diagram_mode == 'medial':
                colors = plt.cm.viridis(np.linspace(0, 1, n_fjy_bins))


            
            # Plot the contour lines
            plotted_lines = 0
            if self.diagram_mode == 'rotation':
                for tjx_bin_idx in range(n_tjx_bins):

                    # Calculate the torque middle value for this bin
                    tjx_range_start = tjx_bins[tjx_bin_idx]
                    tjx_range_end = tjx_bins[tjx_bin_idx + 1]
                    tjx_middle = (tjx_range_start + tjx_range_end) / 2
                    
                    # Skip bins where torque middle is between -0.1 and 0.1
                    if -0.35 < tjx_middle < 0.2:
                        continue
                    
                    # Get data for this tjx bin
                    bin_data = grouped_data[grouped_data['tjx_bin'] == tjx_bin_idx].copy()
                    
                    if len(bin_data) < 10:
                        continue

                    
                    # Sort by flexion for smooth line connection
                    bin_data = bin_data.sort_values('flexion')
                    
                    # Apply moving average if enabled
                    if APPLY_MOVING_AVERAGE and len(bin_data) >= 3:
                        bin_data = self.apply_moving_average(bin_data, MOVING_AVERAGE_WINDOW, MOVING_AVERAGE_METHOD)
                    
                    # Plot the data
                    self.plot_contour_subset(bin_data, tjx_bin_idx, colors, tjx_bins, 
                                        INTERPOLATION_KIND, SMOOTHING_FACTOR, MIN_POINTS_FOR_SMOOTHING)
                    plotted_lines += 1
            
                # Configure the plot
                x_range = max(abs(rotation.min()), abs(rotation.max())) * 1.1
                self.canvas_contour_plot.ax.set_xlim(-x_range, x_range)
                self.canvas_contour_plot.ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
                
                # Set labels and title
                self.canvas_contour_plot.ax.set_xlabel('Internal Rotation [°]      External Rotation [°]', fontsize=12)
                self.canvas_contour_plot.ax.set_ylabel('Flexion [°]', fontsize=12)
                self.canvas_contour_plot.ax.invert_yaxis()
                title = 'Rotation Torque Contour Lines'
                """if APPLY_MOVING_AVERAGE:
                    title += f' + {MOVING_AVERAGE_METHOD.title()} Moving Average)'
                else:
                    title += ')'"""
                self.canvas_contour_plot.ax.set_title(title, fontsize=12)
                
                # Add grid and legend
                self.canvas_contour_plot.ax.grid(True, alpha=0.3)
                
                # Refresh the canvas
                self.canvas_contour_plot.draw()
                
                print(f"Contour plot generated successfully with {plotted_lines} lines")
            elif self.diagram_mode == 'varus_valgus':
                
                for tjy_bin_idx in range(n_tjy_bins):
                    
                    # Get data for this tjx bin
                    bin_data = grouped_data[grouped_data['tjy_bin'] == tjy_bin_idx].copy()
                    
                    if len(bin_data) < 10:
                        continue

                    
                    # Sort by flexion for smooth line connection
                    bin_data = bin_data.sort_values('flexion')
                    
                    # Apply moving average if enabled
                    if APPLY_MOVING_AVERAGE and len(bin_data) >= 3:
                        bin_data = self.apply_moving_average(bin_data, MOVING_AVERAGE_WINDOW, MOVING_AVERAGE_METHOD)
                    
                    # Plot the data
                    self.plot_contour_subset(bin_data, tjy_bin_idx, colors, tjy_bins, 
                                        INTERPOLATION_KIND, SMOOTHING_FACTOR, MIN_POINTS_FOR_SMOOTHING)
                    plotted_lines += 1
            
                # Configure the plot
                x_range = max(abs(medial_joint_gap.max()), abs(lateral_joint_gap.max())) * 1.1
                self.canvas_contour_plot.ax.set_xlim(-x_range, x_range)
                self.canvas_contour_plot.ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
                
                # Set labels and title
                self.canvas_contour_plot.ax.set_xlabel('Medial Joint Gap [mm]     Lateral Joint Gap [mm]', fontsize=12)
                self.canvas_contour_plot.ax.set_ylabel('Flexion [°]', fontsize=12)
                self.canvas_contour_plot.ax.invert_yaxis()
                title = 'Joint Gap Torque Contour Lines'
              
                self.canvas_contour_plot.ax.set_title(title, fontsize=12)
                
                # Add grid and legend
                self.canvas_contour_plot.ax.grid(True, alpha=0.3)
                
                # Refresh the canvas
                self.canvas_contour_plot.draw()
                
                print(f"Contour plot generated successfully with {plotted_lines} lines")
            elif self.diagram_mode == 'adduction':
                for tjy_bin_idx in range(n_tjy_bins):

                    # Calculate the torque middle value for this bin
                    tjy_range_start = tjy_bins[tjy_bin_idx]
                    tjy_range_end = tjy_bins[tjy_bin_idx + 1]
                    tjy_middle = (tjy_range_start + tjy_range_end) / 2
                    
                    # Skip bins where torque middle is between -0.1 and 0.1
                    if -0.35 < tjy_middle < 0.2:
                        continue
                    
                    # Get data for this tjx bin
                    bin_data = grouped_data[grouped_data['tjy_bin'] == tjy_bin_idx].copy()
                    
                    if len(bin_data) < 10:
                        continue

                    
                    # Sort by flexion for smooth line connection
                    bin_data = bin_data.sort_values('flexion')
                    
                    # Apply moving average if enabled
                    if APPLY_MOVING_AVERAGE and len(bin_data) >= 3:
                        bin_data = self.apply_moving_average(bin_data, MOVING_AVERAGE_WINDOW, MOVING_AVERAGE_METHOD)
                    
                    # Plot the data
                    self.plot_contour_subset(bin_data, tjy_bin_idx, colors, tjy_bins, 
                                        INTERPOLATION_KIND, SMOOTHING_FACTOR, MIN_POINTS_FOR_SMOOTHING)
                    plotted_lines += 1
            
                # Configure the plot
                x_range = max(abs(adduction.min()), abs(adduction.max())) * 1.1
                self.canvas_contour_plot.ax.set_xlim(-x_range, x_range)
                self.canvas_contour_plot.ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
                
                # Set labels and title
                self.canvas_contour_plot.ax.set_xlabel('varus angle [°]      valgus angle [°]', fontsize=12)
                self.canvas_contour_plot.ax.set_ylabel('Flexion [°]', fontsize=12)
                self.canvas_contour_plot.ax.invert_yaxis()
                title = 'Rotation Torque Contour Lines'
                """if APPLY_MOVING_AVERAGE:
                    title += f' + {MOVING_AVERAGE_METHOD.title()} Moving Average)'
                else:
                    title += ')'"""
                self.canvas_contour_plot.ax.set_title(title, fontsize=12)
                
                # Add grid and legend
                self.canvas_contour_plot.ax.grid(True, alpha=0.3)
                
                # Refresh the canvas
                self.canvas_contour_plot.draw()
                
                print(f"Contour plot generated successfully with {plotted_lines} lines")
            elif self.diagram_mode == 'anterior':
                for fjx_bin_idx in range(n_fjx_bins):

                    # Calculate the torque middle value for this bin
                    fjx_range_start = fjx_bins[fjx_bin_idx]
                    fjx_range_end = fjx_bins[fjx_bin_idx + 1]
                    fjx_middle = (fjx_range_start + fjx_range_end) / 2
                    
                    # Skip bins where torque middle is between -0.1 and 0.1
                    if -0.35 < fjx_middle < 0.2:
                        continue
                    
                    # Get data for this tjx bin
                    bin_data = grouped_data[grouped_data['fjx_bin'] == fjx_bin_idx].copy()
                    
                    if len(bin_data) < 10:
                        continue

                    
                    # Sort by flexion for smooth line connection
                    bin_data = bin_data.sort_values('flexion')
                    
                    # Apply moving average if enabled
                    if APPLY_MOVING_AVERAGE and len(bin_data) >= 3:
                        bin_data = self.apply_moving_average(bin_data, MOVING_AVERAGE_WINDOW, MOVING_AVERAGE_METHOD)
                    
                    # Plot the data
                    self.plot_contour_subset(bin_data, fjx_bin_idx, colors, fjx_bins, 
                                        INTERPOLATION_KIND, SMOOTHING_FACTOR, MIN_POINTS_FOR_SMOOTHING)
                    plotted_lines += 1
            
                # Configure the plot
                x_range = max(abs(anterior.min()), abs(anterior.max())) * 1.1
                self.canvas_contour_plot.ax.set_xlim(-x_range, x_range)
                self.canvas_contour_plot.ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
                
                # Set labels and title
                self.canvas_contour_plot.ax.set_xlabel('posterior translation [mm]      anterior translation [mm]', fontsize=12)
                self.canvas_contour_plot.ax.set_ylabel('Flexion [°]', fontsize=12)
                self.canvas_contour_plot.ax.invert_yaxis()
                title = 'Rotation Torque Contour Lines'
                """if APPLY_MOVING_AVERAGE:
                    title += f' + {MOVING_AVERAGE_METHOD.title()} Moving Average)'
                else:
                    title += ')'"""
                self.canvas_contour_plot.ax.set_title(title, fontsize=12)
                
                # Add grid and legend
                self.canvas_contour_plot.ax.grid(True, alpha=0.3)
                
                # Refresh the canvas
                self.canvas_contour_plot.draw()
                
                print(f"Contour plot generated successfully with {plotted_lines} lines")
            elif self.diagram_mode == 'medial':
                for fjy_bin_idx in range(n_fjy_bins):

                    # Calculate the torque middle value for this bin
                    fjy_range_start = fjy_bins[fjy_bin_idx]
                    fjy_range_end = fjy_bins[fjy_bin_idx + 1]
                    fjy_middle = (fjy_range_start + fjy_range_end) / 2
                    
                    # Skip bins where torque middle is between -0.1 and 0.1
                    if -0.35 < fjy_middle < 0.2:
                        continue
                    
                    # Get data for this tjx bin
                    bin_data = grouped_data[grouped_data['fjy_bin'] == fjy_bin_idx].copy()
                    
                    if len(bin_data) < 10:
                        continue

                    
                    # Sort by flexion for smooth line connection
                    bin_data = bin_data.sort_values('flexion')
                    
                    # Apply moving average if enabled
                    if APPLY_MOVING_AVERAGE and len(bin_data) >= 3:
                        bin_data = self.apply_moving_average(bin_data, MOVING_AVERAGE_WINDOW, MOVING_AVERAGE_METHOD)
                    
                    # Plot the data
                    self.plot_contour_subset(bin_data, fjy_bin_idx, colors, fjy_bins, 
                                        INTERPOLATION_KIND, SMOOTHING_FACTOR, MIN_POINTS_FOR_SMOOTHING)
                    plotted_lines += 1
            
                # Configure the plot
                x_range = max(abs(medial.min()), abs(medial.max())) * 1.1
                self.canvas_contour_plot.ax.set_xlim(-x_range, x_range)
                self.canvas_contour_plot.ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
                
                # Set labels and title
                self.canvas_contour_plot.ax.set_xlabel('lateral translation [mm]      medial translation [mm]', fontsize=12)
                self.canvas_contour_plot.ax.set_ylabel('Flexion [°]', fontsize=12)
                title = 'Rotation Torque Contour Lines'
                """if APPLY_MOVING_AVERAGE:
                    title += f' + {MOVING_AVERAGE_METHOD.title()} Moving Average)'
                else:
                    title += ')'"""
                self.canvas_contour_plot.ax.set_title(title, fontsize=12)
                
                # Add grid and legend
                self.canvas_contour_plot.ax.grid(True, alpha=0.3)
                
                # Refresh the canvas
                self.canvas_contour_plot.draw()
                
                print(f"Contour plot generated successfully with {plotted_lines} lines")     
        
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            # You might want to show a QMessageBox here to inform the user
        except Exception as e:
            print(f"Error generating contour plot: {e}")
            # You might want to show a QMessageBox here to inform the user

    def calculate_bin_weights(self, values, bin_centers, weight_type='gaussian', sigma_factor=0.3, bin_size=0.4):
        """
        Calculate weights for values within bins, giving higher weight to values 
        closer to bin centers.
        """
        weights = np.ones_like(values)
        
        for i, (val, center) in enumerate(zip(values, bin_centers)):
            distance = abs(val - center)
            
            if weight_type == 'gaussian':
                sigma = bin_size * sigma_factor
                weights[i] = np.exp(-(distance**2) / (2 * sigma**2))
                
            elif weight_type == 'triangular':
                max_distance = bin_size / 2
                weights[i] = max(0, 1 - distance / max_distance)
                
            elif weight_type == 'quadratic':
                max_distance = bin_size / 2
                if distance <= max_distance:
                    weights[i] = 1 - (distance / max_distance)**2
                else:
                    weights[i] = 0
        
        return weights

    def apply_moving_average(self, data, window_size=3, method='simple'):
        """
        Apply moving average to smooth the data.
        """
        from scipy import interpolate
        
        if len(data) < window_size:
            return data.copy()
        
        # Ensure window size is odd for centered window
        if window_size % 2 == 0:
            window_size += 1
        
        smoothed_data = data.copy()
        
        if method == 'simple':
            if self.diagram_mode == 'rotation':
                smoothed_data['rotation'] = data['rotation'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                smoothed_data['flexion'] = data['flexion'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                smoothed_data['tjx'] = data['tjx'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
            elif self.diagram_mode == 'varus_valgus':
                smoothed_data['medial_joint_gap'] = data['medial_joint_gap'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                smoothed_data['lateral_joint_gap'] = data['lateral_joint_gap'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                smoothed_data['flexion'] = data['flexion'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                smoothed_data['tjy'] = data['tjy'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
            elif self.diagram_mode == 'adduction':
                smoothed_data['adduction'] = data['adduction'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                smoothed_data['flexion'] = data['flexion'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                smoothed_data['tjy'] = data['tjy'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
            elif self.diagram_mode == 'anterior':
                smoothed_data['anterior'] = data['anterior'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                smoothed_data['flexion'] = data['flexion'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                smoothed_data['fjx'] = data['fjx'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
            elif self.diagram_mode == 'medial':
                smoothed_data['medial'] = data['medial'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                smoothed_data['flexion'] = data['flexion'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
                smoothed_data['fjy'] = data['fjy'].rolling(
                    window=window_size, center=True, min_periods=1
                ).mean()
            
        elif method == 'weighted':
            def weighted_average(series, window_size):
                weights = np.bartlett(window_size)
                weights = weights / weights.sum()
                
                result = series.copy()
                half_window = window_size // 2
                
                for i in range(len(series)):
                    start_idx = max(0, i - half_window)
                    end_idx = min(len(series), i + half_window + 1)
                    window_data = series.iloc[start_idx:end_idx]
                    
                    if len(window_data) < window_size:
                        if i < half_window:
                            used_weights = weights[-(len(window_data)):]
                        else:
                            used_weights = weights[:len(window_data)]
                        used_weights = used_weights / used_weights.sum()
                    else:
                        used_weights = weights
                    
                    result.iloc[i] = np.average(window_data, weights=used_weights)
                
                return result
            if self.diagram_mode == 'rotation':
                smoothed_data['rotation'] = weighted_average(data['rotation'], window_size)
                smoothed_data['flexion'] = weighted_average(data['flexion'], window_size)
                smoothed_data['tjx'] = weighted_average(data['tjx'], window_size)
            elif self.diagram_mode == 'varus_valgus':
                smoothed_data['medial_joint_gap'] = weighted_average(data['medial_joint_gap'], window_size)
                smoothed_data['lateral_joint_gap'] = weighted_average(data['lateral_joint_gap'], window_size)
                smoothed_data['flexion'] = weighted_average(data['flexion'], window_size)
                smoothed_data['tjy'] = weighted_average(data['tjy'], window_size)
            elif self.diagram_mode == 'adduction':
                smoothed_data['adduction'] = weighted_average(data['adduction'], window_size)
                smoothed_data['flexion'] = weighted_average(data['flexion'], window_size)
                smoothed_data['tjy'] = weighted_average(data['tjy'], window_size)
            elif self.diagram_mode == 'anterior':
                smoothed_data['anterior'] = weighted_average(data['anterior'], window_size)
                smoothed_data['flexion'] = weighted_average(data['flexion'], window_size)
                smoothed_data['fjx'] = weighted_average(data['fjx'], window_size)
            elif self.diagram_mode == 'medial':
                smoothed_data['medial'] = weighted_average(data['medial'], window_size)
                smoothed_data['flexion'] = weighted_average(data['flexion'], window_size)
                smoothed_data['fjy'] = weighted_average(data['fjy'], window_size)
            
        elif method == 'exponential':
            if self.diagram_mode == 'rotation':
                alpha = 2.0 / (window_size + 1)
                smoothed_data['rotation'] = data['rotation'].ewm(alpha=alpha, adjust=False).mean()
                smoothed_data['flexion'] = data['flexion'].ewm(alpha=alpha, adjust=False).mean()
                smoothed_data['tjx'] = data['tjx'].ewm(alpha=alpha, adjust=False).mean()
            elif self.diagram_mode == 'varus_valgus':
                alpha = 2.0 / (window_size + 1)
                smoothed_data['medial_joint_gap'] = data['medial_joint_gap'].ewm(alpha=alpha, adjust=False).mean()
                smoothed_data['lateral_joint_gap'] = data['lateral_joint_gap'].ewm(alpha=alpha, adjust=False).mean()
                smoothed_data['flexion'] = data['flexion'].ewm(alpha=alpha, adjust=False).mean()
                smoothed_data['tjy'] = data['tjy'].ewm(alpha=alpha, adjust=False).mean()
            elif self.diagram_mode == 'adduction':
                alpha = 2.0 / (window_size + 1)
                smoothed_data['adduction'] = data['aduction'].ewm(alpha=alpha, adjust=False).mean()
                smoothed_data['flexion'] = data['flexion'].ewm(alpha=alpha, adjust=False).mean()
                smoothed_data['tjy'] = data['tjy'].ewm(alpha=alpha, adjust=False).mean()
            elif self.diagram_mode == 'anterior':
                alpha = 2.0 / (window_size + 1)
                smoothed_data['anterior'] = data['anterior'].ewm(alpha=alpha, adjust=False).mean()
                smoothed_data['flexion'] = data['flexion'].ewm(alpha=alpha, adjust=False).mean()
                smoothed_data['fjx'] = data['fjx'].ewm(alpha=alpha, adjust=False).mean()
            elif self.diagram_mode == 'medial':
                alpha = 2.0 / (window_size + 1)
                smoothed_data['medial'] = data['medial'].ewm(alpha=alpha, adjust=False).mean()
                smoothed_data['flexion'] = data['flexion'].ewm(alpha=alpha, adjust=False).mean()
                smoothed_data['fjy'] = data['fjy'].ewm(alpha=alpha, adjust=False).mean()
        
        return smoothed_data
    
    def save_current_plot(self):
        """Automatically save plot with timestamp"""
        try:
            save_dir = "saved_plots"
            os.makedirs(save_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"contour_plot_{timestamp}.png")
            
            self.canvas_contour_plot.figure.savefig(
                filename,
                dpi=300,
                bbox_inches='tight',
                facecolor='white'
            )
            
            print(f"Plot auto-saved as {filename}")
            
        except Exception as e:
            print(f"Auto-save failed: {str(e)}")

    def plot_contour_subset(self, bin_data, tjx_bin_idx, colors, tjx_bins, 
                        interpolation_kind, smoothing_factor, min_points_for_smoothing):
        """
        Helper method to plot a subset of contour data.
        """
        from scipy import interpolate
        
        if self.diagram_mode == 'rotation':
            # Separate positive, negative, and zero rotation values
            positive_data = bin_data[bin_data['rotation'] > 0].copy()
            negative_data = bin_data[bin_data['rotation'] < 0].copy()
            zero_data = bin_data[bin_data['rotation'] == 0].copy()
        elif self.diagram_mode == 'varus_valgus':
            positive_data  = bin_data[bin_data['medial_joint_gap'] > 0].copy()
            negative_data = bin_data[bin_data['lateral_joint_gap'] > 0].copy()
            zero_data = pd.DataFrame() 
        elif self.diagram_mode == 'adduction':
            # Separate positive, negative, and zero rotation values
            positive_data = bin_data[bin_data['adduction'] > 0].copy()
            negative_data = bin_data[bin_data['adduction'] < 0].copy()
            zero_data = bin_data[bin_data['adduction'] == 0].copy()  
        elif self.diagram_mode == 'anterior':
            # Separate positive, negative, and zero rotation values
            positive_data = bin_data[bin_data['anterior'] > 0].copy()
            negative_data = bin_data[bin_data['anterior'] < 0].copy()
            zero_data = bin_data[bin_data['anterior'] == 0].copy()  
        elif self.diagram_mode == 'medial':
            # Separate positive, negative, and zero rotation values
            positive_data = bin_data[bin_data['medial'] > 0].copy()
            negative_data = bin_data[bin_data['medial'] < 0].copy()
            zero_data = bin_data[bin_data['medial'] == 0].copy()  
            
        
        def plot_subset(subset_data, test, marker_style='o'):
            if len(subset_data) < 1:
                return 0
            
            subset_data = subset_data.sort_values('flexion')
            if self.diagram_mode == 'rotation':
                x_values = subset_data['rotation'].values
            elif self.diagram_mode == 'varus_valgus':
                if test == 0:
                    x_values = -(subset_data['medial_joint_gap'].values)
                else:
                    x_values = subset_data['lateral_joint_gap'].values
            elif self.diagram_mode == 'adduction':
                x_values = subset_data['adduction'].values
            elif self.diagram_mode == 'anterior':
                x_values = subset_data['anterior'].values
            elif self.diagram_mode == 'medial':
                x_values = subset_data['medial'].values

            y_values = subset_data['flexion'].values
            
            # Plot line if we have enough points
            if len(subset_data) >= 2:
                if len(subset_data) >= min_points_for_smoothing:
                    try:
                        if smoothing_factor > 1:
                            y_smooth = np.linspace(y_values.min(), y_values.max(), 
                                                len(y_values) * smoothing_factor)
                        else:
                            y_smooth = y_values
                        
                        f = interpolate.interp1d(y_values, x_values, kind=interpolation_kind, 
                                            bounds_error=False, fill_value='extrapolate')
                        x_smooth = f(y_smooth)
                        
                        self.canvas_contour_plot.ax.plot(x_smooth, y_smooth, 
                                                        color=colors[tjx_bin_idx], 
                                                        linewidth=2, alpha=0.8)
                        
                    except Exception as e:
                        print(f"Smoothing failed for bin {tjx_bin_idx}: {e}")
                        self.canvas_contour_plot.ax.plot(x_values, y_values, 
                                                        color=colors[tjx_bin_idx], 
                                                        linewidth=2, alpha=0.8)
                else:
                    self.canvas_contour_plot.ax.plot(x_values, y_values, 
                                                    color=colors[tjx_bin_idx], 
                                                    linewidth=2, alpha=0.8)
            
            # Plot the actual averaged points
            marker_size = 2 if marker_style == 's' else 8
            edge_color = 'black' if marker_style == 's' else 'white'
            edge_width = 0.3 if marker_style == 's' else 0.1
            
            self.canvas_contour_plot.ax.scatter(x_values, y_values, 
                                                c=[colors[tjx_bin_idx]], 
                                                alpha=0.9, s=marker_size, marker=marker_style,
                                                edgecolors=edge_color, linewidth=edge_width, zorder=5)
            
            return len(subset_data)
        
        # Plot each subset
        pos_count = plot_subset(positive_data, 1, 'o')
        neg_count = plot_subset(negative_data, 0, 'o') 
        
        zero_count = plot_subset(zero_data, 's')
        
        total_count = pos_count + neg_count + zero_count
        
        if total_count > 0:
            # Create legend entry
            tjx_range_start = tjx_bins[tjx_bin_idx]
            tjx_range_end = tjx_bins[tjx_bin_idx + 1]
            tjx_middle = (tjx_range_start + tjx_range_end) / 2
            
            if self.diagram_mode == 'varus_valgus' or self.diagram_mode == 'adduction' or self.diagram_mode == 'rotation':
                legend_label = f'{tjx_middle:.2f} Nm ({total_count})'
            else:
                legend_label = f'{tjx_middle:.2f} N ({total_count})'
            
            # Add a dummy scatter for legend
            self.canvas_contour_plot.ax.scatter([], [], 
                                                c=[colors[tjx_bin_idx]], 
                                                label=legend_label, s=40)
            # After all your scatter plots and legend entries
            self.canvas_contour_plot.ax.legend(
                loc='best',
                fontsize=6,
                #frameon=True,  # Show frame around legend
                #fancybox=True,  # Rounded corners
                #shadow=True     # Drop shadow
            )
            
    



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