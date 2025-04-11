import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFrame, 
                            QProgressBar, QGridLayout, QSplitter)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
import matplotlib.cm as cm

class MplCanvas(FigureCanvas):
    """Matplotlib canvas class for embedding plots in Qt"""
    def __init__(self, width=5, height=4):
        self.fig = Figure(figsize=(width, height))
        self.axes_force = self.fig.add_subplot(121, projection='3d')
        self.axes_torque = self.fig.add_subplot(122, projection='3d')
        super(MplCanvas, self).__init__(self.fig)
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
        self.timer = QTimer()
        self.timer.timeout.connect(self.rotation_complete)
        self.seconds_timer = QTimer()
        self.seconds_timer.timeout.connect(self.update_seconds_progress)
        
        # Timer for visualization updates (every 0.2 seconds)
        self.viz_timer = QTimer()
        self.viz_timer.timeout.connect(self.update_visualization_timer)
        self.viz_timer.setInterval(200)  # 0.2 seconds
        
        # History for visualization
        self.history_size = 10  # Number of previous arrows to show
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
        filename = "print_data.F_sensor_temp_data_5.txt"
        self.forces = []
        self.torques = []
        
        try:
            with open(filename, 'r') as file:
                for line in file:
                    values = [float(val) for val in line.strip().split(',')]
                    if len(values) >= 6:  # Ensure we have at least 6 values (3 forces + 3 torques)
                        self.forces.append(values[0:3])
                        self.torques.append(values[3:6])
            
            self.forces = np.array(self.forces)
            self.torques = np.array(self.torques)
            print(f"Successfully loaded {len(self.forces)} force/torque data points.")
            
            # Divide the data into segments for each test phase
            self.data_segments = {}
            segment_size = len(self.forces) // (len(self.flexion_angles) * 5)  # 5 tests per angle
            
            idx = 0
            for angle in self.flexion_angles:
                for test_type in ['flexion', 'varus', 'valgus', 'internal_rot', 'external_rot']:
                    end_idx = min(idx + segment_size, len(self.forces))
                    self.data_segments[(angle, test_type)] = (idx, end_idx)
                    idx = end_idx
            
        except FileNotFoundError:
            print(f"Error: File 'print_data.F_sensor_temp_data_5.txt' not found.")
            # Create dummy data if file not found
            self.forces = np.random.rand(100, 3) * 10
            self.torques = np.random.rand(100, 3) * 2
            print("Using random dummy data instead.")
        except Exception as e:
            print(f"An error occurred: {e}")
            # Create dummy data if error
            self.forces = np.random.rand(100, 3) * 10
            self.torques = np.random.rand(100, 3) * 2
            print("Using random dummy data instead.")
    
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
        
        # Left part: Image display
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        self.image_frame = QFrame()
        self.image_frame.setLineWidth(2)
        self.image_frame.setMinimumSize(300, 250)
        
        image_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label)
        self.image_frame.setLayout(image_layout)
        
        #left_layout.addWidget(self.image_frame)
        
        # Add force/torque visualization below the image
        viz_label = QLabel("Force & Torque Visualization")
        viz_label.setAlignment(Qt.AlignCenter)
        viz_label.setFont(QFont("Arial", 12, QFont.Bold))
        #left_layout.addWidget(viz_label)
        
        # Create matplotlib visualization
        self.canvas = MplCanvas(width=3, height=4)
        left_layout.addWidget(self.canvas)
        
        left_widget.setLayout(left_layout)
        bottom_splitter.addWidget(left_widget)
        
        # Right part: Control buttons
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

        record_data_label = QLabel("Record Data")
        record_data_label.setAlignment(Qt.AlignCenter)
        
        # Layout arrangement
        subsub_layout = QHBoxLayout()
        subsub_layout.addWidget(self.start_button)
        subsub_layout.addWidget(self.next_button)

        right_layout.addLayout(subsub_layout, 0, 0)
        right_layout.addWidget(self.next_label, 0, 1)
        right_layout.addWidget(self.image_frame, 2, 1, 5, 1)
        right_layout.addWidget(record_data_label, 1,0,2, 1)
        right_layout.addWidget(self.rotate_button, 2, 0)
        right_layout.addWidget(self.varus_button, 3, 0)
        right_layout.addWidget(self.valgus_button, 4,0)
        right_layout.addWidget(self.internal_rot_button, 5, 0)
        right_layout.addWidget(self.external_rot_button, 6, 0)
        
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
        
    def update_visualization_timer(self):
        """Called by timer to update visualization every 0.2 seconds"""
        if self.experiment_running:
            self.current_data_index += 1
            self.update_visualization(self.current_data_index)
        
    def update_visualization(self, data_index=0, test_type=None):
        """Update the force/torque visualization with the current data."""
        if test_type is None:
            test_type = self.current_test_type
            
        if test_type == 'none':
            test_type = 'flexion'  # Default test type
            
        # Get the data range for current angle and test type
        try:
            if (self.current_angle_index < len(self.flexion_angles) and 
                (self.flexion_angles[self.current_angle_index], test_type) in self.data_segments):
                start_idx, end_idx = self.data_segments[(self.flexion_angles[self.current_angle_index], test_type)]
                # Select data within the range and with a specific index offset
                data_range = end_idx - start_idx
                # Make sure we stay within range, and cycle through data if needed
                idx = start_idx + (data_index % data_range)
                force = self.forces[idx]
                torque = self.torques[idx]
            else:
                # Use random sample if no matching segment
                idx = data_index % len(self.forces)
                force = self.forces[idx]
                torque = self.torques[idx]
        except Exception as e:
            print(f"Error updating visualization: {e}")
            # Fallback to random data
            force = np.random.rand(3) * 10
            torque = np.random.rand(3) * 2
        
        # Add to history
        self.force_history.append(force)
        self.torque_history.append(torque)
        
        # Keep history to specified size
        #if len(self.force_history) > self.history_size:
        #    self.force_history.pop(0)
        #   self.torque_history.pop(0)
        
        # Clear previous plots
        self.canvas.axes_force.clear()
        self.canvas.axes_torque.clear()
        
        # Set up force plot limits and labels
        #force_max = np.max(np.abs(self.forces)) * 1.2
        force_max = 7
        self.canvas.axes_force.set_xlim([-force_max, force_max])
        self.canvas.axes_force.set_ylim([-force_max, force_max])
        self.canvas.axes_force.set_zlim([-force_max, force_max])

        self.canvas.axes_force.set_xlim([-15, 15])
        self.canvas.axes_force.set_ylim([-15, 15])
        self.canvas.axes_force.set_zlim([-15, 15])

        self.canvas.axes_force.set_title('Force Vectors (N)')
        self.canvas.axes_force.set_xlabel('X')
        self.canvas.axes_force.set_ylabel('Y')
        self.canvas.axes_force.set_zlabel('Z')
        
        # Set up torque plot limits and labels
        #torque_max = np.max(np.abs(self.torques)) * 1.2
        torque_max = 2

        self.canvas.axes_torque.set_xlim([-torque_max, torque_max])
        self.canvas.axes_torque.set_ylim([-torque_max, torque_max])
        self.canvas.axes_torque.set_zlim([-torque_max, torque_max])
        self.canvas.axes_torque.set_title('Torque Vectors (Nm)')
        self.canvas.axes_torque.set_xlabel('X')
        self.canvas.axes_torque.set_ylabel('Y')
        self.canvas.axes_torque.set_zlabel('Z')
        
        # Calculate max magnitudes for scaling
        max_force_mag = np.max(np.sqrt(np.sum(self.forces**2, axis=1))) if len(self.forces) > 0 else 1
        max_torque_mag = np.max(np.sqrt(np.sum(self.torques**2, axis=1))) if len(self.torques) > 0 else 1
        
        # Plot history with color gradient (older = more transparent)
        cmap_force = cm.viridis
        cmap_torque = cm.plasma
        
        for i, (hist_force, hist_torque) in enumerate(zip(self.force_history, self.torque_history)):
            # Calculate color and alpha based on position in history
            # Newer arrows are more opaque
            alpha = 0.3 + 0.7 * (i / max(1, len(self.force_history) - 1))
            color_idx = i / max(1, len(self.force_history) - 1)
            
            # Force arrow
            force_mag = np.sqrt(np.sum(hist_force**2))
            width_scale = 0.5 + 2.5 * (force_mag / max_force_mag) if max_force_mag > 0 else 0.5
            
            color_force = cmap_force(color_idx)
            color_force = (*color_force[:3], alpha)  # Set alpha for the color
            
            self.canvas.axes_force.quiver(0, 0, 0, 
                    hist_force[0], hist_force[1], hist_force[2],
                    color=color_force, 
                    linewidth=width_scale,
                    arrow_length_ratio=0.1)
            
            # Torque arrow
            torque_mag = np.sqrt(np.sum(hist_torque**2))
            width_scale = 0.5 + 2.5 * (torque_mag / max_torque_mag) if max_torque_mag > 0 else 0.5
            
            color_torque = cmap_torque(color_idx)
            color_torque = (*color_torque[:3], alpha)  # Set alpha for the color
            
            self.canvas.axes_torque.quiver(0, 0, 0, 
                    hist_torque[0], hist_torque[1], hist_torque[2],
                    color=color_torque, 
                    linewidth=width_scale,
                    arrow_length_ratio=0.1)
        
        # Display magnitudes of the current force/torque
        current_force = self.force_history[-1] if self.force_history else force
        current_torque = self.torque_history[-1] if self.torque_history else torque
        
        force_mag = np.sqrt(np.sum(current_force**2))
        torque_mag = np.sqrt(np.sum(current_torque**2))
        
        self.canvas.axes_force.text2D(0.05, 0.95, f"Force Mag: {force_mag:.2f}N", transform=self.canvas.axes_force.transAxes)
        self.canvas.axes_torque.text2D(0.05, 0.95, f"Torque Mag: {torque_mag:.2f}Nm", transform=self.canvas.axes_torque.transAxes)
        
        # Add legend or note about colors
        #self.canvas.axes_force.text2D(0.05, 0.90, "Newest → Oldest", transform=self.canvas.axes_force.transAxes)
        #self.canvas.axes_torque.text2D(0.05, 0.90, "Newest → Oldest", transform=self.canvas.axes_torque.transAxes)
        
        # Update the figure
        self.canvas.fig.tight_layout()
        self.canvas.draw()
    
    def update_display(self):
        if self.current_angle_index < len(self.flexion_angles):
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
            
            # Reset visualization history
            #self.force_history = []
            #self.torque_history = []
            
            # Don't reset current_data_index to keep visualization continuous
            # Only change test_type to match the new angle
            if self.experiment_running:
                self.current_test_type = 'flexion'
        else:
            self.instruction_label.setText("Experiment Complete!")
            self.overall_progress.setValue(self.current_angle_index)
            self.image_label.clear()
        
            # Disable all buttons
            self.start_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.rotate_button.setEnabled(False)
            self.varus_button.setEnabled(False)
            self.valgus_button.setEnabled(False)
            self.internal_rot_button.setEnabled(False)
            self.external_rot_button.setEnabled(False)

            print("debug test")
        
            # Stop visualization timer
            if self.viz_timer.isActive():
                self.viz_timer.stop()
            
            # Reset experiment running flag
            self.experiment_running = False
            
            # self.reset_buttons_and_labels()
            #self.current_angle_index += 1
    
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
    
    def start_experiment(self):
        self.current_angle_index = 0
        self.current_angle = self.flexion_angles[self.current_angle_index]
        self.overall_progress.setValue(0)
        
        self.force_history = []
        self.torque_history = []
        self.current_data_index = 0
        #self.update_display()
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
        
        # Reset visualization history but keep continuous data index
        #self.force_history = []
        #self.torque_history = []
        #self.current_data_index = 0
        
        # Enable only start button
        self.start_button.setEnabled(False)
        self.next_label.show()
        self.next_button.setEnabled(False)
        self.rotate_button.setEnabled(True)
        
        # Start visualization timer immediately and keep it running throughout the experiment
        if not self.viz_timer.isActive():
            self.viz_timer.start()
    
    def next_angle(self):
        #self.current_angle_index += 1
        self.update_display()
        
        # Reset button states
        self.next_button.setEnabled(False)
        self.rotate_button.setEnabled(True)
        self.varus_button.setEnabled(False)
        self.valgus_button.setEnabled(False)
        self.internal_rot_button.setEnabled(False)
        self.external_rot_button.setEnabled(False)
        
        # Set current test type to 'flexion' for the new angle
        self.current_test_type = 'flexion'

    def start_rotation(self):
        # Disable rotate button
        self.rotate_button.setEnabled(False)
        
        # Enable varus button
        self.varus_button.setEnabled(True)
        
        # Set current test type to flexion
        self.current_test_type = 'flexion'
        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.next_button.setEnabled(False)
        
    def start_varus(self):
        # Disable varus button
        self.varus_button.setEnabled(False)
        
        # Set current test type to varus
        self.current_test_type = 'varus'
        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.valgus_button.setEnabled(True)

    def start_valgus(self):
        # Disable valgus button
        self.valgus_button.setEnabled(False)
        
        # Set current test type to valgus
        self.current_test_type = 'valgus'
        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.internal_rot_button.setEnabled(True)

    def start_internal_rot(self):
        # Disable internal rotation button
        self.internal_rot_button.setEnabled(False)
        
        # Set current test type to internal_rot
        self.current_test_type = 'internal_rot'
        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.external_rot_button.setEnabled(True)

    def start_external_rot(self):
        # Disable external rotation button
        self.external_rot_button.setEnabled(False)
        
        # Set current test type to external_rot
        self.current_test_type = 'external_rot'
        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.next_button.setEnabled(True)
        self.current_angle_index += 1
        if self.current_angle_index == 5:
            self.next_button.setEnabled(False)

    
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
        print(self.current_angle_index)
        print(len(self.flexion_angles))
        
        if self.current_angle_index >= (len(self.flexion_angles)):
            # Experiment complete
            self.instruction_label.setText("Experiment Complete!")
            self.current_angle_index += 1
            self.rotation_progress_label.setVisible(False)
            self.next_label.setVisible(False)
            self.image_label.setVisible(False)
            self.reset_buttons_and_labels()
            self.overall_progress.setValue(5)
            #print("2. debug test")
            #self.update_display()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KneeFlexionExperiment()
    window.show()
    sys.exit(app.exec_())