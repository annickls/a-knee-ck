import sys
import numpy as np
import matplotlib.pyplot as plt


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFrame, 
                            QProgressBar, QGridLayout, QSplitter, QTabWidget)
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


class MplCurrentCanvas(FigureCanvas):
    """Matplotlib canvas class for displaying only current force/torque"""
    def __init__(self, width=5, height=4):
        self.fig = Figure(figsize=(width, height))
        self.axes_force = self.fig.add_subplot(121, projection='3d')
        self.axes_torque = self.fig.add_subplot(122, projection='3d')
        super(MplCurrentCanvas, self).__init__(self.fig)
        self.fig.tight_layout()



from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


"""# Try different import methods for VTK depending on the version
try:
    # For newer VTK versions
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    try:
        # For older VTK versions
        from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    except ImportError:
        print("Could not import QVTKRenderWindowInteractor. Make sure VTK is installed correctly.")
        # Fallback to use PyVista's built-in solution
        QVTKRenderWindowInteractor = None"""

# Try to import PyVista with explicit error messages
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
    print("PyVista imported successfully")
except ImportError as e:
    print(f"PyVista import failed: {e}")
    PYVISTA_AVAILABLE = False

class BoneVisualizationWidget(QWidget):
    """Widget for displaying 3D bone models with force/torque arrows"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout()
        
        if not PYVISTA_AVAILABLE or QVTKRenderWindowInteractor is None:
            # Fallback to a simple message if dependencies are not available
            message = QLabel("3D visualization requires PyVista and VTK libraries.\n"
                            "Please install them with:\n"
                            "pip install pyvista vtk")
            message.setAlignment(Qt.AlignCenter)
            layout.addWidget(message)
            self.setLayout(layout)
            return
            
        # Create VTK widget
        self.frame = QFrame()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        layout.addWidget(self.vtkWidget)
        
        # Initialize PyVista plotter
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.background_color = "white"
        
        # Set up VTK widget to use our plotter's renderer
        self.ren_win = self.vtkWidget.GetRenderWindow()
        self.ren_win.AddRenderer(self.plotter.renderer)
        self.iren = self.ren_win.GetInteractor()
        
        # Try to load bone models
        try:
            # First check if files exist
            import os
            tibia_file = 'tibia.stl'
            femur_file = 'femur.stl'
            
            if not os.path.exists(tibia_file) or not os.path.exists(femur_file):
                # Show message about missing files and create placeholder bones
                message = QLabel(f"STL files not found: {tibia_file}, {femur_file}\n"
                                "Using placeholder cylinders instead.")
                message.setAlignment(Qt.AlignCenter)
                layout.addWidget(message)
                
                # Create simple cylinder placeholders
                self.tibia = pv.Cylinder(center=(0, -5, 0), direction=(0, 1, 0), radius=2, height=10)
                self.femur = pv.Cylinder(center=(0, 5, 0), direction=(0, 1, 0), radius=2, height=10)
            else:
                # Load actual STL files
                self.tibia = pv.read(tibia_file)
                self.femur = pv.read(femur_file)
                
                # Center the meshes at the knee joint
                self.center_bones()
            
            # Add meshes to the plotter
            self.plotter.add_mesh(self.tibia, color='tan', opacity=0.7)
            self.plotter.add_mesh(self.femur, color='ivory', opacity=0.7)
            
            # Add coordinate axes
            self.plotter.add_axes()
            
            # Initialize force and torque arrows
            self.force_arrow = None
            self.torque_arrow = None
            
        except Exception as e:
            print(f"Error in bone visualization setup: {e}")
            # Create a label to display the error
            error_label = QLabel(f"Error setting up 3D visualization: {str(e)}")
            error_label.setWordWrap(True)
            layout.addWidget(error_label)
        
        self.setLayout(layout)
        
        # Initialize the interactor after everything is set up
        self.iren.Initialize()
        self.iren.Start()
    
    def center_bones(self):
        """Position the bones so the knee joint is at the center"""
        # Scale bones to appropriate size
        scale_factor = 0.01  # Adjust this as needed
        self.tibia.scale(scale_factor, inplace=True)
        self.femur.scale(scale_factor, inplace=True)
        
        # Position bones relative to joint center
        # These offsets will need to be adjusted based on your STL file coordinates
        self.tibia.translate([0, -5, 0], inplace=True)  # Move tibia down
        self.femur.translate([0, 5, 0], inplace=True)   # Move femur up
        
        # Rotate bones to the correct orientation
        # This will depend on the coordinate system of your STL files
        self.tibia.rotate_x(90, inplace=True)
        self.femur.rotate_x(90, inplace=True)
    
    def update_force_torque(self, force, torque):
        """Update the force and torque arrows"""
        if not PYVISTA_AVAILABLE:
            return
            
        try:
            # Remove previous arrows
            if hasattr(self, 'force_arrow') and self.force_arrow is not None:
                self.plotter.remove_actor(self.force_arrow)
            if hasattr(self, 'torque_arrow') and self.torque_arrow is not None:
                self.plotter.remove_actor(self.torque_arrow)
            
            # Create new arrows
            start_point = np.array([[0, 0, 0]])
            
            # Force arrow (blue)
            force_mag = np.sqrt(np.sum(force**2))
            if force_mag > 0.01:
                force_scale = 0.5  # Adjust scale based on visualization needs
                force_end = start_point + np.array([[force[0], force[1], force[2]]]) * force_scale
                self.force_arrow = self.plotter.add_arrows(
                    start_point, force_end, color='blue', opacity=0.8
                )
            
            # Torque arrow (red)
            torque_mag = np.sqrt(np.sum(torque**2))
            if torque_mag > 0.01:
                torque_scale = 2.0  # Adjust scale based on visualization needs
                torque_end = start_point + np.array([[torque[0], torque[1], torque[2]]]) * torque_scale
                self.torque_arrow = self.plotter.add_arrows(
                    start_point, torque_end, color='red', opacity=0.8
                )
            
            # Update rendering
            self.ren_win.Render()
        except Exception as e:
            print(f"Error updating force/torque visualization: {e}")
    
    def reset_camera(self):
        """Reset camera to default position"""
        if not PYVISTA_AVAILABLE:
            return
        try:
            self.plotter.reset_camera()
            self.ren_win.Render()
        except Exception as e:
            print(f"Error resetting camera: {e}")



class KneeFlexionExperiment(QMainWindow):
    def __init__(self):
    
        try:
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
            self.viz_timer.setInterval(100)  # 100ms for smoother updates
            
            # History for visualization
            self.history_size = 50  # Reduced history size for better performance
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

        except Exception as e:
            print(f"Critical error during initialization: {e}")
            import traceback
            traceback.print_exc()

    
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
            #print(self.forces)
            #print(self.torques)
            # Ensure we have at least some data
            if len(self.forces) == 0:
                raise ValueError("No valid data points found in file")
                
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            # Create dummy data if file not found or empty
            #self.forces = np.random.rand(100, 3) * 10 - 5  # Range from -5 to 5
            #self.torques = np.random.rand(200, 3) * 2 - 1  # Range from -1 to 1

            self.forces = np.zeros((200,3))
            self.forces[0] = np.random.uniform(-13, 13, size=3)
            for i in range(1, 200):
                delta = np.random.uniform(-0.8, 0.8, size=3)
                self.forces[i] = self.forces[i - 1] + delta
                self.forces[i] = np.clip(self.forces[i], -15, 15)

            self.torques = np.zeros((200,3))
            self.torques[0] = np.random.uniform(-1.5, 1.5, size=3)
            for i in range(1, 200):
                delta = np.random.uniform(-0.2, 0.2, size=3)
                self.torques[i] = self.torques[i - 1] + delta
                self.torques[i] = np.clip(self.torques[i], -1.5, 1.5)


            print("Using random dummy data instead.")
            #print(self.forces)
            #print(self.torques)
            

        except Exception as e:
            print(f"An error occurred: {e}")
            # Create dummy data if error
            self.forces = np.random.rand(100, 3) * 10 - 5
            self.torques = np.random.rand(100, 3) * 2 - 1
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
        
        # Left part: Image display and visualization
        self.left_widget = QWidget()
        left_layout = QVBoxLayout()


         #Create the QTabWidget
        self.tabs = QTabWidget()

        # Create tab pages (as QWidget)
        self.tab1 = QWidget()
        self.tab2 = QWidget()

         # Add first two tabs regardless of 3D visualization capability
        self.tabs.addTab(self.tab1, "Current Data")
        self.tabs.addTab(self.tab2, "Current + Previous Data")


            # Try to create bone visualization tab, but continue if it fails
        try:
            self.tab3 = QWidget()
            tab3_layout = QVBoxLayout()
            
            viz_label_3 = QLabel("3D Bone Model with Force & Torque")
            viz_label_3.setAlignment(Qt.AlignCenter)
            viz_label_3.setFont(QFont("Arial", 12, QFont.Bold))
            tab3_layout.addWidget(viz_label_3)
        
            # Create bone visualization with explicit error handling
            try:
                self.bone_viz = BoneVisualizationWidget()
                tab3_layout.addWidget(self.bone_viz)
                self.tab3.setLayout(tab3_layout)
                self.tabs.addTab(self.tab3, "3D Bone Model")
                print("3D bone visualization loaded successfully")
            except Exception as e:
                print(f"Failed to create bone visualization: {e}")
                import traceback
                traceback.print_exc()
                # Create a message instead
                error_label = QLabel(f"3D visualization failed to load:\n{str(e)}")
                error_label.setWordWrap(True)
                tab3_layout.addWidget(error_label)
                self.tab3.setLayout(tab3_layout)
                self.tabs.addTab(self.tab3, "3D Bone Model (Error)")
        except Exception as e:
            print(f"Failed to create Tab 3: {e}")
          
        
       
        
        
        

       


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
        #left_layout.addWidget(viz_label)
        tab2_layout.addWidget(viz_label_2)
        
        # Create matplotlib visualization
        self.canvas_history = MplCanvas(width=4, height=4)
        #left_layout.addWidget(self.canvas)
        tab2_layout.addWidget(self.canvas_history)
        
        self.tab2.setLayout(tab2_layout)
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
        right_layout.addWidget(self.rotate_button, 2, 0)
        right_layout.addWidget(self.varus_button, 3, 0)
        right_layout.addWidget(self.valgus_button, 4, 0)
        right_layout.addWidget(self.internal_rot_button, 5, 0)
        right_layout.addWidget(self.external_rot_button, 6, 0)
        right_layout.addWidget(self.lachmann_button, 7, 0)  # Add the Lachmann test button
        
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
        """Called by timer to update visualization"""
        if self.experiment_running:
            self.current_data_index = (self.current_data_index + 1) % len(self.forces)
            self.update_visualization(self.current_data_index)
        

    
    def update_visualization(self, data_index=0):
        """Update the force/torque visualization with the current data."""
        # Make sure we have data
        if len(self.forces) == 0 or len(self.torques) == 0:
            return
            
        # Get current data point
        idx = data_index % len(self.forces)
        force = self.forces[idx].copy()  # Make a copy to prevent modification of original data
        torque = self.torques[idx].copy()
        
        # Add to history
        self.force_history.append(force)
        self.torque_history.append(torque)
        
        # Keep history to specified size
        if len(self.force_history) > self.history_size:
            self.force_history.pop(0)
            self.torque_history.pop(0)
        
        # Update both visualizations
        self.update_current_visualization(force, torque)
        self.update_history_visualization()

         # Add code to update the bone visualization with error handling
        try:
            if hasattr(self, 'bone_viz') and self.bone_viz is not None:
                self.bone_viz.update_force_torque(force, torque)
        except Exception as e:
            print(f"Error updating bone visualization: {e}")
        
    def update_current_visualization(self, force, torque):
        """Update the force/torque visualization with only the current data."""
        # Clear previous plots
        self.canvas_current.axes_force.clear()
        self.canvas_current.axes_torque.clear()
        
        # Set up force plot limits and labels
        force_max = 10  # Increased for better visibility
        self.canvas_current.axes_force.set_xlim([-force_max, force_max])
        self.canvas_current.axes_force.set_ylim([-force_max, force_max])
        self.canvas_current.axes_force.set_zlim([-force_max, force_max])
        self.canvas_current.axes_force.set_title('Current Force (N)')
        self.canvas_current.axes_force.set_xlabel('X')
        self.canvas_current.axes_force.set_ylabel('Y')
        self.canvas_current.axes_force.set_zlabel('Z')
        self.canvas_current.axes_force.grid(False)  # Hide grid
        self.canvas_current.axes_force.set_axis_off()  # Hide axes
        
        # Set up torque plot limits and labels
        torque_max = 2
        self.canvas_current.axes_torque.set_xlim([-torque_max, torque_max])
        self.canvas_current.axes_torque.set_ylim([-torque_max, torque_max])
        self.canvas_current.axes_torque.set_zlim([-torque_max, torque_max])
        self.canvas_current.axes_torque.set_title('Current Torque (Nm)')
        self.canvas_current.axes_torque.set_xlabel('X')
        self.canvas_current.axes_torque.set_ylabel('Y')
        self.canvas_current.axes_torque.set_zlabel('Z')
        self.canvas_current.axes_torque.grid(False)  # Hide grid
        self.canvas_current.axes_torque.set_axis_off()  # Hide axes
        
        # Draw reference axes
        self.draw_reference_axes(self.canvas_current.axes_force, force_max * 0.2)
        self.draw_reference_axes(self.canvas_current.axes_torque, torque_max * 0.2)
        
        # Force arrow
        force_mag = np.sqrt(np.sum(force**2))
        if force_mag > 0.01:
            self.canvas_current.axes_force.quiver(0, 0, 0, 
                    force[0], force[1], force[2],
                    color='blue', 
                    linewidth=1,
                    normalize=False,
                    arrow_length_ratio=0.1)
        
        # Torque arrow
        torque_mag = np.sqrt(np.sum(torque**2))
        if torque_mag > 0.01:
            self.canvas_current.axes_torque.quiver(0, 0, 0, 
                    torque[0], torque[1], torque[2],
                    color='red', 
                    linewidth=1,
                    normalize=False,
                    arrow_length_ratio=0.1)
        
        # Display magnitudes
        self.canvas_current.axes_force.text2D(0.05, 0.95, f"Force Mag: {force_mag:.2f}N", transform=self.canvas_current.axes_force.transAxes)
        self.canvas_current.axes_torque.text2D(0.05, 0.95, f"Torque Mag: {torque_mag:.2f}Nm", transform=self.canvas_current.axes_torque.transAxes)
        
        # Add vector component values
        self.canvas_current.axes_force.text2D(0.05, 0.90, 
                              f"Fx: {force[0]:.2f}, Fy: {force[1]:.2f}, Fz: {force[2]:.2f}", 
                              transform=self.canvas_current.axes_force.transAxes, fontsize=8)
        self.canvas_current.axes_torque.text2D(0.05, 0.90, 
                               f"Tx: {torque[0]:.2f}, Ty: {torque[1]:.2f}, Tz: {torque[2]:.2f}", 
                               transform=self.canvas_current.axes_torque.transAxes, fontsize=8)
        
        # Update the figure
        self.canvas_current.fig.tight_layout()
        self.canvas_current.draw()
    
    def update_history_visualization(self):
        """Update the force/torque visualization with history data."""
        # Clear previous plots
        self.canvas_history.axes_force.clear()
        self.canvas_history.axes_torque.clear()
        
        # Set up force plot limits and labels
        force_max = 10  # Increased for better visibility
        self.canvas_history.axes_force.set_xlim([-force_max, force_max])
        self.canvas_history.axes_force.set_ylim([-force_max, force_max])
        self.canvas_history.axes_force.set_zlim([-force_max, force_max])
        self.canvas_history.axes_force.set_title('Force History (N)')
        self.canvas_history.axes_force.set_xlabel('X')
        self.canvas_history.axes_force.set_ylabel('Y')
        self.canvas_history.axes_force.set_zlabel('Z')
        self.canvas_history.axes_force.grid(False)  # Hide grid
        self.canvas_history.axes_force.set_axis_off()  # Hide axes
        
        # Set up torque plot limits and labels
        torque_max = 2
        self.canvas_history.axes_torque.set_xlim([-torque_max, torque_max])
        self.canvas_history.axes_torque.set_ylim([-torque_max, torque_max])
        self.canvas_history.axes_torque.set_zlim([-torque_max, torque_max])
        self.canvas_history.axes_torque.set_title('Torque History (Nm)')
        self.canvas_history.axes_torque.set_xlabel('X')
        self.canvas_history.axes_torque.set_ylabel('Y')
        self.canvas_history.axes_torque.set_zlabel('Z')
        self.canvas_history.axes_torque.grid(False)  # Hide grid
        self.canvas_history.axes_torque.set_axis_off()  # Hide axes
        
        # Draw reference axes
        self.draw_reference_axes(self.canvas_history.axes_force, force_max * 0.2)
        self.draw_reference_axes(self.canvas_history.axes_torque, torque_max * 0.2)
        
        # Plot history with color gradient (older = more transparent)
        cmap_force = plt.get_cmap('Blues')  # Blues goes from white to dark blue
        cmap_torque = plt.get_cmap('PuRd')  # PuRd goes from white to purple to red
        
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
            # Make sure width is proportionate but not too thin
            width_scale = max(0.5, 0.5 + 2.5 * (force_mag / max_force_mag) if max_force_mag > 0 else 0.5)
            
            color_force = cmap_force(color_idx)
            color_force = (*color_force[:3], alpha)  # Set alpha for the color
            
            # Only draw if magnitude is not zero
            if force_mag > 0.01:
                self.canvas_history.axes_force.quiver(0, 0, 0, 
                        hist_force[0], hist_force[1], hist_force[2],
                        color=color_force, 
                        #linewidth=width_scale,
                        linewidth=1,
                        normalize=False,  # Don't normalize to see true magnitudes
                        arrow_length_ratio=0.1)
            
            # Torque arrow
            torque_mag = np.sqrt(np.sum(hist_torque**2))
            #width_scale = max(0.5, 0.5 + 2.5 * (torque_mag / max_torque_mag) if max_torque_mag > 0 else 0.5)
            
            color_torque = cmap_torque(color_idx)
            color_torque = (*color_torque[:3], alpha)  # Set alpha for the color
            
            # Only draw if magnitude is not zero
            if torque_mag > 0.01:
                self.canvas_history.axes_torque.quiver(0, 0, 0, 
                        hist_torque[0], hist_torque[1], hist_torque[2],
                        color=color_torque, 
                        #linewidth=width_scale,
                        linewidth=1,
                        normalize=False,  # Don't normalize to see true magnitudes
                        arrow_length_ratio=0.1)
        
        # Display magnitudes of the current force/torque
        current_force = self.force_history[-1] if self.force_history else None
        current_torque = self.torque_history[-1] if self.torque_history else None
        
        if current_force is not None and current_torque is not None:
            force_mag = np.sqrt(np.sum(current_force**2))
            torque_mag = np.sqrt(np.sum(current_torque**2))
            
            self.canvas_history.axes_force.text2D(0.05, 0.95, f"Force Mag: {force_mag:.2f}N", transform=self.canvas_history.axes_force.transAxes)
            self.canvas_history.axes_torque.text2D(0.05, 0.95, f"Torque Mag: {torque_mag:.2f}Nm", transform=self.canvas_history.axes_torque.transAxes)
            
            # Add vector component values
            self.canvas_history.axes_force.text2D(0.05, 0.90, 
                                  f"Fx: {current_force[0]:.2f}, Fy: {current_force[1]:.2f}, Fz: {current_force[2]:.2f}", 
                                  transform=self.canvas_history.axes_force.transAxes, fontsize=8)
            self.canvas_history.axes_torque.text2D(0.05, 0.90, 
                                   f"Tx: {current_torque[0]:.2f}, Ty: {current_torque[1]:.2f}, Tz: {current_torque[2]:.2f}", 
                                   transform=self.canvas_history.axes_torque.transAxes, fontsize=8)
        
        # Update the figure
        self.canvas_history.fig.tight_layout()
        self.canvas_history.draw()


    
    def draw_reference_axes(self, ax, length):
        """Draw reference axes to help with orientation"""
        # X axis - red
        ax.quiver(0, 0, 0, length, 0, 0, color='red', linewidth=1, arrow_length_ratio=0.1)
        ax.text(length*1.1, 0, 0, "X", color='red')
        
        # Y axis - green
        ax.quiver(0, 0, 0, 0, length, 0, color='green', linewidth=1, arrow_length_ratio=0.1)
        ax.text(0, length*1.1, 0, "Y", color='green')
        
        # Z axis - blue
        ax.quiver(0, 0, 0, 0, 0, length, color='blue', linewidth=1, arrow_length_ratio=0.1)
        ax.text(0, 0, length*1.1, "Z", color='blue')
        
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
        else:
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

            # hide instructions
            self.next_label.hide()
            self.rotation_progress_label.hide()
            self.rotation_progress.hide()
                
            # Stop visualization timer
            if self.viz_timer.isActive():
                self.viz_timer.stop()
                
            # Reset experiment running flag
            self.experiment_running = False
    
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
    
    """def rotation_complete(self):
        self.timer.stop()
        self.seconds_timer.stop()
        self.rotation_progress.setValue(0)
        
        if self.current_angle_index >= (len(self.flexion_angles) - 1) and self.external_rot_button.isEnabled() == False:
            # Experiment complete
            self.instruction_label.setText("Experiment Complete!")
            self.current_angle_index = len(self.flexion_angles)  # Set to length to indicate completion
            self.update_display()"""
    
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
            # End of regular experiment - enable Lachmann test
            #self.lachmann_button.setEnabled(True)
            self.next_button.setEnabled(False)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KneeFlexionExperiment()
    window.show()
    sys.exit(app.exec_())