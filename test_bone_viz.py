import sys
import vtk
import math
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QColorDialog, QCheckBox, QSlider
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class STLViewer(QMainWindow):
    def __init__(self, parent=None):
        super(STLViewer, self).__init__(parent)
        
        # Set window title and size
        self.setWindowTitle("STL Viewer - Femur and Tibia (60° Angle)")
        self.resize(1000, 800)
        
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.main_layout = QHBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)
        
        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self.central_widget)
        self.vtk_widget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        # Create renderer and render window
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        
        # Add VTK widget to layout
        self.main_layout.addWidget(self.vtk_widget, 4)
        
        # Create control panel
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.main_layout.addWidget(self.control_panel, 1)
        
        # Add controls for femur
        self.femur_label = QLabel("<b>Femur Controls</b>")
        self.femur_color_btn = QPushButton("Change Femur Color")
        self.femur_visibility = QCheckBox("Show Femur")
        self.femur_visibility.setChecked(True)
        
        # Add controls for tibia
        self.tibia_label = QLabel("<b>Tibia Controls</b>")
        self.tibia_color_btn = QPushButton("Change Tibia Color")
        self.tibia_visibility = QCheckBox("Show Tibia")
        self.tibia_visibility.setChecked(True)

        # Femur translation sliders
        self.control_layout.addWidget(QLabel("Femur Position (X, Y, Z):"))
        self.femur_x_slider = self.create_translation_slider(lambda val: self.update_translation('femur', 0, val))
        self.femur_y_slider = self.create_translation_slider(lambda val: self.update_translation('femur', 1, val))
        self.femur_z_slider = self.create_translation_slider(lambda val: self.update_translation('femur', 2, val))
        self.control_layout.addWidget(self.femur_x_slider)
        self.control_layout.addWidget(self.femur_y_slider)
        self.control_layout.addWidget(self.femur_z_slider)

        # Tibia translation sliders
        self.control_layout.addWidget(QLabel("Tibia Position (X, Y, Z):"))
        self.tibia_x_slider = self.create_translation_slider(lambda val: self.update_translation('tibia', 0, val))
        self.tibia_y_slider = self.create_translation_slider(lambda val: self.update_translation('tibia', 1, val))
        self.tibia_z_slider = self.create_translation_slider(lambda val: self.update_translation('tibia', 2, val))
        self.control_layout.addWidget(self.tibia_x_slider)
        self.control_layout.addWidget(self.tibia_y_slider)
        self.control_layout.addWidget(self.tibia_z_slider)


        
        # Add angle control
        self.angle_label = QLabel("<b>Angle Controls</b>")
        self.angle_value_label = QLabel("Angle: 60°")
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setMinimum(0)
        self.angle_slider.setMaximum(180)
        self.angle_slider.setValue(60)
        self.angle_slider.setTickPosition(QSlider.TicksBelow)
        self.angle_slider.setTickInterval(15)
        
        # Add general controls
        self.axes_visibility = QCheckBox("Show Axes")
        self.axes_visibility.setChecked(True)
        self.reset_camera_btn = QPushButton("Reset Camera")
        
        # Connect signals
        self.femur_color_btn.clicked.connect(lambda: self.change_color("femur"))
        self.tibia_color_btn.clicked.connect(lambda: self.change_color("tibia"))
        self.femur_visibility.stateChanged.connect(self.update_visibility)
        self.tibia_visibility.stateChanged.connect(self.update_visibility)
        self.axes_visibility.stateChanged.connect(self.update_visibility)
        self.angle_slider.valueChanged.connect(self.update_angle)
        self.reset_camera_btn.clicked.connect(self.reset_camera)
        
        # Add widgets to control layout
        self.control_layout.addWidget(self.femur_label)
        self.control_layout.addWidget(self.femur_color_btn)
        self.control_layout.addWidget(self.femur_visibility)
        self.control_layout.addWidget(self.tibia_label)
        self.control_layout.addWidget(self.tibia_color_btn)
        self.control_layout.addWidget(self.tibia_visibility)
        self.control_layout.addSpacing(10)
        self.control_layout.addWidget(self.angle_label)
        self.control_layout.addWidget(self.angle_value_label)
        self.control_layout.addWidget(self.angle_slider)
        self.control_layout.addSpacing(10)
        self.control_layout.addWidget(self.axes_visibility)
        self.control_layout.addWidget(self.reset_camera_btn)
        self.control_layout.addStretch()
        
        # Initialize actors
        self.femur_actor = None
        self.tibia_actor = None
        self.axes_actor = None

        # Translation values for femur and tibia
        self.femur_translation = [0, 0, -300]
        self.tibia_translation = [0, 0, 300]

        
        # Raw STL data
        self.femur_polydata = None
        self.tibia_polydata = None
        
        # Load STL files
        self.load_stl_files()
        
        # Position bones
        self.position_bones(60)  # Initial angle of 60 degrees
        
        # Initialize VTK widget
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

    def create_translation_slider(self, on_change_callback):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(-300, 300)
        slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)
        slider.valueChanged.connect(on_change_callback)
        return slider

        
    def load_stl_files(self):
        # Load femur STL
        femur_reader = vtk.vtkSTLReader()
        femur_reader.SetFileName("femur.stl")
        femur_reader.Update()
        self.femur_polydata = femur_reader.GetOutput()
        
        # Load tibia STL
        tibia_reader = vtk.vtkSTLReader()
        tibia_reader.SetFileName("tibia.stl")
        tibia_reader.Update()
        self.tibia_polydata = tibia_reader.GetOutput()
        
        # Create mappers
        femur_mapper = vtk.vtkPolyDataMapper()
        femur_mapper.SetInputData(self.femur_polydata)
        
        tibia_mapper = vtk.vtkPolyDataMapper()
        tibia_mapper.SetInputData(self.tibia_polydata)
        
        # Create actors
        self.femur_actor = vtk.vtkActor()
        self.femur_actor.SetMapper(femur_mapper)
        self.femur_actor.GetProperty().SetColor(0.9, 0.7, 0.7)  # Pinkish color for femur
        
        self.tibia_actor = vtk.vtkActor()
        self.tibia_actor.SetMapper(tibia_mapper)
        self.tibia_actor.GetProperty().SetColor(0.7, 0.7, 0.9)  # Bluish color for tibia
        
        # Create coordinate axes
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(50, 50, 50)  # Adjust based on your model size
        axes.SetXAxisLabelText("X")
        axes.SetYAxisLabelText("Y")
        axes.SetZAxisLabelText("Z")
        self.axes_actor = axes
        
        # Add actors to renderer
        self.renderer.AddActor(self.femur_actor)
        self.renderer.AddActor(self.tibia_actor)
        self.renderer.AddActor(self.axes_actor)
        
        # Set background color
        self.renderer.SetBackground(0.2, 0.2, 0.2)
    
    def find_bone_tip(self, polydata):
        """Find the tip point of the bone (assumed to be at the furthest y extent)"""
        bounds = polydata.GetBounds()
        # Assuming y+ is the direction of the bone length
        # You might need to adjust this based on your STL files' orientation
        #return [0, bounds[3], 0]  # Using the max y point as the tip
        return [0, bounds[3], 0]
        print(polydata.GetBounds)

    
    def position_bones(self, angle_degrees):
        if not self.femur_actor or not self.tibia_actor:
            return
            
        # Calculate the angle in radians
        angle_rad = math.radians(angle_degrees)
        
        # Find tips of bones
        femur_tip = self.find_bone_tip(self.femur_polydata)
        tibia_tip = self.find_bone_tip(self.tibia_polydata)
        
        # Reset transformations
        self.femur_actor.SetPosition(0, 0, -100)
        self.femur_actor.SetOrientation(0, 0, 0)
        self.tibia_actor.SetPosition(0, 0, 10)
        self.tibia_actor.SetOrientation(0, 0, 0)
        
        # 1. Position femur
        # Move femur so its tip is at origin, pointing along +y axis
        femur_center = self.femur_polydata.GetCenter()
        femur_tip_to_origin = [-femur_tip[0], -femur_tip[1], -femur_tip[2]]
        self.femur_actor.SetPosition(femur_tip_to_origin)
        
        # 2. Position tibia
        # Move tibia so its tip is at origin
        tibia_center = self.tibia_polydata.GetCenter()
        tibia_tip_to_origin = [-tibia_tip[0], -tibia_tip[1], -tibia_tip[2]]
        self.tibia_actor.SetPosition(tibia_tip_to_origin)
        
        # Rotate tibia around the z-axis by the specified angle
        self.tibia_actor.RotateZ(angle_degrees)
        
        # Refresh the render window
        self.vtk_widget.GetRenderWindow().Render()
    
    def update_angle(self, value):
        self.angle_value_label.setText(f"Angle: {value}°")
        self.position_bones(value)

    def update_translation(self, bone, axis, value):
        if bone == 'femur':
            self.femur_translation[axis] = value
            if self.femur_actor:
                self.femur_actor.SetPosition(*self.femur_translation)
        elif bone == 'tibia':
            self.tibia_translation[axis] = value
            if self.tibia_actor:
                self.tibia_actor.SetPosition(*self.tibia_translation)
        self.vtk_widget.GetRenderWindow().Render()

    
    def change_color(self, bone):
        color = QColorDialog.getColor()
        if color.isValid():
            r, g, b = color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0
            if bone == "femur" and self.femur_actor:
                self.femur_actor.GetProperty().SetColor(r, g, b)
            elif bone == "tibia" and self.tibia_actor:
                self.tibia_actor.GetProperty().SetColor(r, g, b)
            self.vtk_widget.GetRenderWindow().Render()
    
    def update_visibility(self):
        if self.femur_actor:
            self.femur_actor.SetVisibility(self.femur_visibility.isChecked())
        if self.tibia_actor:
            self.tibia_actor.SetVisibility(self.tibia_visibility.isChecked())
        if self.axes_actor:
            self.axes_actor.SetVisibility(self.axes_visibility.isChecked())
        self.vtk_widget.GetRenderWindow().Render()
    
    def reset_camera(self):
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = STLViewer()
    window.show()
    sys.exit(app.exec_())