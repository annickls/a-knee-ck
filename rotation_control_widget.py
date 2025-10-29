"""
Rotation Control GUI Module

This module contains the RotationControlWidget class that provides
an interactive GUI for controlling 3D rotations using sliders and numerical inputs.

Design Philosophy:
- Separation of Concerns: By placing this in a separate module, we keep the rotation
  control logic separate from the main visualization code
- Reusability: This class can be imported and used in multiple projects
- Maintainability: Changes to rotation control don't affect the main application
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSlider, QLineEdit, QGroupBox,
                             QGridLayout, QDoubleSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QDoubleValidator
import numpy as np
from scipy.spatial.transform import Rotation as R


class RotationControlWidget(QWidget):
    """
    A widget for controlling 3D rotations with sliders and numerical inputs.
    
    This class demonstrates several important OOP and GUI programming concepts:
    
    1. **Signals and Slots (Observer Pattern)**:
       - The 'rotation_changed' signal notifies other parts of the application
         when rotation values change
       - This decouples the rotation control from the visualization logic
       - Any other widget can "listen" to changes without tight coupling
    
    2. **Encapsulation**:
       - All rotation logic is contained within this class
       - External code doesn't need to know HOW rotations are calculated
       - It only needs to call get_rotation_quaternion() to get results
    
    3. **State Management**:
       - The class maintains its own state (rotation_offsets dictionary)
       - Changes are synchronized between sliders and spin boxes
       - This prevents inconsistencies in the UI
    
    4. **Composition over Inheritance**:
       - This class inherits from QWidget (a simple base class)
       - It CONTAINS other widgets (sliders, buttons) rather than inheriting from them
       - This is more flexible than creating complex inheritance hierarchies
    
    Attributes:
        rotation_offsets (dict): Current rotation values for x, y, z axes in degrees
        rotation_changed (pyqtSignal): Emitted when any rotation value changes,
                                       sends the new quaternion as a tuple
    """
    
    # Define a custom signal that emits a tuple of 4 floats (quaternion: x, y, z, w)
    # Signals are a Qt mechanism for communication between objects
    rotation_changed = pyqtSignal(tuple)  # Will emit (qx, qy, qz, qw)
    
    def __init__(self, parent=None):
        """
        Initialize the rotation control widget.
        
        Args:
            parent: Parent widget (optional). If provided, this widget becomes
                   a child of that parent in the Qt widget hierarchy.
        
        The __init__ method is called when creating a new instance of the class.
        It sets up the initial state and builds the user interface.
        """
        super().__init__(parent)
        
        # Initialize rotation offsets dictionary
        # This stores the current rotation for each axis
        self.rotation_offsets = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        
        # Store references to UI elements for later access
        # These will be populated by setup_ui()
        self.sliders = {}       # Dictionary to store slider widgets
        self.spin_boxes = {}    # Dictionary to store numerical input widgets
        
        # Build the user interface
        self.setup_ui()
        
    def setup_ui(self):
        """
        Create and arrange all UI elements.
        
        This method demonstrates the typical pattern for building Qt GUIs:
        1. Create a main layout
        2. Create child widgets and layouts
        3. Add widgets to layouts
        4. Set the main layout for this widget
        
        Layouts in Qt automatically handle positioning and resizing of widgets.
        """
        # Main vertical layout - everything stacks vertically
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)  # Space between elements
        main_layout.setContentsMargins(15, 15, 15, 15)  # Margins around edges
        
        # Title label with custom styling
        title_label = QLabel("Tibia Rotation Control")
        title_font = QFont('Arial', 14, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create a group box to visually group related controls
        # This provides a nice border and title around the rotation controls
        rotation_group = QGroupBox("Rotation Offsets (Degrees)")
        rotation_layout = QGridLayout()  # Grid layout allows row/column positioning
        rotation_layout.setSpacing(15)
        
        # Create controls for each axis (X, Y, Z)
        axes = [
            ('X-Axis (Int-Ext)', 'x', 'red'),
            ('Y-Axis (Flex-Ex)', 'y', 'green'),
            ('Z-Axis (Var-Val)', 'z', 'blue')
        ]
        
        for row, (axis_name, axis_key, color) in enumerate(axes):
            # Create controls for this axis
            self._create_axis_controls(rotation_layout, row, axis_name, axis_key, color)
        
        rotation_group.setLayout(rotation_layout)
        main_layout.addWidget(rotation_group)
        
        # Reset All button at the bottom
        reset_all_btn = QPushButton("Reset All Rotations")
        reset_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6b6b;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #ee5a52;
            }
            QPushButton:pressed {
                background-color: #d64444;
            }
        """)
        reset_all_btn.clicked.connect(self.reset_all)
        main_layout.addWidget(reset_all_btn)
        
        # Add a status label to show current quaternion
        self.status_label = QLabel("Current Rotation: Identity (no rotation)")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666; font-size: 10px;")
        main_layout.addWidget(self.status_label)
        
        # Add stretch to push everything to the top
        main_layout.addStretch()
        
        # Set the layout for this widget
        self.setLayout(main_layout)
        
        # Set minimum size for the widget
        self.setMinimumWidth(400)
        self.setMinimumHeight(450)
        
    def _create_axis_controls(self, layout, row, axis_name, axis_key, color):
        """
        Create slider and numerical input controls for one rotation axis.
        
        This is a "private" helper method (indicated by the leading underscore).
        It's called only from within this class to avoid code duplication.
        
        Args:
            layout: The QGridLayout to add widgets to
            row: Which row in the grid to place these controls
            axis_name: Display name for the axis (e.g., "X-Axis (Roll)")
            axis_key: Dictionary key for this axis ('x', 'y', or 'z')
            color: Color name for visual distinction
        
        Design Note: Breaking this into a separate method follows the 
        "Single Responsibility Principle" - each method should do one thing well.
        """
        # Column 0: Axis label
        label = QLabel(f"{axis_name}:")
        label.setStyleSheet(f"font-weight: bold; color: {color};")
        layout.addWidget(label, row, 0)
        
        # Column 1: Slider (much larger than before for precision)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(-1800)  # -180.0 degrees (x10 for decimal precision)
        slider.setMaximum(1800)   #  180.0 degrees (x10 for decimal precision)
        slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(300)  # Tick every 30 degrees
        slider.setMinimumWidth(250)  # Much wider for precision
        
        # Connect slider value change to our handler
        # Lambda is used to pass the axis_key parameter
        slider.valueChanged.connect(lambda value: self._on_slider_changed(axis_key, value))
        
        self.sliders[axis_key] = slider
        layout.addWidget(slider, row, 1)
        
        # Column 2: Numerical input (QDoubleSpinBox)
        # This allows precise numerical entry
        spin_box = QDoubleSpinBox()
        spin_box.setMinimum(-180.0)
        spin_box.setMaximum(180.0)
        spin_box.setValue(0.0)
        spin_box.setDecimals(2)  # Show 2 decimal places
        spin_box.setSingleStep(0.1)  # Increment by 0.1 when using arrows
        spin_box.setSuffix("°")  # Add degree symbol
        spin_box.setMinimumWidth(100)
        
        # Connect spin box value change to our handler
        spin_box.valueChanged.connect(lambda value: self._on_spinbox_changed(axis_key, value))
        
        self.spin_boxes[axis_key] = spin_box
        layout.addWidget(spin_box, row, 2)
        
        # Column 3: Reset button for this axis
        reset_btn = QPushButton(f"Reset {axis_key.upper()}")
        reset_btn.setMaximumWidth(80)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #4ECDC4;
                color: white;
                font-weight: bold;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #45b8b0;
            }
        """)
        # Lambda captures the axis_key for this specific button
        reset_btn.clicked.connect(lambda: self.reset_axis(axis_key))
        layout.addWidget(reset_btn, row, 3)
        
    def _on_slider_changed(self, axis, value):
        """
        Handle slider value changes.
        
        This method is called when the user moves a slider.
        It updates the internal state and synchronizes the spin box.
        
        The underscore prefix indicates this is an internal method.
        External code should not call this directly.
        
        Args:
            axis: Which axis was changed ('x', 'y', or 'z')
            value: New slider value (in 1/10 degrees, so divide by 10)
        """
        # Convert slider value to degrees (slider is x10 for precision)
        degrees = value / 10.0
        
        # Update internal state
        self.rotation_offsets[axis] = degrees
        
        # Synchronize spin box without triggering its valueChanged signal
        # blockSignals temporarily prevents the widget from emitting signals
        self.spin_boxes[axis].blockSignals(True)
        self.spin_boxes[axis].setValue(degrees)
        self.spin_boxes[axis].blockSignals(False)
        
        # Notify listeners that rotation changed
        self._emit_rotation_changed()
        
    def _on_spinbox_changed(self, axis, value):
        """
        Handle numerical input changes.
        
        This method is called when the user types a value or uses spin box arrows.
        It updates the internal state and synchronizes the slider.
        
        Args:
            axis: Which axis was changed ('x', 'y', or 'z')
            value: New value in degrees (float)
        """
        # Update internal state
        self.rotation_offsets[axis] = value
        
        # Synchronize slider without triggering its valueChanged signal
        self.sliders[axis].blockSignals(True)
        self.sliders[axis].setValue(int(value * 10))  # Convert to slider scale
        self.sliders[axis].blockSignals(False)
        
        # Notify listeners that rotation changed
        self._emit_rotation_changed()
        
    def _emit_rotation_changed(self):
        """
        Calculate the resulting quaternion and emit the rotation_changed signal.
        
        This method demonstrates the Signal/Slot pattern:
        - This class doesn't know or care who is listening
        - It simply announces "the rotation changed, here's the new quaternion"
        - Any connected slots will automatically be called
        
        This loose coupling makes the code more modular and testable.
        """
        # Get the current quaternion
        quat = self.get_rotation_quaternion()
        
        # Emit the signal with the quaternion as a tuple
        self.rotation_changed.emit(quat)
        
        # Update status label for user feedback
        self.status_label.setText(
            f"Current Rotation: qx={quat[0]:.4f}, qy={quat[1]:.4f}, "
            f"qz={quat[2]:.4f}, qw={quat[3]:.4f}"
        )
        
    def reset_axis(self, axis):
        """
        Reset a single axis to zero.
        
        Public method that can be called from outside the class.
        
        Args:
            axis: Which axis to reset ('x', 'y', or 'z')
        """
        # Update slider (this will trigger the chain: slider -> _on_slider_changed -> emit)
        self.sliders[axis].setValue(0)
        
    def reset_all(self):
        """
        Reset all axes to zero rotation.
        
        Public method that can be called from outside the class.
        This demonstrates how to batch updates efficiently.
        """
        # Reset all three axes
        for axis in ['x', 'y', 'z']:
            # Block signals during reset to avoid multiple emissions
            self.sliders[axis].blockSignals(True)
            self.spin_boxes[axis].blockSignals(True)
            
            self.sliders[axis].setValue(0)
            self.spin_boxes[axis].setValue(0.0)
            self.rotation_offsets[axis] = 0.0
            
            self.sliders[axis].blockSignals(False)
            self.spin_boxes[axis].blockSignals(False)
        
        # Emit once after all updates
        self._emit_rotation_changed()
        
    def get_rotation_quaternion(self):
        """
        Calculate and return the resulting rotation as a quaternion.
        
        This is the main "output" method of this class.
        It converts the three rotation angles into a single quaternion.
        
        Returns:
            tuple: Quaternion as (qx, qy, qz, qw) in scipy format
        
        Mathematical Note:
        - We use Euler angles (xyz convention) for user input (intuitive)
        - We convert to quaternions for 3D calculations (mathematically robust)
        - Quaternions avoid gimbal lock and interpolate smoothly
        """
        # Get current rotation values
        rot_x = self.rotation_offsets['x']
        rot_y = self.rotation_offsets['y']
        rot_z = self.rotation_offsets['z']
        
        # Create rotation object from Euler angles
        # 'xyz' means: first rotate around X, then Y, then Z (extrinsic)
        rotation = R.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=True)
        
        # Convert to quaternion format: [qx, qy, qz, qw]
        quat = rotation.as_quat()
        
        return tuple(quat)
    
    def apply_rotation_to_quaternion(self, original_quat):
        """
        Apply the current rotation offset to an existing quaternion.
        
        This method demonstrates quaternion composition - combining rotations.
        
        Args:
            original_quat: Original quaternion as [qx, qy, qz, qw] or [qw, qx, qy, qz]
                          The method will auto-detect the format.
        
        Returns:
            numpy.ndarray: Modified quaternion as [qx, qy, qz, qw] (scipy format)
        
        Mathematical Note:
        - Quaternion multiplication is not commutative: Q1 * Q2 ≠ Q2 * Q1
        - We compute: offset_rotation * original_rotation
        - This applies the offset in the global coordinate frame
        """
        # Convert original quaternion to rotation object
        # Scipy uses [x, y, z, w] format
        original_quat = np.array(original_quat)
        
        # Auto-detect quaternion format
        # If the first element has largest absolute value, it's probably [qw, qx, qy, qz]
        if abs(original_quat[0]) > abs(original_quat[1:]).max():
            # Convert from [qw, qx, qy, qz] to [qx, qy, qz, qw]
            original_quat = np.array([
                original_quat[1], original_quat[2], 
                original_quat[3], original_quat[0]
            ])
        
        original_rot = R.from_quat(original_quat)
        
        # Get offset rotation
        rot_x = self.rotation_offsets['x']
        rot_y = self.rotation_offsets['y']
        rot_z = self.rotation_offsets['z']
        
        # Create offset rotation
        offset_rot = R.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=True)
        
        # Combine rotations: offset * original
        # This applies the offset rotation first, in the global frame
        combined_rot = offset_rot * original_rot
        
        # Convert back to quaternion
        modified_quat = combined_rot.as_quat()
        
        return modified_quat
    
    def get_rotation_values(self):
        """
        Get the current rotation values in degrees.
        
        Returns:
            dict: Dictionary with 'x', 'y', 'z' keys and degree values
        """
        return self.rotation_offsets.copy()


# Example usage and testing
if __name__ == "__main__":
    """
    This block runs only when this file is executed directly,
    not when it's imported as a module.
    
    It demonstrates how to use the RotationControlWidget.
    """
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QSplitter
    
    class TestWindow(QMainWindow):
        """Simple test window to demonstrate the rotation control."""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Rotation Control Test")
            self.setGeometry(100, 100, 800, 500)
            
            # Create splitter to show rotation control and output side by side
            splitter = QSplitter(Qt.Horizontal)
            
            # Create rotation control widget
            self.rotation_control = RotationControlWidget()
            
            # Connect its signal to our slot (callback method)
            self.rotation_control.rotation_changed.connect(self.on_rotation_changed)
            
            # Create text area to display quaternion output
            self.output_text = QTextEdit()
            self.output_text.setReadOnly(True)
            self.output_text.setPlaceholderText("Quaternion output will appear here...")
            
            splitter.addWidget(self.rotation_control)
            splitter.addWidget(self.output_text)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            
            self.setCentralWidget(splitter)
            
        def on_rotation_changed(self, quaternion):
            """
            This slot is called whenever the rotation changes.
            It demonstrates how to receive and use the quaternion output.
            """
            qx, qy, qz, qw = quaternion
            
            # Get the rotation values in degrees
            rot_values = self.rotation_control.get_rotation_values()
            
            output = f"""
Rotation Updated:
-----------------
X-axis (Roll):  {rot_values['x']:7.2f}°
Y-axis (Pitch): {rot_values['y']:7.2f}°
Z-axis (Yaw):   {rot_values['z']:7.2f}°

Resulting Quaternion:
---------------------
qx: {qx: .6f}
qy: {qy: .6f}
qz: {qz: .6f}
qw: {qw: .6f}

To use in your code:
--------------------
quaternion = {quaternion}
            """
            
            self.output_text.setPlainText(output)
    
    # Create and run the application
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())