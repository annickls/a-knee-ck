"""
Rotation Control GUI Module - Dual Bone Version (Fixed)

This module provides an interactive GUI for controlling 3D rotations of both
tibia and femur bones using sliders and numerical inputs.

Key Features:
- Dual bone support (tibia and femur)
- Vertically stacked layout (both bones visible at once)
- Large sliders for precise control
- Numerical input fields
- Individual and global reset buttons
- Separate signals for each bone
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSlider, QGroupBox,
                             QGridLayout, QDoubleSpinBox, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import numpy as np
from scipy.spatial.transform import Rotation as R


class RotationControlWidget(QWidget):
    """
    A widget for controlling 3D rotations of both tibia and femur bones.
    
    Both bones are displayed in a single window, stacked vertically for
    easy comparison and simultaneous adjustment.
    
    Signals:
        tibia_rotation_changed: Emitted when tibia rotation changes
        femur_rotation_changed: Emitted when femur rotation changes
    """
    
    # Define separate signals for each bone
    tibia_rotation_changed = pyqtSignal(tuple)  # (qx, qy, qz, qw)
    femur_rotation_changed = pyqtSignal(tuple)  # (qx, qy, qz, qw)
    
    def __init__(self, parent=None):
        """Initialize the dual-bone rotation control widget."""
        super().__init__(parent)
        
        # Initialize rotation offsets for both bones
        self.rotation_offsets = {
            'tibia': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'femur': {'x': 0.0, 'y': 0.0, 'z': 0.0}
        }
        
        # Store UI elements organized by bone
        self.sliders = {
            'tibia': {},
            'femur': {}
        }
        self.spin_boxes = {
            'tibia': {},
            'femur': {}
        }
        self.status_labels = {}
        
        # Build the user interface
        self.setup_ui()
        
    def setup_ui(self):
        """Create and arrange all UI elements."""
        # Main vertical layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel("Bone Rotation Control")
        title_font = QFont('Arial', 14, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Create control panels for both bones (stacked vertically)
        tibia_group = self._create_bone_control_group('tibia', 'Tibia', '#3498db')
        femur_group = self._create_bone_control_group('femur', 'Femur', '#e74c3c')
        
        main_layout.addWidget(tibia_group)
        main_layout.addWidget(femur_group)
        
        # Global reset button
        reset_all_btn = QPushButton("🔄 Reset All Rotations (Both Bones)")
        reset_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6b6b;
                color: white;
                font-weight: bold;
                padding: 12px;
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
        
        # Add stretch at the bottom
        main_layout.addStretch()
        
        # Set the layout
        self.setLayout(main_layout)
        
        # Set minimum size
        self.setMinimumWidth(550)
        self.setMinimumHeight(650)
        
    def _create_bone_control_group(self, bone_name, display_name, accent_color):
        """
        Create a control group for one bone.
        
        Args:
            bone_name: Internal identifier ('tibia' or 'femur')
            display_name: Display name ('Tibia' or 'Femur')
            accent_color: Color for the group box border
            
        Returns:
            QGroupBox: Complete control panel for the bone
        """
        # Create group box
        group_box = QGroupBox(f"{display_name} Rotation Offsets (Degrees)")
        group_box.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 11pt;
                border: 2px solid {accent_color};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                color: {accent_color};
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        
        # Main layout for this group
        group_layout = QVBoxLayout()
        group_layout.setSpacing(10)
        
        # Grid for sliders and controls
        controls_grid = QGridLayout()
        controls_grid.setSpacing(15)
        
        # Create controls for each axis
        axes = [
            ('X-Axis (Roll)', 'x', '#e74c3c'),   # Red
            ('Y-Axis (Pitch)', 'y', '#2ecc71'),  # Green
            ('Z-Axis (Yaw)', 'z', '#3498db')     # Blue
        ]
        
        for row, (axis_name, axis_key, axis_color) in enumerate(axes):
            self._create_axis_controls(
                controls_grid, row, axis_name, axis_key, 
                axis_color, bone_name
            )
        
        group_layout.addLayout(controls_grid)
        
        # Reset button for this bone
        reset_bone_btn = QPushButton(f"Reset {display_name} Rotation")
        reset_bone_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {accent_color};
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
                font-size: 10pt;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(accent_color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(accent_color, 0.3)};
            }}
        """)
        reset_bone_btn.clicked.connect(lambda: self.reset_bone(bone_name))
        group_layout.addWidget(reset_bone_btn)
        
        # Status label
        status_label = QLabel(f"Current {display_name} Rotation: Identity (no rotation)")
        status_label.setWordWrap(True)
        status_label.setStyleSheet("color: #666; font-size: 9pt; padding: 5px;")
        self.status_labels[bone_name] = status_label
        group_layout.addWidget(status_label)
        
        group_box.setLayout(group_layout)
        return group_box
        
    def _create_axis_controls(self, layout, row, axis_name, axis_key, color, bone_name):
        """
        Create slider and numerical input controls for one rotation axis.
        
        Args:
            layout: QGridLayout to add widgets to
            row: Row number in the grid
            axis_name: Display name (e.g., "X-Axis (Roll)")
            axis_key: Axis identifier ('x', 'y', or 'z')
            color: Color for the axis label
            bone_name: Bone identifier ('tibia' or 'femur')
        """
        # Column 0: Axis label
        label = QLabel(f"{axis_name}:")
        label.setStyleSheet(f"font-weight: bold; color: {color};")
        layout.addWidget(label, row, 0)
        
        # Column 1: Slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(-1800)  # -180.0 degrees (×10)
        slider.setMaximum(1800)   # +180.0 degrees (×10)
        slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(300)  # Tick every 30 degrees
        slider.setMinimumWidth(250)
        
        # Connect slider
        slider.valueChanged.connect(
            lambda value: self._on_slider_changed(bone_name, axis_key, value)
        )
        
        self.sliders[bone_name][axis_key] = slider
        layout.addWidget(slider, row, 1)
        
        # Column 2: Numerical input
        spin_box = QDoubleSpinBox()
        spin_box.setMinimum(-180.0)
        spin_box.setMaximum(180.0)
        spin_box.setValue(0.0)
        spin_box.setDecimals(2)
        spin_box.setSingleStep(0.1)
        spin_box.setSuffix("°")
        spin_box.setMinimumWidth(100)
        
        # Connect spin box
        spin_box.valueChanged.connect(
            lambda value: self._on_spinbox_changed(bone_name, axis_key, value)
        )
        
        self.spin_boxes[bone_name][axis_key] = spin_box
        layout.addWidget(spin_box, row, 2)
        
        # Column 3: Reset button
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
        reset_btn.clicked.connect(lambda: self.reset_axis(bone_name, axis_key))
        layout.addWidget(reset_btn, row, 3)
        
    def _darken_color(self, hex_color, factor=0.15):
        """Darken a hex color by a given factor."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        r = int(r * (1 - factor))
        g = int(g * (1 - factor))
        b = int(b * (1 - factor))
        
        return f'#{r:02x}{g:02x}{b:02x}'
        
    def _on_slider_changed(self, bone_name, axis, value):
        """Handle slider value changes."""
        degrees = value / 10.0
        
        # Update internal state
        self.rotation_offsets[bone_name][axis] = degrees
        
        # Synchronize spin box
        self.spin_boxes[bone_name][axis].blockSignals(True)
        self.spin_boxes[bone_name][axis].setValue(degrees)
        self.spin_boxes[bone_name][axis].blockSignals(False)
        
        # Emit signal
        self._emit_rotation_changed(bone_name)
        
    def _on_spinbox_changed(self, bone_name, axis, value):
        """Handle numerical input changes."""
        # Update internal state
        self.rotation_offsets[bone_name][axis] = value
        
        # Synchronize slider
        self.sliders[bone_name][axis].blockSignals(True)
        self.sliders[bone_name][axis].setValue(int(value * 10))
        self.sliders[bone_name][axis].blockSignals(False)
        
        # Emit signal
        self._emit_rotation_changed(bone_name)
        
    def _emit_rotation_changed(self, bone_name):
        """Calculate quaternion and emit the appropriate signal."""
        quat = self.get_rotation_quaternion(bone_name)
        
        # Emit appropriate signal
        if bone_name == 'tibia':
            self.tibia_rotation_changed.emit(quat)
        elif bone_name == 'femur':
            self.femur_rotation_changed.emit(quat)
        
        # Update status label
        if bone_name in self.status_labels:
            bone_display = bone_name.capitalize()
            self.status_labels[bone_name].setText(
                f"Current {bone_display} Rotation: "
                f"qx={quat[0]:.4f}, qy={quat[1]:.4f}, "
                f"qz={quat[2]:.4f}, qw={quat[3]:.4f}"
            )
            
    def reset_axis(self, bone_name, axis):
        """Reset a single axis to zero for a specific bone."""
        self.sliders[bone_name][axis].setValue(0)
        
    def reset_bone(self, bone_name):
        """Reset all axes to zero for a specific bone."""
        for axis in ['x', 'y', 'z']:
            self.sliders[bone_name][axis].blockSignals(True)
            self.spin_boxes[bone_name][axis].blockSignals(True)
            
            self.sliders[bone_name][axis].setValue(0)
            self.spin_boxes[bone_name][axis].setValue(0.0)
            self.rotation_offsets[bone_name][axis] = 0.0
            
            self.sliders[bone_name][axis].blockSignals(False)
            self.spin_boxes[bone_name][axis].blockSignals(False)
        
        # Emit once after all updates
        self._emit_rotation_changed(bone_name)
        
    def reset_all(self):
        """Reset all axes to zero for both bones."""
        for bone_name in ['tibia', 'femur']:
            for axis in ['x', 'y', 'z']:
                self.sliders[bone_name][axis].blockSignals(True)
                self.spin_boxes[bone_name][axis].blockSignals(True)
                
                self.sliders[bone_name][axis].setValue(0)
                self.spin_boxes[bone_name][axis].setValue(0.0)
                self.rotation_offsets[bone_name][axis] = 0.0
                
                self.sliders[bone_name][axis].blockSignals(False)
                self.spin_boxes[bone_name][axis].blockSignals(False)
        
        # Emit for both bones
        self._emit_rotation_changed('tibia')
        self._emit_rotation_changed('femur')
        
    def get_rotation_quaternion(self, bone_name='tibia'):
        """
        Calculate and return the rotation quaternion for a specific bone.
        
        Args:
            bone_name: 'tibia' or 'femur'
            
        Returns:
            tuple: Quaternion as (qx, qy, qz, qw)
        """
        rot_x = self.rotation_offsets[bone_name]['x']
        rot_y = self.rotation_offsets[bone_name]['y']
        rot_z = self.rotation_offsets[bone_name]['z']
        
        rotation = R.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=True)
        quat = rotation.as_quat()
        
        return tuple(quat)
        
    def get_tibia_quaternion(self):
        """Convenience method to get tibia quaternion."""
        return self.get_rotation_quaternion('tibia')
        
    def get_femur_quaternion(self):
        """Convenience method to get femur quaternion."""
        return self.get_rotation_quaternion('femur')
        
    def apply_rotation_to_quaternion(self, original_quat, bone_name='tibia'):
        """
        Apply rotation offset to an existing quaternion.
        
        Args:
            original_quat: Original quaternion [qx, qy, qz, qw] or [qw, qx, qy, qz]
            bone_name: 'tibia' or 'femur'
            
        Returns:
            numpy.ndarray: Modified quaternion [qx, qy, qz, qw]
        """
        original_quat = np.array(original_quat)
        
        # Auto-detect format
        if abs(original_quat[0]) > abs(original_quat[1:]).max():
            # Convert [qw, qx, qy, qz] to [qx, qy, qz, qw]
            original_quat = np.array([
                original_quat[1], original_quat[2], 
                original_quat[3], original_quat[0]
            ])
        
        original_rot = R.from_quat(original_quat)
        
        # Get offset rotation
        rot_x = self.rotation_offsets[bone_name]['x']
        rot_y = self.rotation_offsets[bone_name]['y']
        rot_z = self.rotation_offsets[bone_name]['z']
        
        offset_rot = R.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=True)
        
        # Combine rotations
        combined_rot = offset_rot * original_rot
        modified_quat = combined_rot.as_quat()
        
        return modified_quat
        
    def get_rotation_values(self, bone_name=None):
        """
        Get current rotation values in degrees.
        
        Args:
            bone_name: 'tibia', 'femur', or None for all bones
            
        Returns:
            dict: Rotation values
        """
        if bone_name is None:
            import copy
            return copy.deepcopy(self.rotation_offsets)
        else:
            return self.rotation_offsets[bone_name].copy()


# Test application
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QSplitter
    
    class TestWindow(QMainWindow):
        """Test window for the dual-bone rotation control."""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Dual Bone Rotation Control Test")
            self.setGeometry(100, 100, 1000, 700)
            
            # Create splitter
            splitter = QSplitter(Qt.Horizontal)
            
            # Create rotation control
            self.rotation_control = RotationControlWidget()
            
            # Connect signals
            self.rotation_control.tibia_rotation_changed.connect(
                self.on_rotation_changed
            )
            self.rotation_control.femur_rotation_changed.connect(
                self.on_rotation_changed
            )
            
            # Create output display
            self.output_text = QTextEdit()
            self.output_text.setReadOnly(True)
            
            splitter.addWidget(self.rotation_control)
            splitter.addWidget(self.output_text)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            
            self.setCentralWidget(splitter)
            self.update_display()
            
        def on_rotation_changed(self, quaternion):
            """Handle rotation changes."""
            self.update_display()
            
        def update_display(self):
            """Update the output display."""
            tibia_rot = self.rotation_control.get_rotation_values('tibia')
            femur_rot = self.rotation_control.get_rotation_values('femur')
            tibia_quat = self.rotation_control.get_tibia_quaternion()
            femur_quat = self.rotation_control.get_femur_quaternion()
            
            output = f"""
═══════════════════════════════════════════════════════════
                    TIBIA ROTATION
═══════════════════════════════════════════════════════════

Euler Angles:
  X (Roll):   {tibia_rot['x']:7.2f}°
  Y (Pitch):  {tibia_rot['y']:7.2f}°
  Z (Yaw):    {tibia_rot['z']:7.2f}°

Quaternion (x, y, z, w):
  qx: {tibia_quat[0]: .6f}
  qy: {tibia_quat[1]: .6f}
  qz: {tibia_quat[2]: .6f}
  qw: {tibia_quat[3]: .6f}

═══════════════════════════════════════════════════════════
                    FEMUR ROTATION
═══════════════════════════════════════════════════════════

Euler Angles:
  X (Roll):   {femur_rot['x']:7.2f}°
  Y (Pitch):  {femur_rot['y']:7.2f}°
  Z (Yaw):    {femur_rot['z']:7.2f}°

Quaternion (x, y, z, w):
  qx: {femur_quat[0]: .6f}
  qy: {femur_quat[1]: .6f}
  qz: {femur_quat[2]: .6f}
  qw: {femur_quat[3]: .6f}

═══════════════════════════════════════════════════════════
                    USAGE EXAMPLE
═══════════════════════════════════════════════════════════

# Get quaternions
tibia_quat = widget.get_tibia_quaternion()
femur_quat = widget.get_femur_quaternion()

# Apply to existing quaternions
modified_tibia = widget.apply_rotation_to_quaternion(
    original_tibia_quat, 'tibia'
)
modified_femur = widget.apply_rotation_to_quaternion(
    original_femur_quat, 'femur'
)
            """
            
            self.output_text.setPlainText(output)
    
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())