
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QPixmap, QFont
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui
from OpenGL.GL import glBegin, glEnd, glVertex3f, glColor4f, GL_LINES, GL_LINE_SMOOTH, glEnable, glHint, GL_LINE_SMOOTH_HINT, GL_NICEST
import pyqtgraph.opengl as gl
import constants
import numpy as np
from mesh_utils import MeshUtils
from PyQt5.QtCore import Qt, QTimer
class UpdateVisualization():
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
        tibia_pos = MeshUtils.get_tibia_force_origin(self.last_tibia_position)
        
        # Calculate end point for the arrow
        end_point = tibia_pos + force_scaled
        
        # First, remove old arrows if they exist
        if hasattr(self, 'force_arrow_shaft') and self.force_arrow_shaft is not None:
            self.gl_view.removeItem(self.force_arrow_shaft)
        if hasattr(self, 'force_arrow_head') and self.force_arrow_head is not None:
            self.gl_view.removeItem(self.force_arrow_head)
        
        # Create new arrows
        self.force_arrow_shaft, self.force_arrow_head = MeshUtils.create_arrow(
            tibia_pos, end_point, color=(1, 0, 0, 1), arrow_size=constants.ARROW_SIZE, shaft_width=constants.SHAFT_WIDTH
        )
        
        # Add new arrows to view
        if self.force_arrow_shaft is not None:
            self.gl_view.addItem(self.force_arrow_shaft)
        if self.force_arrow_head is not None:
            self.gl_view.addItem(self.force_arrow_head)

        # Update bone angles
        UpdateVisualization.update_bone_angles(self, data_index)
        
        # Update anatomical axes visualization
        UpdateVisualization.update_axes_visualization(self, data_index)

   
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

    @staticmethod
    def update_bone_angles(self, data_index=0):
        """Update joint angles display based on current bone positions"""
        # Check if we have necessary data and bone tab is active
        if (not hasattr(self, 'last_femur_position') or 
            not hasattr(self, 'last_tibia_position') or
            self.tabs.currentIndex() != 2):
            return
            
        # If we have an analyzer, calculate the angles
        if hasattr(self, 'knee_analyzer') and self.knee_analyzer is not None:
            try:
                # Convert quaternions to landmarks
                femur_current_markers = UpdateVisualization.quaternion_to_landmarks(
                    self,
                    self.last_femur_position, 
                    self.last_femur_quaternion,
                    'femur'
                )
                
                tibia_current_markers = UpdateVisualization.quaternion_to_landmarks(
                    self,
                    self.last_tibia_position,
                    self.last_tibia_quaternion,
                    'tibia'
                )
                
                # Convert any numpy arrays to lists for the knee analyzer
                femur_markers_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                    for k, v in femur_current_markers.items()}
                tibia_markers_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                    for k, v in tibia_current_markers.items()}
                
                # Calculate angles
                angles = self.knee_analyzer.update_transformations(
                    femur_markers_dict, tibia_markers_dict)
                
                # Update text display
                self.joint_angles_text.setText(
                    f"Joint Angles: Flexion: {angles['flexion']:.1f}°, "
                    f"Varus/Valgus: {angles['varus_valgus']:.1f}°, "
                    f"Rotation: {angles['rotation']:.1f}°"
                )
            except Exception as e:
                print(f"Error calculating joint angles: {str(e)}")
                import traceback
                traceback.print_exc()

    @staticmethod
    def quaternion_to_landmarks(self, position, quaternion, bone_type):
        """Convert position and quaternion to landmarks for joint angle calculation"""
        # Define the original landmarks without any offsets
        if bone_type == 'femur':
            original_landmarks = {
                'proximal': np.array([77.49647521972656, -127.54686737060547, 911.6983032226562]),
                'distal': np.array([65.46070098876953, -113.15875244140625, 1384.9970703125]),
                'lateral': np.array([67.22425079345703, -157.83193969726562, 1399.614990234375]),
                'medial': np.array([83.37752532958984, -106.33291625976562, 1398.119384765625])
            }
        else:  # tibia
            original_landmarks = {
                'proximal': [66.52336883544922+15.419721603393555, -121.91870880126953+153.50636291503906, 1399.853271484375-1636.604736328125],
                'distal': [65.01982879638672+15.419721603393555, -115.64944458007812+153.50636291503906, 1804.212646484375-1636.604736328125],
                'lateral': [63.146968841552734+15.419721603393555, -147.86354064941406+153.50636291503906, 1407.7625732421875-1636.604736328125],
                'medial': [66.68541717529297+15.419721603393555, -103.38368225097656+153.50636291503906, 1400.172119140625-1636.604736328125]

            }
        
        # Apply consistent translation offset to all landmarks to align with reference frame
        # This should match the translation used in the Kabsch algorithm in load_femur and load_tibia
        offset = np.array([15.419721603393555, 153.50636291503906, -1636.604736328125])
        for key in original_landmarks:
            original_landmarks[key] = original_landmarks[key] + offset
        
        # Convert quaternion to rotation matrix (assuming quaternion = [w, x, y, z])
        qw, qx, qy, qz = quaternion
        rotation_matrix = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])
        
        # Transform each landmark by first applying rotation then translation
        transformed_landmarks = {}
        for key, point in original_landmarks.items():
            transformed_point = rotation_matrix @ point + position
            transformed_landmarks[key] = transformed_point
        
        return transformed_landmarks

    @staticmethod
    def visualize_anatomical_axes(self):
        """Create and store the anatomical axes visualization objects"""
        # Create objects for femur axes
        self.femur_axis_visuals = {
            'x': None,  # AP axis (anteroposterior)
            'y': None,  # PD axis (proximodistal)
            'z': None,  # ML axis (mediolateral)
            'origin': None  # Origin point
        }
        
        # Create objects for tibia axes
        self.tibia_axis_visuals = {
            'x': None,  # AP axis
            'y': None,  # PD axis
            'z': None,  # ML axis
            'origin': None  # Origin point
        }

    @staticmethod
    def update_axes_visualization(self, data_index=0):
        """Update the anatomical axes visualization in 3D bone view"""
        # Skip if not on the bone visualization tab
        if self.tabs.currentIndex() != 2:
            return
        
        # Skip if we don't have bone positions
        if not hasattr(self, 'last_femur_position') or not hasattr(self, 'last_tibia_position'):
            return
        
        # Skip if knee_analyzer is not initialized
        if not hasattr(self, 'knee_analyzer') or self.knee_analyzer is None:
            print("Warning: knee_analyzer is not initialized, skipping axes visualization update")
            return
        
        # Create axes visualization objects if they don't exist yet
        if not hasattr(self, 'femur_axis_visuals'):
            UpdateVisualization.visualize_anatomical_axes(self)
        
        # Get transformed landmarks - use the improved quaternion_to_landmarks function
        femur_landmarks = UpdateVisualization.quaternion_to_landmarks(
            self,
            self.last_femur_position,
            self.last_femur_quaternion,
            'femur'
        )
        
        tibia_landmarks = UpdateVisualization.quaternion_to_landmarks(
            self,
            self.last_tibia_position,
            self.last_tibia_quaternion,
            'tibia'
        )
        
        # Use KneeJointAnalyzer to get the current anatomical axes
        try:
            # Convert any numpy arrays to lists for the knee analyzer
            femur_landmarks_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                for k, v in femur_landmarks.items()}
            tibia_landmarks_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                for k, v in tibia_landmarks.items()}
            
            # Update the transformations
            angles = self.knee_analyzer.update_transformations(femur_landmarks_dict, tibia_landmarks_dict)
            
            # Get the current axes from the analyzer
            femur_axes = self.knee_analyzer.current_femur_axes
            tibia_axes = self.knee_analyzer.current_tibia_axes
            
            # Get origins - using distal femur and proximal tibia as joint center points
            femur_origin = femur_landmarks['distal']
            tibia_origin = tibia_landmarks['proximal']
            
            # Define axis lengths (scale as needed)
            axis_length = 50.0
            
            # Update femur axes visualization
            UpdateVisualization._update_axis_visual(
                self,
                self.femur_axis_visuals, 'x', 
                femur_origin, 
                femur_origin + femur_axes[:, 0] * axis_length,
                color=(1, 0, 0, 1)  # Red for AP
            )
            UpdateVisualization._update_axis_visual(
                self,
                self.femur_axis_visuals, 'y', 
                femur_origin, 
                femur_origin + femur_axes[:, 1] * axis_length,
                color=(0, 1, 0, 1)  # Green for PD
            )
            UpdateVisualization._update_axis_visual(
                self,
                self.femur_axis_visuals, 'z', 
                femur_origin, 
                femur_origin + femur_axes[:, 2] * axis_length,
                color=(0, 0, 1, 1)  # Blue for ML
            )
            
            # Update tibia axes visualization
            UpdateVisualization._update_axis_visual(
                self,
                self.tibia_axis_visuals, 'x', 
                tibia_origin, 
                tibia_origin + tibia_axes[:, 0] * axis_length,
                color=(1, 0, 0, 1)  # Red for AP
            )
            UpdateVisualization._update_axis_visual(
                self,
                self.tibia_axis_visuals, 'y', 
                tibia_origin, 
                tibia_origin + tibia_axes[:, 1] * axis_length,
                color=(0, 1, 0, 1)  # Green for PD
            )
            UpdateVisualization._update_axis_visual(
                self,
                self.tibia_axis_visuals, 'z', 
                tibia_origin, 
                tibia_origin + tibia_axes[:, 2] * axis_length,
                color=(0, 0, 1, 1)  # Blue for ML
            )
            
            # Update origin points
            UpdateVisualization._update_origin_visual(self, self.femur_axis_visuals, femur_origin, color=(1, 1, 1, 1))
            UpdateVisualization._update_origin_visual(self, self.tibia_axis_visuals, tibia_origin, color=(1, 1, 1, 1))
            
            # Update angle display
            self.joint_angles_text.setText(
                f"Joint Angles: Flexion: {angles['flexion']:.1f}°, " +
                f"Varus/Valgus: {angles['varus_valgus']:.1f}°, " +
                f"Rotation: {angles['rotation']:.1f}°"
            )
        except Exception as e:
            print(f"Error updating anatomical axes: {str(e)}")
            import traceback
            traceback.print_exc()


    @staticmethod
    def _update_axis_visual(self, axes_dict, axis_name, start_point, end_point, color):
        """Helper to update a single axis line visual"""
        # Remove existing item if it exists
        if axes_dict[axis_name] is not None:
            self.gl_view.removeItem(axes_dict[axis_name])
        
        # Create the new axis line
        axis_line = gl.GLLinePlotItem(
            pos=np.array([start_point, end_point]),
            color=color,
            width=constants.SHAFT_WIDTH,
            antialias=True
        )
        
        # Add to view and store reference
        self.gl_view.addItem(axis_line)
        axes_dict[axis_name] = axis_line

    @staticmethod
    def _update_origin_visual(self, axes_dict, position, color, size=5.0):
        """Helper to update the origin point visualization"""
        # Remove existing item if it exists
        if axes_dict['origin'] is not None:
            self.gl_view.removeItem(axes_dict['origin'])
        
        # Create a small sphere to represent the origin
        md = gl.MeshData.sphere(rows=10, cols=10, radius=size)
        origin_point = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=color,
            shader='shaded',
            glOptions='translucent'
        )
        origin_point.setGLOptions('opaque')
        origin_point.translate(position[0], position[1], position[2])
        
        # Add to view and store reference
        self.gl_view.addItem(origin_point)
        axes_dict['origin'] = origin_point

    @staticmethod
    def toggle_anatomical_axes(self, visible=True):
        """Toggle the visibility of anatomical axes"""
        if not hasattr(self, 'femur_axis_visuals') or not hasattr(self, 'tibia_axis_visuals'):
            # Axes haven't been created yet
            return
        
        # Toggle femur axes
        for key, item in self.femur_axis_visuals.items():
            if item is not None:
                if visible:
                    self.gl_view.addItem(item)
                else:
                    self.gl_view.removeItem(item)
        
        # Toggle tibia axes
        for key, item in self.tibia_axis_visuals.items():
            if item is not None:
                if visible:
                    self.gl_view.addItem(item)
                else:
                    self.gl_view.removeItem(item)

    def add_landmark(self, position, name):
        """
        Create a landmark in a fixed position
        """
        if not hasattr(self, "landmarks"):
            self.landmarks = {}
            self.landmarks_origin = {}

        landmark_size = 5
        # Create a sphere to represent the landmark
        md = gl.MeshData.sphere(rows=10, cols=10, radius=landmark_size)
        landmark_sphere = gl.GLMeshItem(
            meshdata=md,
            smooth=True,
            color=(1, 0.5, 0, 1),
            shader='shaded',
            glOptions='translucent'
        )
        self.gl_view.addItem(landmark_sphere)
        landmark_sphere.translate(position[0], position[1], position[2])
        
        # Add Sphere to class to update it later on
        self.landmarks[name] = landmark_sphere
        self.landmarks_origin[name] = position

    def update_landmark_alex(self, position, quaternion, name):
        """
        Update landmarks position
        """
        # Reset transformation cause setting a new translation does not replace the old transformation
        self.landmarks[name].resetTransform()

        # Calculating new landmark position
        transform_mesh = MeshUtils.quaternion_to_transform_matrix(quaternion, position)
        origin = self.landmarks_origin[name]
        transform = transform_mesh[:3,:3]@origin + transform_mesh[:3,3]
        self.landmarks[name].translate(transform[0], transform[1], transform[2])

    def add_coordinate_axes(self, position, rotation, name, axis_length=50.0):
        """
        Draws a 3D coordinate system at the given position and orientation.
        Args:
            position (array-like): The origin of the coordinate system (x, y, z).
            quaternion (array-like): The orientation as (qw, qx, qy, qz).
            axis_length (float): Length of each axis.
        Returns:
            dict: Dictionary with keys 'x', 'y', 'z' and their GLLinePlotItem objects.
        """

        if not hasattr(self, "CoSy"):
            self.CoSy = {}
            self.Cosy_origin = {}

        # Check if quaternion is a 4-element array (assume [qw, qx, qy, qz]), else treat as rotation matrix
        rotation = np.array(rotation)
        if rotation.shape == (4,):
            qw, qx, qy, qz = rotation
            rotation_matrix = np.array([
                [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
            ])
        elif rotation.shape == (3, 3):
            rotation_matrix = rotation
        else:
            raise ValueError("Input must be a quaternion (4,) or a rotation matrix (3,3)")
        
        position = np.array(position)

        # Define axis directions in local space
        axes = {
            'x': (rotation_matrix @ np.array([1, 0, 0])) * axis_length,
            'y': (rotation_matrix @ np.array([0, 1, 0])) * axis_length,
            'z': (rotation_matrix @ np.array([0, 0, 1])) * axis_length,
        }
        colors = {
            'x': (1, 0, 0, 1),  # Red
            'y': (0, 1, 0, 1),  # Green
            'z': (0, 0, 1, 1),  # Blue
        }

        self.CoSy[name] = {}
        for axis, vec in axes.items():
            start = position
            end = position + vec
            axis_line = gl.GLLinePlotItem(
                pos=np.array([start, end]),
                color=colors[axis],
                width=2,
                antialias=True
            )
            self.gl_view.addItem(axis_line)
            self.CoSy[name][axis] = axis_line
        self.Cosy_origin[name] = position
