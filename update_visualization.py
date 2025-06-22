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
import math


class UpdateVisualization():
        # Add class variables to store landmark positions for angle calculations
    tibia_landmarks = {}
    femur_landmarks = {}
    current_knee_angles = {'flexion': 0.0, 
                           'adduction': 0.0, 
                           'rotation': 0.0, 
                           'anterior_posterior': 0.0, 
                           'medial_lateral': 0.0, 
                           'proximal_distal': 0.0, 
                           'medial_joint_gap': 0.0, 
                           'lateral_joint_gap':0.0}
    femurmediallateral = [0,0,0]
    femurproximaldistal = [0, 0, 0]
    femurvarusaxis = [0, 0, 0]

    tibiamediallateral = [0, 0, 0]
    tibiaproximaldistal = [0, 0, 0]
    tibiavarusaxis = [0, 0, 0]

    floatingaxis = [0, 0, 0]

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
                color='red', 
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
                color='blue', 
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
            cmap_force = plt.get_cmap('PuRd')
            cmap_torque = plt.get_cmap('Blues')
            
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
        force_scaled = force * constants.SCALE_FACTOR_ARROW

        # Set the position of the force arrow - attach to tibia at specific point
        tibiaproximal= UpdateVisualization.tibia_landmarks['tibia_proximal']['position']
        
        # Calculate end point for the arrow
        end_point = tibiaproximal + force_scaled
        
        # First, remove old arrows if they exist
        if hasattr(self, 'force_arrow_shaft') and self.force_arrow_shaft is not None:
            self.gl_view.removeItem(self.force_arrow_shaft)
        if hasattr(self, 'force_arrow_head') and self.force_arrow_head is not None:
            self.gl_view.removeItem(self.force_arrow_head)
        
        
        # Create new arrows
        self.force_arrow_shaft, self.force_arrow_head = MeshUtils.create_arrow(
            tibiaproximal, 
            end_point, 
            color=(1, 0, 0, 1), 
            arrow_size=constants.ARROW_SIZE_FORCE, 
            shaft_width=constants.SHAFT_WIDTH,
            mode = 'force'
        )
        
        # Add new arrows to view
        if self.force_arrow_shaft is not None:
            self.gl_view.addItem(self.force_arrow_shaft)
        if self.force_arrow_head is not None:
            self.gl_view.addItem(self.force_arrow_head)

        
        # same with torques
        # Get current data point
        idx = data_index % len(self.torques)
        torque = self.torques[idx].copy()

         # Scale forces for better visualization
        torque_scaled = torque * constants.SCALE_FACTOR_ARROW

        # Calculate end point for the arrow
        end_point_torque = tibiaproximal + torque_scaled

        # First, remove old arrows if they exist
        if hasattr(self, 'torque_arrow_shaft') and self.torque_arrow_shaft is not None:
            self.gl_view.removeItem(self.torque_arrow_shaft)
        if hasattr(self, 'torque_arrow_head') and self.torque_arrow_head is not None:
            self.gl_view.removeItem(self.torque_arrow_head)

        # Create new arrows
        self.torque_arrow_shaft, self.torque_arrow_head = MeshUtils.create_arrow(
            tibiaproximal, 
            end_point_torque, 
            color=constants.DEEPSKYBLUE, 
            arrow_size=constants.ARROW_SIZE_TORQUE, 
            shaft_width=constants.SHAFT_WIDTH,
            mode = 'torque'
        )

        # Add new arrows to view
        if self.torque_arrow_shaft is not None:
            self.gl_view.addItem(self.torque_arrow_shaft)
        if self.torque_arrow_head is not None:
            self.gl_view.addItem(self.torque_arrow_head)


        """# visualize important axes for grood and suntay
        femurdistal= UpdateVisualization.femur_landmarks['femur_distal']['position']
        femurmedial= UpdateVisualization.femur_landmarks['femur_medial']['position']
        femurlateral= UpdateVisualization.femur_landmarks['femur_lateral']['position']
        tibiamedial= UpdateVisualization.tibia_landmarks['tibia_medial']['position']
        
        #remove old axes from the plot
        if hasattr(self, 'femur_axis_shaft_ml') and self.femur_axis_shaft_ml is not None:
            self.gl_view.removeItem(self.femur_axis_shaft_ml)

        if hasattr(self, 'tibia_axis_shaft_pd') and self.tibia_axis_shaft_pd is not None:
            self.gl_view.removeItem(self.tibia_axis_shaft_pd)

        if hasattr(self, 'tibia_femur_floating_axis') and self.tibia_femur_floating_axis  is not None:
            self.gl_view.removeItem(self.tibia_femur_floating_axis)


        #Femur medial-lateral axis
        self.femur_axis_shaft_ml = MeshUtils.create_tibia_axis(
            femurmedial, 
            femurmedial + UpdateVisualization.femurmediallateral, 
            color=constants.SALMON, 
            arrow_size=500, shaft_width=2
        )
        #Tibia proximal-distal axis
        self.tibia_axis_shaft_pd = MeshUtils.create_tibia_axis(
            tibiaproximal, 
            tibiaproximal + UpdateVisualization.tibiaproximaldistal, 
            color=constants.LIMEGREEN, 
            arrow_size=500, shaft_width=2
        )
        #floating axis (cross product between the other two axes)
        self.tibia_femur_floating_axis = MeshUtils.create_tibia_axis(
            tibiaproximal, 
            tibiaproximal + UpdateVisualization.floatingaxis, 
            color=constants.MEDIUMSLATEBLUE, 
            arrow_size=500, 
            shaft_width=2
        )


        if self.femur_axis_shaft_ml is not None:
            self.gl_view.addItem(self.femur_axis_shaft_ml)

        if self.tibia_axis_shaft_pd is not None:
            self.gl_view.addItem(self.tibia_axis_shaft_pd)

        if self.tibia_femur_floating_axis is not None:
            self.gl_view.addItem(self.tibia_femur_floating_axis)"""

        # Create/update legend
        UpdateVisualization.create_legend(self)
        # Update bone angles
        #UpdateVisualization.update_bone_angles(self, data_index)
        
        # Update anatomical axes visualization
        #UpdateVisualization.update_axes_visualization(self, data_index)

   
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
        #print(name)
        #print(transform)

        # Store the landmark position directly (since it's already correctly calculated)
        landmark_data = {
            'position': np.array(transform)
        }
        
        if name.startswith('tibia'):
            UpdateVisualization.tibia_landmarks[name] = landmark_data
        elif name.startswith('femur'):
            UpdateVisualization.femur_landmarks[name] = landmark_data
        else:
            print(f"Warning: Unknown landmark type {name}")
            return
        
        # Calculate knee angles if we have sufficient landmarks
        if UpdateVisualization._has_required_landmarks():
            angles = UpdateVisualization.calculate_grood_suntay_angles()
            UpdateVisualization.current_knee_angles = angles
            #print(angles['adduction'])
            
            
            self.joint_angles_text.setText(
                    f"Joint Angles: \n Flexion: {int(angles['flexion'])}°\n "
                    f"Varus (-) / Valgus (+): {int(angles['adduction'])}°\n "
                    f"Int (-) and Ext (+) Rotation: {int(angles['rotation'])}°"
                )
            
            self.joint_translations_text.setText(
                    f"Translation: \n anterior(+) / posterior(-): {int(angles['anterior_posterior'])}mm\n "
                    f"medial(+) / lateral(-): {int(angles['medial_lateral'])}mm\n "
                    f"distal(+) / proximal(-): {int(angles['proximal_distal'])}mm"
                )
            


    @staticmethod
    def _has_required_landmarks():
        """Check if we have the minimum required landmarks for angle calculation."""
        required_tibia = ['tibia_medial', 'tibia_lateral', 'tibia_proximal', 'tibia_distal']
        required_femur = ['femur_medial', 'femur_lateral', 'femur_proximal', 'femur_distal']
        
        tibia_available = all(landmark in UpdateVisualization.tibia_landmarks for landmark in required_tibia)
        femur_available = all(landmark in UpdateVisualization.femur_landmarks for landmark in required_femur)
        
        return tibia_available and femur_available
    
    @staticmethod
    def calculate_grood_suntay_angles():
        """
        Calculate knee angles using the Grood and Suntay method.
        
        Returns:
            dict: Dictionary containing flexion, adduction, and rotation angles in degrees
        """
        try:
            # Get landmark positions directly (already transformed and correct)
            tibia_medial = UpdateVisualization.tibia_landmarks['tibia_medial']['position']
            tibia_lateral = UpdateVisualization.tibia_landmarks['tibia_lateral']['position']
            tibia_proximal = UpdateVisualization.tibia_landmarks['tibia_proximal']['position']
            tibia_distal = UpdateVisualization.tibia_landmarks['tibia_distal']['position']
            
            femur_medial = UpdateVisualization.femur_landmarks['femur_medial']['position']
            femur_lateral = UpdateVisualization.femur_landmarks['femur_lateral']['position']
            femur_proximal = UpdateVisualization.femur_landmarks['femur_proximal']['position']
            femur_distal = UpdateVisualization.femur_landmarks['femur_distal']['position']
            
            
            # Define coordinate systems according to Grood and Suntay
            
            # Femoral coordinate system
            # e1f: femoral flexion-extension axis (lateral - medial direction)
            e1f = femur_lateral - femur_medial
            if np.linalg.norm(e1f) < 1e-10:
                print("Warning: Femur medial-lateral vector is too small")
                return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
            e1f = e1f / np.linalg.norm(e1f)
            UpdateVisualization.femurmediallateral = e1f
            
            # Temporary femoral long axis (proximal - distal direction)
            temp_femur = femur_proximal - femur_distal
            if np.linalg.norm(temp_femur) < 1e-10:
                print("Warning: Femur proximal-distal vector is too small")
                return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
            temp_femur = temp_femur / np.linalg.norm(temp_femur)
            
            # e2f: femoral anterior-posterior axis (perpendicular to e1f and temp_femur)
            e2f = np.cross(e1f, temp_femur)
            if np.linalg.norm(e2f) < 1e-10:
                print("Warning: Femur coordinate system is degenerate")
                return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
            e2f = e2f / np.linalg.norm(e2f)

            UpdateVisualization.femurvarusaxis = e2f
            
            # e3f: femoral long axis (corrected, perpendicular to e3f and e1f)
            e3f = np.cross(e2f, e1f)
            e3f = e3f / np.linalg.norm(e3f)

            UpdateVisualization.femurproximaldistal = e2f
            
            # Tibial coordinate system
            # e3t: tibial long axis (proximal - distal direction)
            e3t = tibia_proximal - tibia_distal
            if np.linalg.norm(e3t) < 1e-10:
                print("Warning: Tibia proximal-distal vector is too small")
                return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
            e3t = e3t / np.linalg.norm(e3t)

            UpdateVisualization.tibiaproximaldistal = e3t
            
            # Temporary tibial medial-lateral axis
            temp_tibia = tibia_lateral - tibia_medial
            if np.linalg.norm(temp_tibia) < 1e-10:
                print("Warning: Tibia medial-lateral vector is too small")
                return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
            temp_tibia = temp_tibia / np.linalg.norm(temp_tibia)

            
            
            # e2t: tibial anterior-posterior axis
            e2t = np.cross(temp_tibia, e3t)
            if np.linalg.norm(e2t) < 1e-10:
                print("Warning: Tibia coordinate system is degenerate")
                return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
            e2t = e2t / np.linalg.norm(e3t)
            
            UpdateVisualization.tibiavarusaxis = e3t

            # e1t: tibial medial-lateral axis (corrected)
            e1t = np.cross(e3t, e2t)
            e1t = e1t / np.linalg.norm(e1t)

            UpdateVisualization.tibiamediallateral = e1t
            
            # Calculate floating axis (common perpendicular to e1f and e2t)
            floating_axis = np.cross(e1f, e3t)
            if np.linalg.norm(floating_axis) < 1e-10:
                print("Warning: Floating axis is degenerate")
                return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}
            floating_axis = floating_axis / np.linalg.norm(floating_axis)

            UpdateVisualization.floatingaxis = floating_axis

            # Define origins of coordinate systems (typically midpoints of key landmarks)
            femur_origin = (femur_medial + femur_lateral) / 2.0  # Midpoint of femoral condyles
            tibia_origin = (tibia_medial + tibia_lateral) / 2.0   # Midpoint of tibial plateau

                # ============= ANGLE CALCULATIONS =============
            
            # 1. FLEXION/EXTENSION ANGLE

            dot_product = np.dot(e2f,floating_axis)
            magnitude_e2f = np.linalg.norm(e2f)
            magnitude_floating_axis = np.linalg.norm(floating_axis)
            cos_flexion = dot_product / (magnitude_e2f*magnitude_floating_axis)
            #flexion_sign = np.cross(floating_axis, e2f)
            flexion = math.acos(cos_flexion)
            flexion_angle = flexion* 180.0 / np.pi

            """dot_product = np.dot(-floating_axis,e2f)
            magnitude_e2f = np.linalg.norm(e2f)
            magnitude_floating_axis = np.linalg.norm(floating_axis)
            sin_flexion = dot_product / (magnitude_floating_axis * magnitude_e2f)
            flexion = math.asin(sin_flexion)
            flexion_angle = flexion* 180.0 / np.pi"""
            
            
            # 2. ABDUCTION/ADDUCTION ANGLE  

            dot_product = np.dot(e1f, e3t)
            magnitude_e1f = np.linalg.norm(e1f)
            magnitude_e3t = np.linalg.norm(e3t)
            cos_adduction = dot_product / (magnitude_e1f * magnitude_e3t )
            adduction = math.acos(cos_adduction)
            adduction_angle = (adduction -(np.pi/2))* 180.0 / np.pi
            
            # 3. INTERNAL/EXTERNAL ROTATION ANGLE

            dot_product = np.dot(floating_axis, e1t)
            magnitude_e1t = np.linalg.norm(e1t)
            magnitude_floating_axis = np.linalg.norm(floating_axis)
            sin_rotation = dot_product / (magnitude_e1t * magnitude_floating_axis )
            rotation = math.asin(-sin_rotation)
            rotation_angle = rotation * 180.0 / np.pi


            # ============= TRANSLATION CALCULATIONS =============
        
            # Calculate translation vector from tibia origin to femur origin
            translation_vector = femur_origin - tibia_origin
            
            # Project translation onto Grood & Suntay axes 
            # Anterior-Posterior translation: along floating axis
            # (+ = anterior, - = posterior)
            anterior_posterior = -np.dot(translation_vector, floating_axis)

            # Medial-Lateral translation: along femoral flexion-extension axis (e1f)
            # (+ = medial, - = lateral)  get's influenced by adduction angle
            s1 = np.dot(translation_vector, e1f)
            s3 = np.dot(translation_vector, e3t)
            medial_lateral = s1 + s3 * math.cos(adduction)
            
            # Proximal-Distal translation: along tibial long axis (e3t)
            # (+ = proximal, - = distal) also gets influenced by adduction angle
            proximal_distal = -(-s3 -s1 *math.cos(adduction))
            

            #===========Clalculation distances femur condyles - tibia plateau medial and lateral========
            # medial
            medial_tibia_femur = femur_medial - tibia_medial
            m1 = np.dot(medial_tibia_femur, e1f)
            #m3 = np.dot(medial_tibia_femur, e2t) / np.linalg.norm(e2t)
            m3 = np.dot(medial_tibia_femur, e3t)
            medial_joint_gap = -(-m3 - m1 * math.cos(adduction))
            
            #lateral
            lateral_tibia_femur = femur_lateral - tibia_lateral
            l1 = np.dot(lateral_tibia_femur, e1f)
            #l3 = np.dot(lateral_tibia_femur, e2t) / np.linalg.norm(e2t)
            l3 = np.dot(lateral_tibia_femur, e3t)
            lateral_joint_gap = -(-l3 - l1 *math.cos(adduction))

            # Store results for debugging/visualization
            UpdateVisualization.knee_angles = {
                'flexion': flexion_angle,
                'adduction': adduction_angle, 
                'rotation': rotation_angle
            }
            
            UpdateVisualization.knee_translations = {
                'anterior_posterior': anterior_posterior,
                'medial_lateral': medial_lateral,
                'proximal_distal': proximal_distal,
                'medial_tibia_femur': medial_joint_gap,
                'lateral_tibia_femur': lateral_joint_gap
            }
            
            return {
                'flexion': flexion_angle,
                'adduction': adduction_angle,
                'rotation': rotation_angle,
                'anterior_posterior': anterior_posterior,
                'medial_lateral': medial_lateral,
                'proximal_distal': proximal_distal,
                'medial_tibia_femur': medial_joint_gap,
                'lateral_tibia_femur': lateral_joint_gap
            }
            
        except Exception as e:
            print(f"Error calculating Grood and Suntay angles and translations: {e}")
            import traceback
            traceback.print_exc()
            return {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0,
                'anterior_posterior': 0.0, 'medial_lateral': 0.0, 'proximal_distal': 0.0}
    
    @staticmethod
    def get_current_knee_angles():
        """Get the current knee angles."""
        return UpdateVisualization.current_knee_angles.copy()
    
    @staticmethod
    def reset_landmarks():
        """Reset all stored landmarks."""
        UpdateVisualization.tibia_landmarks.clear()
        UpdateVisualization.femur_landmarks.clear()
        UpdateVisualization.current_knee_angles = {'flexion': 0.0, 'adduction': 0.0, 'rotation': 0.0}




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


    def update_tibia_path(self):
        """Update the tibia position path visualization"""
        #tibia_position_annick = UpdateVisualization.tibia_landmarks['tibia_distal']['position']
        #print(tibia_position_annick)
        #print(UpdateVisualization.tibia_landmarks)
        try:
            # Extract tibia position data
            #tibia_position_annick = UpdateVisualization.tibia_landmarks['tibia_distal']['position']
            tibia_position_annick = [1, 1, 1]

            tibia_pos_x = tibia_position_annick[0]
            tibia_pos_y = tibia_position_annick[1]
            tibia_pos_z = tibia_position_annick[2]
            
            # Get time array (adjust column index as needed)
            time_array = 1
            
            # Update the canvas
            self.canvas_path.update_tibia_position_path(
                tibia_pos_x, tibia_pos_y, tibia_pos_z, time_array
            )
            
        except Exception as e:
            print(f"Error updating tibia path: {e}")

    def clear_tibia_path(self):
        """Clear the tibia position path visualization"""
        if hasattr(self.canvas_path, 'axes_position'):
            self.canvas_path.axes_position.clear()
            self.canvas_path._setup_position_plot()
            self.canvas_path.draw()

    # Alternative: If you want to automatically update when data changes
    def on_data_update(self):
        """Called whenever your data is updated"""
        # Your existing data update code...
        
        # Automatically update the tibia path if tab4 is active
        if hasattr(self, 'canvas_path') and self.canvas_path.mode == "position_path":
            self.update_tibia_path()

    @staticmethod
    def create_legend(main_window):
        """Create a legend for force and torque arrows"""
        # Remove existing legend items
        if hasattr(main_window, 'legend_items'):
            for item in main_window.legend_items:
                if item is not None:
                    main_window.gl_view.removeItem(item)
        
        main_window.legend_items = []
        
        # Position legend
        legend_x = -50
        legend_y = 50
        legend_z = 0
        
        try:
            # Create colored text items
            force_text = gl.GLTextItem(pos=(legend_x, legend_y, legend_z), 
                                      text="■ Force", color=(1, 0, 0, 1))
            torque_text = gl.GLTextItem(pos=(legend_x, legend_y - 15, legend_z), 
                                       text="■ Torque", color=(0, 0, 1, 1))
            
            main_window.legend_items = [force_text, torque_text]
            
            # Add to view
            for item in main_window.legend_items:
                main_window.gl_view.addItem(item)
                
        except Exception as e:
            print(f"Error creating legend: {e}")
            main_window.legend_items = []