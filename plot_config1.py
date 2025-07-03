from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph.opengl as gl
from OpenGL.GL import glBegin, glEnd, glVertex3f, glColor4f, GL_LINES, GL_LINE_SMOOTH, glEnable, glHint, GL_LINE_SMOOTH_HINT, GL_NICEST
import constants
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from update_visualization import UpdateVisualization
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import queue
import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPainter, QPen, QBrush, QFont, QColor
from PyQt5.QtCore import Qt, QTimer
from PIL import Image
import time
import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush
import constants

class MplCanvas(FigureCanvas):
    """Matplotlib canvas class for embedding plots in Qt that can display either current or historical force/torque data"""
    def __init__(self, width=5, height=4, mode="current"):
        self.fig = Figure(figsize=(width, height))
        
        self.mode = mode  # "current", "history", or "position_path"
        
        if mode == "position_path":
            # Single 3D plot for tibia position path
            self.axes_position = self.fig.add_subplot(111, projection='3d')
            self._setup_position_plot()
        else:
            # Original dual plot setup for force/torque
            self.axes_force = self.fig.add_subplot(121, projection='3d')
            self.axes_torque = self.fig.add_subplot(122, projection='3d')
            self._setup_force_torque_plots()
        
        # Initialize the FigureCanvas
        super().__init__(self.fig)
        self.fig.tight_layout()

        if mode == "varus_valgus" or mode == "rotation" or mode =="adduction" or mode == "anterior" or mode == "medial":
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel('x-axis')
            self.ax.set_ylabel('Flexion Angle [°]')
            #self.ax.set_title('Real-time Flexion vs Varus/Valgus')
            self.ax.grid(True, alpha=0.3)

            self.fig.subplots_adjust(left=0.15, bottom=0.15, right = 0.95, top =0.90)
            
            # Set initial axis limits
            self.ax.set_xlim(-constants.X_LIM_ROT, constants.X_LIM_ROT)  # Adjust range as needed
            self.ax.set_ylim(constants.Y_MIN_FLEX, constants.Y_MAX_FLEX)   # Adjust range as needed
            
            # Add vertical line at x=0 for reference
            self.ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Initialize empty line for real-time plotting
            self.line, = self.ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
            self.current_point, = self.ax.plot([], [], 'ro', markersize=8)
            
            # Store data for plotting
            self.varus_valgus_data = []
            self.flexion_data = []
            self.testvariable = 0
     
    
    def _setup_position_plot(self):
        """Setup the tibia position path plot"""
        # Set initial limits (will be adjusted based on data)
        self.axes_position.set_xlim([-50, 50])  # Adjust based on your data range
        self.axes_position.set_ylim([-50, 50])
        self.axes_position.set_zlim([-50, 50])
        
        self.axes_position.set_xlabel('X position (mm)')
        self.axes_position.set_ylabel('Y position (mm)')
        self.axes_position.set_zlabel('Z position (mm)')
        self.axes_position.set_title('Tibia Position Path')
        self.axes_position.grid(True, alpha=0.3)
        
        # Store reference to scatter and line plots for updates
        self.position_scatter = None
        self.position_line = None
        self.position_colorbar = None
    
    def _setup_force_torque_plots(self):
        """Setup the original force/torque plots"""
        # Set up axes once
        force_max = constants.FORCE_MAX
        torque_max = constants.TORQUE_MAX
        
        # Force plot setup
        self.axes_force.set_xlim([-force_max, force_max])
        self.axes_force.set_ylim([-force_max, force_max])
        self.axes_force.set_zlim([-force_max, force_max])
        self.axes_force.grid(False)
        self.axes_force.set_axis_off()
        
        # Torque plot setup
        self.axes_torque.set_xlim([-torque_max, torque_max])
        self.axes_torque.set_ylim([-torque_max, torque_max])
        self.axes_torque.set_zlim([-torque_max, torque_max])
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
        if self.mode == "history":
            self.force_arrows = []
            self.torque_arrows = []
        
        # Text elements for magnitudes and components
        self.force_mag_text = self.axes_force.text2D(0.32, 1.0, "", transform=self.axes_force.transAxes)
        self.torque_mag_text = self.axes_torque.text2D(0.4, 1.0, "", transform=self.axes_torque.transAxes)
        self.force_comp_text = self.axes_force.text2D(0.32, 0.95, "", transform=self.axes_force.transAxes, fontsize=8)
        self.torque_comp_text = self.axes_torque.text2D(0.4, 0.95, "", transform=self.axes_torque.transAxes, fontsize=8)

    """def update_varus_valgus_plot(self, flexion_angle, var_val_displacement, mode, mode_points):
        #Update the varus/valgus vs flexion plot by adding only the newest data point

        
        self.testvariable +=1
        
        
        
        # For the first point, setup the plot
        #if len(self.varus_valgus_data) == 1:
        if self.testvariable == 1:
            self.ax.clear()
            self.ax.set_xlabel('x-axis')
            self.ax.set_ylabel('flexion angle [°]')
            self.ax.set_title('medial/lateral joint gap [mm]')
            self.ax.grid(True, alpha=0.3)
            self.ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            self.ax.set_xlim(-constants.X_LIM_ROT, constants.X_LIM_ROT)
            self.ax.set_ylim(constants.Y_MIN_FLEX, constants.Y_MAX_FLEX)

        if mode == "varus_valgus":
            self.ax.set_xlabel('medial joint gap          lateral joint gap')
            self.ax.set_xlim(-constants.X_LIM_VAL, constants.X_LIM_VAL)
        elif mode == "rotation":
            self.ax.set_xlabel('external rotation         internal rotation')
            self.ax.set_xlim(-constants.X_LIM_ROT, constants.X_LIM_ROT)
        elif mode == "adduction":
            self.ax.set_xlabel('valgus        varus')
            self.ax.set_xlim(-constants.X_LIM_VAL, constants.X_LIM_VAL)
        
        color = constants.SALMON if var_val_displacement > 0 else constants.LIMEGREEN
        if mode_points == "bars":
            
            # Draw horizontal line from 0 to displacement value for new point
            self.line, = self.ax.plot([0, var_val_displacement], [flexion_angle, flexion_angle], 
                                color=color, linewidth=2, alpha=0.7)
            
            # Add current point (red dot) - remove previous current point if it exists
            if hasattr(self, 'current_point') and self.current_point:
                self.current_point.remove()
            
            self.current_point, = self.ax.plot(var_val_displacement, flexion_angle, 'ro', 
                                            markersize=8, zorder=10)
            
            # Add current bar highlight - remove previous highlight if it exists
            #if hasattr(self, 'current_highlight') and self.current_highlight:
            #    self.current_highlight.remove()
                
            #current_color = 'red' if var_val_displacement > 0 else 'blue'
            #self.current_highlight, = self.ax.plot([0, var_val_displacement], [flexion_angle, flexion_angle], 
            #                                 color=current_color, linewidth=4, alpha=0.8, zorder=5)
        else:
            range_filter_plot = constants.RANGE_FILTER_PLOT
            if mode == "rotation":
                if -range_filter_plot < UpdateVisualization.current_knee_angles['adduction'] < range_filter_plot:
                    self.current_point, = self.ax.plot(var_val_displacement, flexion_angle, 'o', color = color, 
                                                    markersize=4, zorder=10)
                    
            else:
                if -range_filter_plot < UpdateVisualization.current_knee_angles['rotation'] < range_filter_plot:
                    self.current_point, = self.ax.plot(var_val_displacement, flexion_angle, 'o', color = color, 
                                                    markersize=4, zorder=10)

        
        self.draw()"""

    
    def update_tibia_position_path(self, tibia_pos_x, tibia_pos_y, tibia_pos_z, time_array):
        """Update the tibia position path visualization"""
        if self.mode != "position_path":
            return
        
        # Clear previous plots
        if self.position_scatter:
            self.position_scatter.remove()
        if self.position_line:
            self.position_line[0].remove()
        if self.position_colorbar:
            self.position_colorbar.remove()
        
        # Create new scatter plot with time-based coloring
        self.position_scatter = self.axes_position.scatter(
            tibia_pos_x, tibia_pos_y, tibia_pos_z,
            c=time_array, cmap='viridis', s=40, alpha=0.8
        )
        
        # Create trajectory line
        self.position_line = self.axes_position.plot(
            tibia_pos_x, tibia_pos_y, tibia_pos_z, 
            'b-', linewidth=1.5
        )
        
        # Add colorbar for time reference
        self.position_colorbar = self.fig.colorbar(
            self.position_scatter, ax=self.axes_position, pad=0.1
        )
        self.position_colorbar.set_label('Time (s)')
        
        # Adjust plot limits based on data
        margin = 0.1
        x_range = np.ptp(tibia_pos_x)
        y_range = np.ptp(tibia_pos_y)
        z_range = np.ptp(tibia_pos_z)
        
        self.axes_position.set_xlim([
            np.min(tibia_pos_x) - margin * x_range,
            np.max(tibia_pos_x) + margin * x_range
        ])
        self.axes_position.set_ylim([
            np.min(tibia_pos_y) - margin * y_range,
            np.max(tibia_pos_y) + margin * y_range
        ])
        self.axes_position.set_zlim([
            np.min(tibia_pos_z) - margin * z_range,
            np.max(tibia_pos_z) + margin * z_range
        ])
        
        # Refresh the canvas
        self.draw()


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




    
class OptimizedVarusValgusPlot(QWidget):
    """Ultra-optimized varus/valgus vs flexion plot for real-time knee joint data - PyQt version."""
    
    def __init__(self, parent=None, width=800, height=600, max_points=10000):
        super().__init__(parent)
    
        self.width, self.height = width, height
        self.max_points = max_points  # Remove the reassignment line
        
        # Set widget size
        self.setFixedSize(width, height)
        
        # Pre-allocate all data arrays (zero memory allocation during runtime)
        # Make sure these are NumPy arrays, not Python lists
        self.flexion_data = np.zeros(self.max_points, dtype=np.float32)
        self.varus_valgus_data = np.zeros(self.max_points, dtype=np.float32)
        self.rotation_data = np.zeros(self.max_points, dtype=np.float32)
        self.adduction_data = np.zeros(self.max_points, dtype=np.float32)
        self.mode_data = np.zeros(self.max_points, dtype=np.uint8)
        self.anterior_data = np.zeros(self.max_points, dtype=np.float32)
        self.medial_data = np.zeros(self.max_points, dtype=np.float32)
        
        # Debug: Print array types and shapes to verify they're created correctly
        print(f"Array types and shapes:")
        print(f"flexion_data: {type(self.flexion_data)}, shape: {self.flexion_data.shape}")
        print(f"max_points: {self.max_points}")
        
        self.write_idx = 0
        self.point_count = 0
        
        # Rest of your initialization code...
        # Plot bounds - adjust these based on your constants
        self.x_lim_val = constants.X_LIM_VAL
        self.x_lim_rot = constants.X_LIM_ROT
        self.x_lim_anterior = constants.X_LIM_ANTERIOR
        self.x_lim_medial = constants.X_LIM_MEDIAL
        self.y_min_flex = constants.Y_MIN_FLEX
        self.y_max_flex = constants.Y_MAX_FLEX
        
        # Current mode and display settings
        self.current_mode = "varus_valgus"
        self.current_point_mode = "bars"
        
        # Pre-allocate image array for fast rendering
        self.img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Colors (RGB values)
        self.bg_color = np.array([255, 255, 255], dtype=np.uint8)
        self.grid_color = np.array([200, 200, 200], dtype=np.uint8)
        self.axis_color = np.array([100, 100, 100], dtype=np.uint8)
        self.salmon_color = np.array([250, 128, 114], dtype=np.uint8)
        self.limegreen_color = np.array([50, 205, 50], dtype=np.uint8)
        self.current_point_color = np.array([255, 0, 0], dtype=np.uint8)
        
        # Pre-compute coordinate transform parameters
        self._update_transform_params()
        
        # Initialize plot
        self._draw_static_elements()
        
        # Performance tracking
        self.frame_count = 0
        self.last_time = time.time()
        
        # Timer for automatic updates (optional)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        
    def _update_transform_params(self):
        """Update coordinate transformation parameters based on current mode."""
        if self.current_mode == "varus_valgus":
            self.x_min, self.x_max = -self.x_lim_val, self.x_lim_val
        elif self.current_mode == "rotation":
            self.x_min, self.x_max = -self.x_lim_rot, self.x_lim_rot
        elif self.current_mode == "adduction":
            self.x_min, self.x_max = -self.x_lim_val, self.x_lim_val
        elif self.current_mode == "anterior":
            self.x_min, self.x_max = -self.x_lim_anterior, self.x_lim_anterior
        elif self.current_mode == "medial":
            self.x_min, self.x_max = -self.x_lim_anterior, self.x_lim_anterior

            
        # Plot area margins
        self.margin_left = 80
        self.margin_right = 40
        self.margin_top = 40
        self.margin_bottom = 80
        
        self.plot_width = self.width - self.margin_left - self.margin_right
        self.plot_height = self.height - self.margin_top - self.margin_bottom
        
        # Transform parameters
        self.x_scale = self.plot_width / (self.x_max - self.x_min)
        self.y_scale = self.plot_height / (self.y_max_flex - self.y_min_flex)
        
    def _world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        screen_x = int(self.margin_left + (x - self.x_min) * self.x_scale)
        screen_y = int(self.margin_top + self.plot_height - (y - self.y_min_flex) * self.y_scale)
        return screen_x, screen_y
        
    def _draw_static_elements(self):
        """Draw grid, axes, and labels that don't change."""
        # Clear background
        self.img_array[:] = self.bg_color
        
        # Draw grid lines
        self._draw_grid()
        
        # Draw axes
        self._draw_axes()
        
    def _draw_grid(self):
        """Draw grid lines with ticks."""
        # Vertical grid lines
        x_step = (self.x_max - self.x_min) / 10
        for i in range(11):
            x = self.x_min + i * x_step
            screen_x, _ = self._world_to_screen(x, 0)
            if 0 <= screen_x < self.width:
                # Draw vertical grid line
                self.img_array[self.margin_top:self.margin_top + self.plot_height, screen_x] = self.grid_color
                
                # Draw tick marks on x-axis
                y_axis_screen_y = self.margin_top + self.plot_height
                if y_axis_screen_y < self.height:
                    # Draw tick mark (extend below the axis)
                    tick_length = 8
                    for tick_y in range(tick_length):
                        if y_axis_screen_y + tick_y < self.height:
                            self.img_array[y_axis_screen_y + tick_y, screen_x] = self.axis_color
        
        # Horizontal grid lines
        y_step = (self.y_max_flex - self.y_min_flex) / 13
        for i in range(14):
            y = self.y_min_flex + i * y_step
            _, screen_y = self._world_to_screen(0, y)
            if 0 <= screen_y < self.height:
                # Draw horizontal grid line
                self.img_array[screen_y, self.margin_left:self.margin_left + self.plot_width] = self.grid_color
                
                # Draw tick marks on y-axis
                y_axis_screen_x = self.margin_left
                if y_axis_screen_x >= 0:
                    # Draw tick mark (extend left of the axis)
                    tick_length = 8
                    for tick_x in range(tick_length):
                        if y_axis_screen_x - tick_x >= 0:
                            self.img_array[screen_y, y_axis_screen_x - tick_x] = self.axis_color

    def _draw_tick_labels(self, painter):
        """Draw tick labels using QPainter for better text rendering."""
        painter.setPen(QPen(Qt.black))
        painter.setFont(QFont("Arial", 8))
        
        # X-axis tick labels
        x_step = (self.x_max - self.x_min) / 10
        for i in range(11):
            x_value = self.x_min + i * x_step
            screen_x, _ = self._world_to_screen(x_value, 0)
            
            if 0 <= screen_x < self.width:
                # Format the label based on the value range
                if abs(x_value) < 0.01:  # Very close to zero
                    label = "0"
                elif abs(x_value) < 1:
                    label = f"{x_value:.1f}"
                else:
                    label = f"{x_value:.0f}"
                
                # Draw label below the tick
                label_y = self.margin_top + self.plot_height + 25
                painter.drawText(screen_x - 10, label_y, label)
        
        # Y-axis tick labels
        y_step = (self.y_max_flex - self.y_min_flex) / 13
        for i in range(14):
            y_value = self.y_min_flex + i * y_step
            _, screen_y = self._world_to_screen(0, y_value)
            
            if 0 <= screen_y < self.height:
                # Format the label
                if abs(y_value) < 0.01:  # Very close to zero
                    label = "0"
                elif abs(y_value) < 1:
                    label = f"{y_value:.1f}"
                else:
                    label = f"{y_value:.0f}"
                
                # Draw label to the left of the tick
                label_x = self.margin_left - 35
                painter.drawText(label_x, screen_y + 4, label)
                
    def _draw_axes(self):
        """Draw main axes."""
        # Y-axis (flexion)
        screen_x, _ = self._world_to_screen(0, 0)
        if 0 <= screen_x < self.width:
            self.img_array[self.margin_top:self.margin_top + self.plot_height, screen_x:screen_x+2] = self.axis_color
        
    def _draw_line(self, x1, y1, x2, y2, color, thickness=1):
        """Draw a line between two points."""
        screen_x1, screen_y1 = self._world_to_screen(x1, y1)
        screen_x2, screen_y2 = self._world_to_screen(x2, y2)
        
        # Simple line drawing
        dx = abs(screen_x2 - screen_x1)
        dy = abs(screen_y2 - screen_y1)
        
        if dx == 0 and dy == 0:
            return
            
        steps = max(dx, dy)
        if steps == 0:
            return
            
        x_inc = (screen_x2 - screen_x1) / steps
        y_inc = (screen_y2 - screen_y1) / steps
        
        for i in range(steps + 1):
            x = int(screen_x1 + i * x_inc)
            y = int(screen_y1 + i * y_inc)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                for t in range(thickness):
                    for u in range(thickness):
                        if x + t < self.width and y + u < self.height:
                            self.img_array[y + u, x + t] = color
                            
    def _draw_point(self, x, y, color, size = 2):
        """#Draw a simple point
        screen_x, screen_y = self._world_to_screen(x, y)
        
        if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
            self.img_array[screen_y, screen_x] = color"""
        """Draw a simple point - single pixel or 3x3 square."""
        screen_x, screen_y = self._world_to_screen(x, y)
        
        if size == 1:
            # Single pixel
            if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                self.img_array[screen_y, screen_x] = color
        else:
            # 4x4 square around the pixel
            for dy in range(-2, 3):  # -1, 0, 1
                for dx in range(-2, 3):  # -1, 0, 1
                    px, py = screen_x + dx, screen_y + dy
                    if 0 <= px < self.width and 0 <= py < self.height:
                        self.img_array[py, px] = color
    
    def add_point(self, flexion_angle, var_val_displacement, rotation_angle, adduction_angle, anterior_translation, medial_translation, mode):
        """Add a new data point to the circular buffer."""
        self.flexion_data[self.write_idx] = flexion_angle
        self.varus_valgus_data[self.write_idx] = var_val_displacement
        self.rotation_data[self.write_idx] = rotation_angle
        self.adduction_data[self.write_idx] = adduction_angle
        self.anterior_data[self.write_idx] = anterior_translation
        self.medial_data[self.write_idx] = medial_translation
        
        # Encode mode
        if mode == "varus_valgus":
            self.mode_data[self.write_idx] = 0
        elif mode == "rotation":
            self.mode_data[self.write_idx] = 1
        elif mode == "adduction":
            self.mode_data[self.write_idx] = 2
        elif mode == "anterior":
            self.mode_data[self.write_idx] = 3
        elif mode == "medial":
            self.mode_data[self.write_idx] = 4
            
        self.write_idx = (self.write_idx + 1) % self.max_points
        self.point_count = min(self.point_count + 1, self.max_points)
        
        
    def update_varus_valgus_plot(self, flexion_angle, var_val_displacement, rotation_angle, adduction_angle, anterior_translation, medial_translation, mode, mode_points):
        """Main update method - optimized for real-time performance."""
        # Update mode if changed
        if mode != self.current_mode or mode_points != self.current_point_mode:
            self.current_mode = mode
            self.current_point_mode = mode_points
            self._update_transform_params()
            self._draw_static_elements()
        
        # Add new data point
        self.add_point(flexion_angle, var_val_displacement, rotation_angle, adduction_angle, anterior_translation, medial_translation, mode)
        
        # Render frame
        self._render_frame()
        
        # Trigger Qt repaint
        self.update()
    
    def _apply_angle_filter(self, flexion, vv, rotation, adduction, anterior, medial, modes):
        """Apply angle-based filtering based on current mode."""
        if self.current_mode == "varus_valgus":
            current_mode_val = 0  
        elif self.current_mode == "rotation":
            current_mode_val = 1
        elif self.current_mode == "adduction":
            current_mode_val = 2
        elif self.current_mode == "anterior":
            current_mode_val = 3
        elif self.current_mode == "medial":
            current_mode_val = 4
        
        # Start with mode filter
        mode_mask = modes == current_mode_val
        
        if self.current_mode == "rotation":
            # For rotation mode: filter where -2.5 < adduction < 2.5
            angle_mask = (adduction >= -constants.RANGE_FILTER_PLOT) & (adduction <= constants.RANGE_FILTER_PLOT)
        elif self.current_mode == "varus_valgus" or self.current_mode == "adduction":  # varus_valgus or adduction mode
            # For varus_valgus or adduction mode: filter where -2.5 < rotation < 2.5
            angle_mask = (rotation >= -constants.RANGE_FILTER_PLOT) & (rotation <= constants.RANGE_FILTER_PLOT)
        elif self.current_mode == "anterior":
            angle_mask = ( abs(medial) <= constants.RANGE_FILTER_PLOT_TRANSLATION)
        elif self.current_mode == "medial":
            angle_mask = ( abs(anterior) <= constants.RANGE_FILTER_PLOT_TRANSLATION)
        else:
            # Default case - no additional filtering beyond mode
            angle_mask = np.ones(len(flexion), dtype=bool)
            print("attention no values filtered")
        
        # Combine both filters
        combined_mask = mode_mask & angle_mask
        
        return (flexion[combined_mask], 
                vv[combined_mask], 
                rotation[combined_mask], 
                adduction[combined_mask],
                anterior[combined_mask],
                medial[combined_mask])
        #print("DEBUG: Filtering disabled - returning all data")
        #return flexion, vv, rotation, adduction, anterior, medial
        
        
    def _render_frame(self):
        """Render the current frame with all data points."""
        # Redraw static elements
        self._draw_static_elements()
        
        if self.point_count == 0:
            return
            
        # Get active data
        if self.point_count < self.max_points:
            flexion = self.flexion_data[:self.point_count]
            vv = self.varus_valgus_data[:self.point_count]
            rotation = self.rotation_data[:self.point_count]
            adduction = self.adduction_data[:self.point_count]
            modes = self.mode_data[:self.point_count]
            anterior = self.anterior_data[:self.point_count]
            medial = self.medial_data[:self.point_count]
        else:
            # Handle circular buffer
            flexion = np.concatenate([self.flexion_data[self.write_idx:], self.flexion_data[:self.write_idx]])
            vv = np.concatenate([self.varus_valgus_data[self.write_idx:], self.varus_valgus_data[:self.write_idx]])
            rotation = np.concatenate([self.rotation_data[self.write_idx:], self.rotation_data[:self.write_idx]])
            adduction = np.concatenate([self.adduction_data[self.write_idx:], self.adduction_data[:self.write_idx]])
            modes = np.concatenate([self.mode_data[self.write_idx:], self.mode_data[:self.write_idx]])
            anterior = np.concatenate([self.anterior_data[self.write_idx:], self.anterior_data[:self.write_idx]])
            medial = np.concatenate([self.medial_data[self.write_idx:], self.medial_data[:self.write_idx]])
        
        # Apply filtering
        flex_filtered, vv_filtered, rotation_filtered, adduction_filtered, anterior_filtered, medial_filtered = self._apply_angle_filter(
            flexion, vv, rotation, adduction, anterior, medial, modes)
        
        
        if len(flex_filtered) == 0:
            return
        
        # Determine which angle to plot based on current mode
        if self.current_mode == "rotation":
            angle_to_plot = rotation_filtered
        elif self.current_mode == "adduction":
            angle_to_plot = adduction_filtered
        elif self.current_mode == "varus_valgus":  # varus_valgus
            angle_to_plot = vv_filtered
        elif self.current_mode == "anterior":
            angle_to_plot = anterior_filtered
        elif self.current_mode == "medial":
            angle_to_plot = medial_filtered
            
        # Draw points based on mode
        if self.current_point_mode == "bars":
            self._draw_bars(flex_filtered, angle_to_plot)
        else:
            self._draw_points(flex_filtered, angle_to_plot)
            
        # Draw current point (most recent)
        if len(flex_filtered) > 0:
            latest_flex = flex_filtered[-1]
            latest_angle = angle_to_plot[-1]
            self._draw_point(latest_angle, latest_flex, self.current_point_color, size=3)
    
    def _draw_bars(self, flexion, vv):
        #Draw horizontal bars from 0 to displacement value.
        for i in range(len(flexion)):
            color = self.salmon_color if vv[i] > 0 else self.limegreen_color
            self._draw_line(0, flexion[i], vv[i], flexion[i], color, thickness=2)
            
    def _draw_points(self, flexion, vv):
        """Draw simple scatter points."""
        for i in range(len(flexion)):
            color = self.salmon_color if vv[i] > 0 else self.limegreen_color
            self._draw_point(vv[i], flexion[i], color, size=3)  # size=3 for 3x3 squares
    
    def paintEvent(self, event):
        """PyQt paint event - renders the image to the widget."""
        painter = QPainter(self)
        
        # Convert numpy array to QImage
        h, w, ch = self.img_array.shape
        bytes_per_line = ch * w
        qimage = QImage(self.img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Draw the image
        painter.drawImage(0, 0, qimage)
        
        # Draw text labels using QPainter (better performance than drawing on image)
        self._draw_labels(painter)
        
    def _draw_labels(self, painter):
        """Draw axis labels and title using QPainter."""
        painter.setPen(QPen(Qt.black))
        painter.setFont(QFont("Arial", 10))
        
        # Draw tick labels first
        self._draw_tick_labels(painter)
        
        # Y-axis label (rotated)
        painter.save()
        painter.translate(20, self.height//2)
        painter.rotate(-90)
        painter.drawText(0, 0, "Flexion Angle [°]")
        painter.restore()
        
        # X-axis label
        if self.current_mode == "varus_valgus":
            label = "medial joint gap          lateral joint gap"
        elif self.current_mode == "rotation":
            label = "external rotation         internal rotation"
        elif self.current_mode == "adduction":
            label = "varus angle        valgus angle"
        elif self.current_mode == "anterior":
            label = "anterior translation        posterior translation"
        elif self.current_mode == "medial":
            label = "medial translation        lateral translation"    
        else:
            label = "x-axis"
            
        painter.drawText(self.width//2 - 100, self.height - 20, label)
        
        # Title
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        if self.current_mode == "varus_valgus":
            title = "medial/lateral joint gap [mm]"
        else:
            title = f"{self.current_mode} [°]"
            
        painter.drawText(self.width//2 - 100, 20, title)
    def clear_data(self):
        """Clear all plot data and reset the circular buffer."""
        # Reset all data arrays to zero
        self.flexion_data.fill(0)
        self.varus_valgus_data.fill(0)
        self.rotation_data.fill(0)
        self.adduction_data.fill(0)
        self.mode_data.fill(0)
        
        # Reset buffer pointers and counters
        self.write_idx = 0
        self.point_count = 0
        
        # Redraw static elements (clears the plot visually)
        self._draw_static_elements()
        
        # Trigger Qt repaint to show the cleared plot
        self.update()
        
        print("Plot data cleared")



