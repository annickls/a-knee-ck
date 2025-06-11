from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph.opengl as gl
from OpenGL.GL import glBegin, glEnd, glVertex3f, glColor4f, GL_LINES, GL_LINE_SMOOTH, glEnable, glHint, GL_LINE_SMOOTH_HINT, GL_NICEST
import constants
import numpy as np

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

        if mode == "varus_valgus":
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel('Varus/Valgus Displacement')
            self.ax.set_ylabel('Flexion Angle (degrees)')
            self.ax.set_title('Real-time Flexion vs Varus/Valgus')
            self.ax.grid(True, alpha=0.3)

            self.fig.subplots_adjust(left=0.15, bottom=0.15, right = 0.95, top =0.90)
            
            # Set initial axis limits
            self.ax.set_xlim(-20, 20)  # Adjust range as needed
            self.ax.set_ylim(0, 120)   # Adjust range as needed
            
            # Add vertical line at x=0 for reference
            self.ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            # Initialize empty line for real-time plotting
            self.line, = self.ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
            self.current_point, = self.ax.plot([], [], 'ro', markersize=8)
            
            # Store data for plotting
            self.varus_valgus_data = []
            self.flexion_data = []
    
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

    def update_varus_valgus_plot(self, flexion_angle, var_val_displacement):
        """Update the varus/valgus vs flexion plot by adding only the newest data point"""
        """if self.mode == "varus_valgus":
            # Add new data point
            self.varus_valgus_data.append(var_val_displacement)
            self.flexion_data.append(flexion_angle)
            
            # Keep only last N points for performance (adjust as needed)
             max_points = 1000
            if len(self.varus_valgus_data) > max_points:
                self.varus_valgus_data = self.varus_valgus_data[-max_points:]
                self.flexion_data = self.flexion_data[-max_points:]
                # If we hit the limit, we need to redraw everything
                self._redraw_full_plot(flexion_angle, var_val_displacement)
                return
            
            # For the first point, setup the plot
            if len(self.varus_valgus_data) == 1:
                self.ax.clear()
                self.ax.set_xlabel('Varus/Valgus Displacement')
                self.ax.set_ylabel('Flexion Angle (degrees)')
                self.ax.set_title('Real-time Flexion vs Varus/Valgus')
                self.ax.grid(True, alpha=0.3)
                self.ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                self.ax.set_xlim(-50, 50)
                self.ax.set_ylim(-10, 120)"""

# Add new data point
        self.varus_valgus_data.append(var_val_displacement)
        self.flexion_data.append(flexion_angle)
        
        # Keep only last N points for performance (adjust as needed)
        """ max_points = 1000
        if len(self.varus_valgus_data) > max_points:
            self.varus_valgus_data = self.varus_valgus_data[-max_points:]
            self.flexion_data = self.flexion_data[-max_points:]
            # If we hit the limit, we need to redraw everything
            self._redraw_full_plot(flexion_angle, var_val_displacement)
            return"""
        
        # For the first point, setup the plot
        if len(self.varus_valgus_data) == 1:
            self.ax.clear()
            self.ax.set_xlabel('Varus/Valgus Displacement')
            self.ax.set_ylabel('Flexion Angle (degrees)')
            self.ax.set_title('Real-time Flexion vs Varus/Valgus')
            self.ax.grid(True, alpha=0.3)
            self.ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            self.ax.set_xlim(-50, 50)
            self.ax.set_ylim(-10, 120)
            
            # Only add the newest data point
            color = constants.SALMON if var_val_displacement > 0 else constants.LIMEGREEN
            
            # Draw horizontal line from 0 to displacement value for new point
            line, = self.ax.plot([0, var_val_displacement], [flexion_angle, flexion_angle], 
                                color=color, linewidth=2, alpha=0.7)
            
            # Add current point (red dot) - remove previous current point if it exists
            if hasattr(self, 'current_point') and self.current_point:
                self.current_point.remove()
            
            self.current_point, = self.ax.plot(var_val_displacement, flexion_angle, 'ro', 
                                            markersize=8, zorder=10)
            
            # Add current bar highlight - remove previous highlight if it exists
            if hasattr(self, 'current_highlight') and self.current_highlight:
                self.current_highlight.remove()
                
            current_color = 'red' if var_val_displacement > 0 else 'blue'
            self.current_highlight, = self.ax.plot([0, var_val_displacement], [flexion_angle, flexion_angle], 
                                                color=current_color, linewidth=4, alpha=0.8, zorder=5)
            
            self.draw()
    
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






