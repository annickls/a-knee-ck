from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph.opengl as gl
from OpenGL.GL import glBegin, glEnd, glVertex3f, glColor4f, GL_LINES, GL_LINE_SMOOTH, glEnable, glHint, GL_LINE_SMOOTH_HINT, GL_NICEST
import pyqtgraph.opengl as gl
import constants
import numpy as np

class MplCanvas(FigureCanvas):
    """Matplotlib canvas class for embedding plots in Qt that can display either current or historical force/torque data"""
    def __init__(self, width=5, height=4, mode="current"):
        self.fig = Figure(figsize=(width, height))
        self.axes_force = self.fig.add_subplot(121, projection='3d')
        self.axes_torque = self.fig.add_subplot(122, projection='3d')
        self.mode = mode  # "current" or "history"
        
        # Set up axes once
        force_max = constants.FORCE_MAX #if mode == "current" else 11
        torque_max = constants.TORQUE_MAX
        
        # Force plot setup
        self.axes_force.set_xlim([-force_max, force_max])
        self.axes_force.set_ylim([-force_max, force_max])
        self.axes_force.set_zlim([-force_max, force_max])
        #self.axes_force.set_title('Current Force (N)' if mode == "current" else 'Force History (N)')
        self.axes_force.grid(False)
        self.axes_force.set_axis_off()
        
        # Torque plot setup
        self.axes_torque.set_xlim([-torque_max, torque_max])
        self.axes_torque.set_ylim([-torque_max, torque_max])
        self.axes_torque.set_zlim([-torque_max, torque_max])
        #self.axes_torque.set_title('Current Torque (Nm)' if mode == "current" else 'Torque History (Nm)')
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
        if mode == "history":
            self.force_arrows = []
            self.torque_arrows = []
        
        # Text elements for magnitudes and components
        self.force_mag_text = self.axes_force.text2D(0.32, 1.0, "", transform=self.axes_force.transAxes)
        self.torque_mag_text = self.axes_torque.text2D(0.4, 1.0, "", transform=self.axes_torque.transAxes)
        self.force_comp_text = self.axes_force.text2D(0.32, 0.95, "", transform=self.axes_force.transAxes, fontsize=8)
        self.torque_comp_text = self.axes_torque.text2D(0.4, 0.95, "", transform=self.axes_torque.transAxes, fontsize=8)
        
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()


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