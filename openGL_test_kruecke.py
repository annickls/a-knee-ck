import sys
import os



# Now import the libraries
from PyQt5.QtWidgets import QApplication
import pyqtgraph.opengl as gl
import numpy as np

# Create the application
app = QApplication([])

# Create the OpenGL widget with a modified configuration
view = gl.GLViewWidget()
view.opts['distance'] = 40  # Increase view distance for better visibility
view.show()

# Add a grid
grid = gl.GLGridItem()
grid.setSize(x=10, y=10, z=10)
grid.setSpacing(x=1, y=1, z=1)
view.addItem(grid)

# Add a simple line plot
pts = np.array([[0, 0, 0], [5, 5, 5]])
line = gl.GLLinePlotItem(pos=pts, color=(1, 0, 0, 1), width=2)
view.addItem(line)

# Add a simple mesh (a plane)
vertices = np.array([
    [0, 0, 0],
    [5, 0, 0],
    [5, 5, 0],
    [0, 5, 0]
])
faces = np.array([
    [0, 1, 2],
    [0, 2, 3]
])
mesh_data = gl.MeshData(vertexes=vertices, faces=faces)
mesh = gl.GLMeshItem(meshdata=mesh_data, smooth=False, color=(0, 1, 0, 0.5), shader='shaded')
view.addItem(mesh)

# Execute the application
if __name__ == '__main__':
    if (sys.flags.interactive != 1):
        app.exec_()