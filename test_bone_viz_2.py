import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget, QMainWindow
import pyqtgraph.opengl as gl
from stl import mesh

def load_stl_as_mesh(filename):
    stl_mesh = mesh.Mesh.from_file(filename)
    vertices = np.array(stl_mesh.vectors).reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    return vertices, faces

def apply_transform(vertices, translation=(0, 0, 0), rotation=(0, 0, 0)):
    tx, ty, tz = translation
    rx, ry, rz = np.radians(rotation)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    transformed = (vertices @ R.T) + np.array([tx, ty, tz])
    return transformed

class STLTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=200)
        layout.addWidget(self.view)
        self.setLayout(layout)

        # Grid
        grid = gl.GLGridItem()
        grid.scale(10, 10, 1)
        self.view.addItem(grid)

        # Load STL 1
        vertices1, faces1 = load_stl_as_mesh('femur.stl')
        transformed1 = apply_transform(vertices1, translation=(0, 0, 0), rotation=(0, 0, 0))
        mesh1 = gl.GLMeshItem(vertexes=transformed1, faces=faces1, color=(1, 0, 0, 0.7), drawEdges=True)
        self.view.addItem(mesh1)

        # Load STL 2
        vertices2, faces2 = load_stl_as_mesh('tibia.stl')
        transformed2 = apply_transform(vertices2, translation=(0, 0, 0), rotation=(0, 90, 0))
        mesh2 = gl.GLMeshItem(vertexes=transformed2, faces=faces2, color=(0, 0, 1, 0.7), drawEdges=True)
        self.view.addItem(mesh2)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My GUI with STL Viewer Tab")
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # Add STL viewer tab
        stl_tab = STLTab()
        tabs.addTab(stl_tab, "3D Models")

        # Add more tabs if needed
        # tabs.addTab(QWidget(), "Settings")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
