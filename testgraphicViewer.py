import sys
import pyqtgraph.opengl as gl
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                             QWidget, QGraphicsView, QGraphicsScene, 
                             QGraphicsProxyWidget, QPushButton)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rotated GLViewWidget")
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create graphics view and scene
        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        
        # Create GLViewWidget
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setFixedSize(600, 400)
        
        # Add some 3D content for testing
        self.add_test_content()
        
        # Wrap GLViewWidget in proxy and rotate
        self.proxy = QGraphicsProxyWidget()
        self.proxy.setWidget(self.gl_view)
        self.proxy.setRotation(90)  # 90 degrees rotation
        
        self.scene.addItem(self.proxy)
        
        # Add rotation control button
        rotate_btn = QPushButton("Toggle Rotation (0°/90°)")
        rotate_btn.clicked.connect(self.toggle_rotation)
        
        layout.addWidget(self.graphics_view)
        layout.addWidget(rotate_btn)
        
        self.current_rotation = 90
    
    def add_test_content(self):
        """Add some test 3D objects to the GLViewWidget"""
        # Create a simple mesh (cube)
        vertices = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ])
        faces = np.array([
            [0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7],
            [0, 1, 4], [1, 4, 5], [2, 3, 6], [3, 6, 7],
            [0, 2, 4], [2, 4, 6], [1, 3, 5], [3, 5, 7]
        ])
        
        mesh = gl.GLMeshItem(vertexes=vertices, faces=faces, 
                           faceColors=[[1, 0, 0, 0.5]] * 12)
        self.gl_view.addItem(mesh)
        
        # Add coordinate axes
        axis = gl.GLAxisItem()
        self.gl_view.addItem(axis)
    
    def toggle_rotation(self):
        """Toggle between 0° and 90° rotation"""
        if self.current_rotation == 90:
            self.proxy.setRotation(0)
            self.current_rotation = 0
        else:
            self.proxy.setRotation(90)
            self.current_rotation = 90

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())