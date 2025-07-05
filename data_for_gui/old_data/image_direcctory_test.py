import os
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap

# For debugging purposes
def debug_image_loading():
    print("="*50)
    print("DEBUGGING IMAGE LOADING")
    print("="*50)
    
    # Current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # List files in current directory
    print(f"Files in current directory: {os.listdir(cwd)}")
    
    # Check if data_for_gui exists
    data_dir = os.path.join(cwd, "data_for_gui")
    print(f"data_for_gui full path: {data_dir}")
    print(f"data_for_gui exists: {os.path.exists(data_dir)}")
    
    if os.path.exists(data_dir):
        print(f"Files in data_for_gui: {os.listdir(data_dir)}")
    
    # Try different path formats
    angle = 0  # Replace with your self.current_angle value
    
    paths_to_try = [
        f"data_for_gui/KW{angle}.jpg",                      # Relative path
        f"./data_for_gui/KW{angle}.jpg",                    # Explicit relative path
        os.path.join(cwd, f"data_for_gui/KW{angle}.jpg"),   # Full path using os.path.join
        os.path.abspath(f"data_for_gui/KW{angle}.jpg")      # Absolute path
    ]
    
    for i, path in enumerate(paths_to_try):
        print(f"\nPath {i+1}: {path}")
        print(f"Path exists: {os.path.exists(path)}")

# Run the debug function
debug_image_loading()

# Now try to load an image with Qt
app = QApplication(sys.argv)
window = QWidget()
layout = QVBoxLayout()
window.setLayout(layout)

angle = 0  # Replace with your self.current_angle value
paths_to_try = [
    f"data_for_gui/KW{angle}.jpg",
    f"./data_for_gui/KW{angle}.jpg",
    os.path.join(os.getcwd(), f"data_for_gui/KW{angle}.jpg"),
    os.path.abspath(f"data_for_gui/KW{angle}.jpg"),
    # Try without the KW prefix or with different case
    f"data_for_gui/{angle}.jpg",
    f"data_for_gui/kw{angle}.jpg"
]

for i, path in enumerate(paths_to_try):
    label = QLabel(f"Path {i+1}: {path}")
    layout.addWidget(label)
    
    pixmap = QPixmap(path)
    if not pixmap.isNull():
        print(f"Successfully loaded image from: {path}")
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label)
    else:
        print(f"Failed to load image from: {path}")

window.show()
sys.exit(app.exec_())