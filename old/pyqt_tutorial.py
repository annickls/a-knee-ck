import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setGeometry(100, 100, 800, 600)
    window.setWindowTitle("Test Window")
    label = QLabel("Hello, World!", window)
    label.move(300, 250)
    window.show()
    sys.exit(app.exec_())