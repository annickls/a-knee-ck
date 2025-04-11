import sys
from PyQt5.QtWidgets import QApplication, QWidget, QTabWidget, QVBoxLayout, QLabel

class TabExample(QWidget):
    def __init__(self):
        super().__init__()

        # Create the QTabWidget
        self.tabs = QTabWidget()

        # Create tab pages (as QWidget)
        self.tab1 = QWidget()
        self.tab2 = QWidget()

        # Add tabs to the tab widget
        self.tabs.addTab(self.tab1, "Tab 1")
        self.tabs.addTab(self.tab2, "Tab 2")

        # Add content to the first tab
        tab1_layout = QVBoxLayout()
        tab1_layout.addWidget(QLabel("This is Tab 1"))
        self.tab1.setLayout(tab1_layout)

        # Add content to the second tab
        tab2_layout = QVBoxLayout()
        tab2_layout.addWidget(QLabel("This is Tab 2"))
        self.tab2.setLayout(tab2_layout)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

        self.setWindowTitle("QTabWidget Example")
        self.resize(400, 300)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TabExample()
    window.show()
    sys.exit(app.exec_())
