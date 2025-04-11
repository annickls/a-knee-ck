import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFrame, 
                            QProgressBar, QGridLayout)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer

class KneeFlexionExperiment(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Configuration
        self.setWindowTitle("Knee Test Bench")
        self.setGeometry(100, 100, 800, 600)
        
        # Experiment parameters
        self.flexion_angles = [0, 30, 60, 90, 120]
        self.current_angle_index = 0
        self.rotation_time = 5  # seconds
        self.timer = QTimer()
        self.timer.timeout.connect(self.rotation_complete)
        self.seconds_timer = QTimer()
        self.seconds_timer.timeout.connect(self.update_seconds_progress)
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        #main_layout = QVBoxLayout()
        main_layout = QGridLayout()
        #main_second_layout = QHBoxLayout()
        
        # Instruction label
        self.instruction_label = QLabel("Knee Test Bench")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(self.instruction_label, 0, 0, 1, 2)
        
        # Rotation timer progress bar
        rotation_progress_layout = QVBoxLayout()
        rotation_progress_label = QLabel("Please Flex the knee to the desired flexion angle, then hold the desired positions for the shown amount of time")
        rotation_progress_layout.addWidget(rotation_progress_label)
        
        self.rotation_progress = QProgressBar()
        self.rotation_progress.setRange(0, self.rotation_time)
        self.rotation_progress.setValue(self.rotation_time)
        self.rotation_progress.setTextVisible(True)
        self.rotation_progress.setFormat("%v seconds remaining")
        rotation_progress_layout.addWidget(self.rotation_progress)
        #main_layout.addLayout(rotation_progress_layout)
        main_layout.addLayout(rotation_progress_layout, 1, 0, 1, 2)
        
        # Image display - smaller size
        self.image_frame = QFrame()
        #self.image_frame.setFrameShape(QFrame.Box)
        self.image_frame.setLineWidth(2)
        self.image_frame.setFixedSize(400, 300)
        
        image_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label)
        self.image_frame.setLayout(image_layout)
        
        # main_layout.addWidget(self.image_frame, alignment=Qt.AlignLeft)
        main_layout.addWidget(self.image_frame, 2,0)
        # main_second_layout.addWidget(self.image_frame, alignment=Qt.AlignLeft)
        
        # Control buttons
        button_layout = QVBoxLayout()
        
        
        self.start_button = QPushButton("Start Experiment")
        self.start_button.clicked.connect(self.start_experiment)
        

        self.next_button = QPushButton("Next Angle")
        #self.instruction_label.setText(f"Please flex knee to {current_angle} degrees")
        self.next_button.clicked.connect(self.next_angle)
        self.next_button.setEnabled(False)

        current_angle = self.flexion_angles[self.current_angle_index]
        self.next_label = QLabel("test1")
        font = self.next_label.font()
        font.setPointSize(12)
        self.next_label.setFont(font)
        self.next_label.setAlignment(Qt.AlignTop)
        

        self.rotate_button = QPushButton("Record Data Flexion")
        self.rotate_button.clicked.connect(self.start_rotation)
        self.rotate_button.setEnabled(False)

        self.rotate_label = QLabel("Hold Position for 5s")
        font = self.rotate_label.font()
        font.setPointSize(12)
        self.rotate_label.setFont(font)
        self.rotate_label.setAlignment(Qt.AlignTop)


        self.varus_button = QPushButton("Record Data Varus")
        self.varus_button.clicked.connect(self.start_varus)
        self.varus_button.setEnabled(False)

        self.varus_label = QLabel("Apply a Varus load of 15N for 5s")
        font = self.varus_label.font()
        font.setPointSize(12)
        self.varus_label.setFont(font)
        self.varus_label.setAlignment(Qt.AlignTop)

        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.next_label)
        button_layout.addWidget(self.rotate_button)
        button_layout.addWidget(self.rotate_label)
        button_layout.addWidget(self.varus_button)
        button_layout.addWidget(self.varus_label)
        
        # main_layout.addLayout(button_layout)
        main_layout.addLayout(button_layout, 2, 1)
        #main_second_layout.addLayout(button_layout)
        #main_layout.addLayout( main_second_layout)

        
        # Overall progress bar - moved to bottom with green color
        overall_progress_layout = QVBoxLayout()
        overall_progress_label = QLabel("Overall Experiment Progress:")
        overall_progress_layout.addWidget(overall_progress_label)
        
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, len(self.flexion_angles))
        self.overall_progress.setValue(0)
        self.overall_progress.setTextVisible(True)
        self.overall_progress.setFormat("%v/%m angles completed")
        
        # Set green color for the overall progress bar
        self.overall_progress.setStyleSheet("QProgressBar {border: 1px solid grey; border-radius: 3px; text-align: center;}"
                                           "QProgressBar::chunk {background-color: #4CAF50; width: 10px;}")
        
        overall_progress_layout.addWidget(self.overall_progress)
        #main_layout.addLayout(overall_progress_layout)
        main_layout.addLayout(overall_progress_layout, 3,0, 1, 2)
        
        # Set main layout
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Initial display
        self.update_display()
    
    def update_display(self):
        if self.current_angle_index < len(self.flexion_angles):
            current_angle = self.flexion_angles[self.current_angle_index]
            #self.instruction_label.setText(f"Please flex knee to {current_angle} degrees")
            self.next_label.setText(f"Please flex knee to {current_angle} degrees")
            
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
        else:
            self.instruction_label.setText("Experiment Complete!")
            self.image_label.clear()
            self.next_button.setEnabled(False)
            self.rotate_button.setEnabled(False)
            self.varus_button.setEnabled(False)
            self.start_button.setEnabled(True)
            self.overall_progress.setValue(len(self.flexion_angles))
            self.current_angle_index += 1
            self.update_display()
    
    def start_experiment(self):
        self.current_angle_index = 0
        self.overall_progress.setValue(0)
        self.update_display()
        self.start_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.rotate_button.setEnabled(True)
    
    def next_angle(self):
        self.current_angle_index += 1
        self.update_display()
        self.next_button.setEnabled(False)
        self.rotate_button.setEnabled(True)
    
    def start_rotation(self):
        self.rotate_button.setEnabled(False)
        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.varus_button.setEnabled(True)

    def start_varus(self):
        self.varus_button.setEnabled(False)
        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
    
    def update_seconds_progress(self):
        self.remaining_time -= 1
        self.rotation_progress.setValue(self.remaining_time)
        
        if self.remaining_time <= 0:
            self.seconds_timer.stop()
            self.rotation_complete()
    
    def rotation_complete(self):
        self.timer.stop()
        self.seconds_timer.stop()
        self.rotation_progress.setValue(0)
        
        if self.current_angle_index < len(self.flexion_angles) - 1:
            self.next_button.setEnabled(True)
        else:
            self.instruction_label.setText("Experiment Complete!")
            self.start_button.setEnabled(True)
            self.current_angle_index += 1
            self.update_display()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KneeFlexionExperiment()
    window.show()
    sys.exit(app.exec_())