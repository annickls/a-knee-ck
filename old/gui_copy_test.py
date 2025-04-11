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
        main_layout = QGridLayout()
        
        # Instruction label
        self.instruction_label = QLabel("Knee Test Bench")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(self.instruction_label, 0, 0, 1, 2)
        
        # Rotation timer progress bar
        rotation_progress_layout = QVBoxLayout()
        rotation_progress_label = QLabel("Please Flex the knee to the desired flexion angle, then hold the desired positions for the shown amount of time")
        rotation_progress_label.setAlignment(Qt.AlignCenter)
        rotation_progress_layout.addWidget(rotation_progress_label)
        
        self.rotation_progress = QProgressBar()
        self.rotation_progress.setRange(0, self.rotation_time)
        self.rotation_progress.setValue(self.rotation_time)
        self.rotation_progress.setTextVisible(True)
        self.rotation_progress.setFormat("%v seconds remaining")
        rotation_progress_layout.addWidget(self.rotation_progress)
        main_layout.addLayout(rotation_progress_layout, 1, 0, 1, 2)
        
        # Image display - smaller size
        self.image_frame = QFrame()
        self.image_frame.setLineWidth(2)
        self.image_frame.setFixedSize(400, 300)
        
        image_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label)
        self.image_frame.setLayout(image_layout)
        
        main_layout.addWidget(self.image_frame, 2,0)
        
        # Control buttons
        sub_layout = QGridLayout()
        
        # Start Experiment Button
        self.start_button = QPushButton("Start Experiment")
        self.start_button.clicked.connect(self.start_experiment)
        
        # Next Angle Button
        self.next_button = QPushButton("Next Angle")
        self.next_button.clicked.connect(self.next_angle)
        self.next_button.setEnabled(False)

        # Next Angle Label
        self.next_label = QLabel("test1")
        font = self.next_label.font()
        font.setPointSize(12)
        self.next_label.setFont(font)
        self.next_label.setAlignment(Qt.AlignTop)
        
        # Rotate Button
        self.rotate_button = QPushButton("Hold Flexion for 5 s")
        self.rotate_button.clicked.connect(self.start_rotation)
        self.rotate_button.setEnabled(False)

        # Rotate Label
        #self.rotate_label = QLabel("Hold Position for 5s")
        #font = self.rotate_label.font()
        #font.setPointSize(12)
        #self.rotate_label.setFont(font)
        #self.rotate_label.setAlignment(Qt.AlignTop)
        #self.rotate_label.hide()

        # Varus Button
        self.varus_button = QPushButton("Apply Varus Load for 5 s")
        self.varus_button.clicked.connect(self.start_varus)
        self.varus_button.setEnabled(False)

        # Varus Label
        #self.varus_label = QLabel("Apply a Varus load of 15N for 5s")
        #font = self.varus_label.font()
        #font.setPointSize(12)
        #self.varus_label.setFont(font)
        #self.varus_label.setAlignment(Qt.AlignTop)
        #self.varus_label.hide()

        # Valgus Button
        self.valgus_button = QPushButton("Apply Valgus Load for 5 s")
        self.valgus_button.clicked.connect(self.start_valgus)
        self.valgus_button.setEnabled(False)

        # IR Button
        self.internal_rot_button = QPushButton("Rotate Tibia internally for 5 s")
        self.internal_rot_button.clicked.connect(self.start_internal_rot)
        self.internal_rot_button.setEnabled(False)

        # ER Button
        self.external_rot_button = QPushButton("Rotate Tibia externally for 5 s")
        self.external_rot_button.clicked.connect(self.start_external_rot)
        self.external_rot_button.setEnabled(False)



        record_data_label = QLabel("Record Data")
        record_data_label.setAlignment(Qt.AlignCenter)
        
        # Layout arrangement
        subsub_layout = QHBoxLayout()
        subsub_layout.addWidget(self.start_button)
        subsub_layout.addWidget(self.next_button)

        sub_layout.addLayout(subsub_layout, 0, 0)
        sub_layout.addWidget(self.next_label, 0, 1)
        sub_layout.addWidget(record_data_label, 1, 0, 1, 1)
        sub_layout.addWidget(self.rotate_button, 2, 0)
        #sub_layout.addWidget(self.rotate_label, 2, 1)
        #self.rotate_label.show()
        #self.varus_label.hide()
        sub_layout.addWidget(self.varus_button, 3, 0)
        #sub_layout.addWidget(self.varus_label, 3, 1)
        #self.rotate_label.hide()
       # self.varus_label.show()

        sub_layout.addWidget(self.valgus_button, 4, 0)
        sub_layout.addWidget(self.internal_rot_button, 5, 0)
        sub_layout.addWidget(self.external_rot_button, 6, 0)


        
        main_layout.addLayout(sub_layout, 2, 1)
        
        # Overall progress bar
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
        main_layout.addLayout(overall_progress_layout, 3,0, 1, 2)
        
        # Set main layout
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Initial display
        self.update_display()
    
    def update_display(self):
        if self.current_angle_index < len(self.flexion_angles):
            current_angle = self.flexion_angles[self.current_angle_index]
            self.next_label.setText(f"Please flex knee to {current_angle} degrees")
            self.next_label.setAlignment(Qt.AlignCenter)
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
            self.reset_buttons_and_labels()
            self.current_angle_index += 1
    
    def reset_buttons_and_labels(self):
        # Disable all buttons
        self.start_button.setEnabled(True)
        self.next_button.setEnabled(False)
        self.rotate_button.setEnabled(False)
        self.varus_button.setEnabled(False)
        
        # Hide all labels
        #self.next_label.hide()
        #self.rotate_label.hide()
        #self.varus_label.hide()
    
    def start_experiment(self):
        self.current_angle_index = 0
        self.overall_progress.setValue(0)
        self.update_display()
        
        # Enable only start button
        self.start_button.setEnabled(False)
        self.next_label.show()
        self.next_button.setEnabled(False)
        self.rotate_button.setEnabled(True)
        #self.rotate_label.show()
    
    def next_angle(self):
        self.current_angle_index += 1
        self.update_display()
        
        # Reset button states
        self.next_button.setEnabled(False)
        #self.next_label.hide()
        self.rotate_button.setEnabled(True)
        #self.rotate_label.show()

    def start_rotation(self):
        # Disable rotate button
        self.rotate_button.setEnabled(False)
        
        # Enable varus button
        self.varus_button.setEnabled(True)
        
        # Hide rotate label, show varus label
        #self.rotate_label.show()
        #self.varus_label.hide()

        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.next_button.setEnabled(False)
        
    def start_varus(self):
        # Disable varus button
        self.varus_button.setEnabled(False)

        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.valgus_button.setEnabled(True)

    def start_valgus(self):
        # Disable varus button
        self.valgus_button.setEnabled(False)

        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.internal_rot_button.setEnabled(True)

    def start_internal_rot(self):
        # Disable varus button
        self.internal_rot_button.setEnabled(False)

        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.external_rot_button.setEnabled(True)

    def start_external_rot(self):
        # Disable varus button
        self.external_rot_button.setEnabled(False)

        self.remaining_time = self.rotation_time
        self.rotation_progress.setValue(self.remaining_time)
        self.seconds_timer.start(1000)  # Update every second
        self.next_button.setEnabled(True)



    
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
        
        # Hide both labels
        #self.rotate_label.hide()
        #self.varus_label.hide()
        
        if self.current_angle_index < len(self.flexion_angles) - 1:
            # Enable next button and label
            #self.next_button.setEnabled(True)
            self.next_label.show()
        else:
            # Experiment complete
            self.instruction_label.setText("Experiment Complete!")
            self.reset_buttons_and_labels()
            self.current_angle_index += 1
            self.update_display()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KneeFlexionExperiment()
    window.show()
    sys.exit(app.exec_())