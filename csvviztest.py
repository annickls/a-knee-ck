import sys
import csv
import time
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                            QVBoxLayout, QTextEdit, QWidget)
from PyQt5.QtCore import QTimer, Qt
import pandas as pd

class CSVReaderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("CSV Data Reader")
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        
        # Terminal-like display
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setStyleSheet("background-color: black; color: white; font-family: Courier;")
        self.terminal.setFontPointSize(10)
        layout.addWidget(self.terminal)
        
        # Start button
        self.start_button = QPushButton("Start Reading")
        self.start_button.setFixedSize(150, 40)
        self.start_button.clicked.connect(self.toggle_monitoring)
        layout.addWidget(self.start_button, 0, Qt.AlignCenter)
        
        # Initialize variables
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_csv_data)
        self.monitoring = False
        self.csv_path = "data.csv"
        self.last_modified_time = 0
        self.last_size = 0
        
        # Add initial message
        self.terminal.append("CSV Data Reader Ready")
        self.terminal.append(f"Waiting to monitor: {self.csv_path}")
        self.terminal.append("Press 'Start Reading' to begin...")
    
    def toggle_monitoring(self):
        if not self.monitoring:
            # Start monitoring
            self.monitoring = True
            self.start_button.setText("Stop Reading")
            self.terminal.append("\n--- Monitoring Started ---")
            
            # Initialize file stats
            csv_file = Path(self.csv_path)
            if csv_file.exists():
                self.last_modified_time = csv_file.stat().st_mtime
                self.last_size = csv_file.stat().st_size
                self.read_csv_data()  # Read initial data
            else:
                self.terminal.append(f"Error: {self.csv_path} not found!")
                self.toggle_monitoring()  # Stop monitoring
                return
                
            # Start timer to check for changes (check every 500ms)
            self.timer.start(500)
        else:
            # Stop monitoring
            self.monitoring = False
            self.timer.stop()
            self.start_button.setText("Start Reading")
            self.terminal.append("\n--- Monitoring Stopped ---")
    
    def read_csv_data(self):
        csv_file = Path(self.csv_path)
        
        if not csv_file.exists():
            self.terminal.append(f"Error: {self.csv_path} not found!")
            return
        
        current_modified_time = csv_file.stat().st_mtime
        current_size = csv_file.stat().st_size
        
        # Check if file has been modified
        if current_modified_time > self.last_modified_time or current_size != self.last_size:
            try:
                # Read the CSV file
                df = pd.read_csv(self.csv_path)
                
                # Display data
                self.terminal.append("\n--- Data Updated ---")
                self.terminal.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                self.terminal.append(f"Rows: {len(df)}")
                
                # Format and display the data
                with open(self.csv_path, 'r') as f:
                    reader = csv.reader(f)
                    headers = next(reader)  # Get headers
                    
                    for row in reader:
                        if row:  # Check if row is not empty
                            self.terminal.append("\nRow Data:")
                            for i, value in enumerate(row):
                                if i < len(headers):
                                    self.terminal.append(f"  {headers[i]}: {value}")
                
                # Update last modified time and size
                self.last_modified_time = current_modified_time
                self.last_size = current_size
                
                # Scroll to the end
                self.terminal.verticalScrollBar().setValue(
                    self.terminal.verticalScrollBar().maximum()
                )
                
            except Exception as e:
                self.terminal.append(f"Error reading CSV: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSVReaderApp()
    window.show()
    sys.exit(app.exec_())