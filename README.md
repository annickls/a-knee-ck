# Knee Evaluation Test Bench


A real-time data visualization system for knee evaluation using OptiTrack motion capture and force/torque (FT) sensor data.

## Overview

This system provides real-time visualization of biomechanical data for knee evaluation studies. The main application displays live data streams from OptiTrack motion capture systems and force/torque sensors in an interactive GUI.

## Main Components

### Core Application
- **`GUI_with_bones_lets_go.py`** - Main application file containing the real-time data visualization interface

### Supporting Classes
- **`constants.py`** - System constants and configuration parameters
- **`mesh_utils.py`** - 3D mesh handling and processing utilities
- **`plot_config1.py`** - Plotting configuration and styling settings
- **`update_visualization.py`** - Real-time visualization update logic

## Getting Started

### Prerequisites
- ROS2 workspace configured
- OptiTrack system setup
- Force/torque sensor hardware

### Running with OptiTrack + FT Data

1. **Start the data streams:**
   ```bash
   cd knee_eval_ws
   ros2 launch pkg_launcher global_launcher.launch.py
   ```

2. **Begin CSV recording:**
   ```bash
   ros2 service call /start_csv_recording std_srvs/srv/Trigger {}
   ```

3. **Zero the force/torque sensor:**
   ```bash
   ros2 service call /ft_sensor/zero std_srvs/srv/Trigger {}
   ```

4. **Launch the main visualization:**
   ```bash
   python GUI_with_bones_lets_go.py
   ```

## Testing & Development

### Testing Without Hardware

For development and testing purposes, you can run the system without OptiTrack hardware:

#### Option 1: Automated Test Environment
- **`both.py`** - Launches both the main program and dummy data generator simultaneously
- **`update_csv_at_50hz.py`** - Generates dummy CSV data at 50Hz for realistic testing

#### Option 2: Random Data Testing
- **`GUI_with_dummy_data.py`** - Uses randomly generated dummy data to demonstrate functionality

#### Data Storage
- **`data.csv`** - CSV file used for storing and reading dummy data during testing

### Usage
```bash
# For automated testing with CSV data
python both.py

# For random dummy data testing
python GUI_with_dummy_data.py
```

## System Architecture

The system follows a modular design:
- Real-time data acquisition through ROS2 services
- Configurable visualization parameters
- Mesh-based 3D rendering
- CSV data logging and playback capabilities

## Development Notes

- The system operates at 50Hz for real-time performance
- All visualization updates are handled asynchronously
- CSV recording can be started/stopped via ROS2 services
- Force/torque sensor supports zeroing for calibration"""

# Write the README file
with open('README.md', 'w') as f:
    f.write(readme_content)

print("README.md file created successfully!")