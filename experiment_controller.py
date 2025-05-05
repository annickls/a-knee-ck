import os
import time
import datetime
import constants
import numpy as np

class ExperimentController:
    """
    Controls the knee flexion experiment flow, managing test states,
    transitions, and data recording.
    """
    
    def __init__(self, parent):
        # Reference to the main UI
        self.parent = parent
        
        # Experiment state
        self.experiment_running = False
        self.current_angle_index = 0
        self.current_test_type = 'none'
        self.remaining_time = 0
        
        # Data recording
        self.recording = False
        self.current_recording_data = []
        self.recording_start_time = None
        self.current_test_name = ""
        
        # Data storage
        self.forces = np.zeros((0, 3))
        self.torques = np.zeros((0, 3))
        self.current_data_index = 0
        
        # History
        self.force_history = []
        self.torque_history = []
        
        # Bone tracking data
        self.last_femur_position = None
        self.last_femur_quaternion = None
        self.last_tibia_position = None
        self.last_tibia_quaternion = None
        
        # Ensure directory exists for data files
        os.makedirs("recorded_data", exist_ok=True)

    def start_experiment(self):
        """Initialize a new experiment session"""
        self.current_angle_index = 0
        self.current_angle = constants.FLEXION_ANGLES[self.current_angle_index]
        self.experiment_running = True
        self.current_test_type = 'none'
        self.force_history = []
        self.torque_history = []
        self.current_data_index = 0
        
        # Call UI update callback
        if hasattr(self.parent, 'on_experiment_started'):
            self.parent.on_experiment_started()
            
        return self.current_angle
    
    def next_angle(self):
        """Advance to the next angle in the experiment sequence"""
        self.current_angle_index += 1
        
        if self.current_angle_index < len(constants.FLEXION_ANGLES):
            self.current_angle = constants.FLEXION_ANGLES[self.current_angle_index]
            
            # Call UI update callback
            if hasattr(self.parent, 'on_angle_changed'):
                self.parent.on_angle_changed()
                
            return self.current_angle
        else:
            return None  # No more angles
    
    def start_test_phase(self, test_name):
        """Start a specific test phase with recording"""
        self.current_test_type = test_name
        self.remaining_time = constants.LACHMANN_TIME if test_name == 'lachmann' else constants.HOLD_TIME
        self.start_recording(test_name)
        
        # Call UI update callback
        if hasattr(self.parent, 'on_test_phase_started'):
            self.parent.on_test_phase_started(test_name, self.remaining_time)
            
        return self.remaining_time
    
    def update_timer(self):
        """Update countdown timer and check if complete"""
        self.remaining_time -= 1
        
        if self.remaining_time <= 0:
            self.complete_current_phase()
            return True  # Phase completed
        
        # Call UI update callback
        if hasattr(self.parent, 'on_timer_updated'):
            self.parent.on_timer_updated(self.remaining_time)
            
        return False  # Phase still in progress
    
    def complete_current_phase(self):
        """Complete the current test phase"""
        # Stop recording data
        if self.recording:
            self.stop_recording()
        
        # Special case for Lachmann test completion (end of experiment)
        if self.current_test_type == 'lachmann':
            self.complete_experiment()
        else:
            # Call UI update callback for phase completion
            if hasattr(self.parent, 'on_phase_completed'):
                self.parent.on_phase_completed(self.current_test_type)
    
    def complete_experiment(self):
        """Complete the entire experiment"""
        self.experiment_running = False
        self.current_test_type = 'none'
        
        # Call UI update callback
        if hasattr(self.parent, 'on_experiment_completed'):
            self.parent.on_experiment_completed()
    
    def is_last_angle(self):
        """Check if we're at the last flexion angle"""
        return self.current_angle_index >= (len(constants.FLEXION_ANGLES) - 1)
    
    def start_recording(self, test_name):
        """Start recording data for the current test"""
        self.recording = True
        self.current_recording_data = []
        self.recording_start_time = time.time()
        self.current_test_name = test_name
        print(f"Started recording data for {test_name}")

    def stop_recording(self):
        """Stop recording and save data to file"""
        if not self.recording:
            return
            
        self.recording = False
        
        # Create a filename with timestamp, angle, and test type
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        angle = constants.FLEXION_ANGLES[self.current_angle_index]
        filename = f"recorded_data/{timestamp}_{angle}deg_{self.current_test_name}.txt"
        
        # Write data to file
        with open(filename, 'w') as f:
            f.write("# Timestamp, Fx, Fy, Fz, Tx, Ty, Tz, FemurPosX, FemurPosY, FemurPosZ, FemurQuatW, FemurQuatX, FemurQuatY, FemurQuatZ, TibiaPosX, TibiaPosY, TibiaPosZ, TibiaQuatW, TibiaQuatX, TibiaQuatY, TibiaQuatZ\n")
            for data_point in self.current_recording_data:
                f.write(','.join(map(str, data_point)) + '\n')
        
        print(f"Saved {len(self.current_recording_data)} data points to {filename}")
        self.current_recording_data = []
    
    def process_new_data(self, force, torque, femur_position, femur_quaternion, tibia_position, tibia_quaternion):
        """Process new sensor data from the system"""
        # Store positions and quaternions
        self.last_femur_position = femur_position
        self.last_femur_quaternion = femur_quaternion
        self.last_tibia_position = tibia_position
        self.last_tibia_quaternion = tibia_quaternion
        
        # Store force/torque in arrays
        if len(self.forces) > 100:  # Keep only last 100 points
            self.forces = np.vstack([self.forces[1:], force])
            self.torques = np.vstack([self.torques[1:], torque])
        else:
            if len(self.forces) == 0:
                self.forces = np.array([force])
                self.torques = np.array([torque])
            else:
                self.forces = np.vstack([self.forces, force])
                self.torques = np.vstack([self.torques, torque])
        
        self.current_data_index = len(self.forces) - 1
        
        # Add to history if experiment is running
        if self.experiment_running:
            self.force_history.append(force)
            self.torque_history.append(torque)
            
            # Keep history to specified size
            if len(self.force_history) > constants.HISTORY_SIZE:
                self.force_history.pop(0)
                self.torque_history.pop(0)
        
        # Record data point if recording is active
        if self.recording:
            current_time = time.time() - self.recording_start_time
            data_point = [
                current_time,
                force[0], force[1], force[2],
                torque[0], torque[1], torque[2],
                femur_position[0], femur_position[1], femur_position[2],
                femur_quaternion[0], femur_quaternion[1], femur_quaternion[2], femur_quaternion[3],
                tibia_position[0], tibia_position[1], tibia_position[2],
                tibia_quaternion[0], tibia_quaternion[1], tibia_quaternion[2], tibia_quaternion[3]
            ]
            self.current_recording_data.append(data_point)
            
        # Call UI update callback
        if hasattr(self.parent, 'on_data_updated'):
            self.parent.on_data_updated()
            
        return self.current_data_index