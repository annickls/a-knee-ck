
import os
import numpy as np

# experiment parameters
HOLD_TIME = 5 #seconds to hold knee positions
LACHMANN_TIME = 8 # seconds for lachmann test
FLEXION_ANGLES = [0, 30, 60, 90, 120]

# current folder
current_folder = os.path.dirname(os.path.abspath(__file__))

# Bone STLs
#FEMUR= "/home/annick/a-knee-ck/data_for_gui/femur_new.stl"
#FEMUR = "simplify_Segmentation_1_femur.stl"
femur_fileName = "femur_new.stl"
FEMUR = os.path.join(current_folder, "data_for_gui", femur_fileName)

#TIBIA = "/home/annick/a-knee-ck/data_for_gui/tibia_new.stl"
#TIBIA = "simplify_Segmentation_1_tibia.stl"
tibia_fileName = "tibia_test2.stl"
TIBIA = os.path.join(current_folder, "data_for_gui", tibia_fileName)

PIVOT_POINT_FEMUR = [0, 0, 0]
PIVOT_POINT_TIBA = [0, 0, 0]
DISTANCE_BONE_VIZ = 2000
#TRACKER_FEMUR = [-50.0, -200.0, 1220.0]
TRACKER_FEMUR = [0.0, 0.0, 0.0]
#TRACKER_TIBIA = [-100.0, -200.0, 1520.0]
TRACKER_TIBIA = [0.0, 0.0, 0.0]

#landmarks
FEMUR_MEDIAL = np.array([83.37752532958984, -106.33291625976562, 1398.119384765625])
FEMUR_LATERAL = np.array([67.22425079345703, -157.83193969726562, 1399.614990234375])
FEMUR_PROXIMAL= np.array([77.49647521972656, -127.54686737060547, 911.6983032226562])
FEMUR_DISTAL = np.array([65.46070098876953, -113.15875244140625, 1384.9970703125])
TIBIA_MEDIAL = np.array([66.68541717529297, -103.38368225097656, 1400.172119140625])
TIBIA_LATERAL = np.array([63.146968841552734, -147.86354064941406, 1407.7625732421875])
TIBIA_PROXIMAL = np.array([66.52336883544922, -121.91870880126953, 1399.853271484375])
TIBIA_DISTAL = np.array([65.01982879638672, -115.64944458007812, 1804.212646484375])

# plot settings
AXIS_FACTOR = 0.5
AXIS_LINEWIDTH = 0.85
HISTORY_SIZE = 100
FORCE_MAX = 12
TORQUE_MAX = 3
ARROW_SIZE = 6.0
SHAFT_WIDTH = 2.0

# colors
SALMON =  (0.980, 0.502, 0.447, 1.0)
LIMEGREEN = (0.196, 0.804, 0.196, 1.0)
DEEPSKYBLUE = (0.0, 0.749, 1.0, 1.0)

# buttons
BUTTON_HEIGHT = 60

# data
root_folder = os.path.dirname(current_folder)
#DATA_PREVIOUS_TEST = "print_data.F_sensor_temp_data_79.txt"
DATA_PREVIOUS_TEST = "/home/annick/GUI/data_for_gui/print_data.F_sensor_temp_data_7.txt"
DATA_CSV = os.path.join(root_folder, "knee_eval_ws", "data.csv")
#DATA_CSV = "data_new.csv"