import os
import numpy as np

# experiment parameters
HOLD_TIME = 5 #seconds to hold knee positions
LACHMANN_TIME = 8 # seconds for lachmann test
FLEXION_ANGLES = [0, 30, 60, 90, 120]

# current folder
current_folder = os.path.dirname(os.path.abspath(__file__))

# Bone STLs
femur_fileName = "femur_new.stl"
FEMUR = os.path.join(current_folder, "data_for_gui", femur_fileName)

tibia_fileName = "tibia_new.stl"
TIBIA = os.path.join(current_folder, "data_for_gui", tibia_fileName)

#camera settings bone visualization
DISTANCE_BONE_VIZ = 2000
SCALE_FACTOR_ARROW = 20


#landmarks of femur and tibia for grood & suntay calculations
FEMUR_LATERAL = np.array([110.0960693359375, -108.0730972290039, 1385.2410888671875])
FEMUR_MEDIAL = np.array([96.95680236816406,-164.77444458007812,1386.5252685546875])
FEMUR_PROXIMAL= np.array([75,-130,935])
FEMUR_DISTAL = np.array([83.05928802490234,-130.9730682373047,1373.7659912109375])

TIBIA_LATERAL = np.array([80,-105,1401.037])
TIBIA_MEDIAL= np.array([69.4353256225586,-142.4228515625,1407.4371337890625])
TIBIA_PROXIMAL = np.array([66.03421783447266,-120.49935913085938,1400.6976318359375])
TIBIA_DISTAL = np.array([56.0771484375,-104.6276626586914,1806.37841796875])


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
DATA_PREVIOUS_TEST = "/home/annick/GUI/data_for_gui/print_data.F_sensor_temp_data_7.txt"
DATA_CSV = os.path.join(root_folder, "knee_eval_ws", "data.csv")
DATA_CSV = "data_new.csv"