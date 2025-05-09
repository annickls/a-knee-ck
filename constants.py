# experiment parameters
HOLD_TIME = 15 #seconds to hold knee positions
LACHMANN_TIME = 8 # seconds for lachmann test
FLEXION_ANGLES = [0, 30, 60, 90, 120]

# Bone STLs
TIBIA= "/home/annick/a-knee-ck/data_for_gui/femur_simplified.stl"
#FEMUR = "simplify_Segmentation_1_femur.stl"
#FEMUR = "/home/annick/GUI/data_for_gui/simplify_Segmentation_1_femur.stl"

FEMUR = "/home/annick/a-knee-ck/data_for_gui/tibia_simplified.stl"
#TIBIA = "simplify_Segmentation_1_tibia.stl"
#TIBIA = "/home/annick/GUI/data_for_gui/simplify_simplify_Segmentation_1_tibia.stl"

PIVOT_POINT_FEMUR = [0, 0, 0]
PIVOT_POINT_TIBA = [0, 0, 0]
DISTANCE_BONE_VIZ = 3000
#TRACKER_FEMUR = [-50.0, -200.0, 1220.0]
TRACKER_FEMUR = [300.0, 0.0, 200.0]
#TRACKER_TIBIA = [-100.0, -200.0, 1520.0]
TRACKER_TIBIA = [0.0, 0.0, 0.0]


#TRACKER_TIBIA = [-100.0, -200.0, 1520.0]
TRACKER_FEMUR = [0.0, 0.0, 0.0]
TRACKER_TIBIA = [0.0, 0.0, 0.0]

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
#DATA_PREVIOUS_TEST = "print_data.F_sensor_temp_data_79.txt"
DATA_PREVIOUS_TEST = "/home/annick/GUI/data_for_gui/print_data.F_sensor_temp_data_7.txt"
DATA_CSV = "/home/annick/knee_eval_ws/data.csv"
#DATA_CSV = "data.csv"
