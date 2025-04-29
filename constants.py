# experiment parameters
HOLD_TIME = 5 #seconds to hold knee positions
LACHMANN_TIME = 8 # seconds for lachmann test
FLEXION_ANGLES = [0, 30, 60, 90, 120]

# Bone STLs
#FEMUR = "femur_simplified.stl"
FEMUR = "simplify_Segmentation_1_femur.stl"
#TIBIA = "tibia_simplified.stl"
TIBIA = "simplify_Segmentation_1_tibia.stl"

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
DATA_PREVIOUS_TEST = "print_data.F_sensor_temp_data_79.txt"
DATA_CSV = "data.csv"
