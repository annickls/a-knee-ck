import os
import numpy as np

# experiment parameters
HOLD_TIME = 30 #seconds to hold knee positions
HOLD_INDIVIDUAL = 30 #seconds for individual data recording
LACHMANN_TIME = 15 # seconds for lachmann test
FLEXION_ANGLES = [0, 30, 60, 90, 120]

# current folder
current_folder = os.path.dirname(os.path.abspath(__file__))

# Bone STLs
femur_fileName = "femur_new.stl"
FEMUR = os.path.join(current_folder, "data_for_gui", femur_fileName)

tibia_fileName = "tibia_new.stl"
TIBIA = os.path.join(current_folder, "data_for_gui", tibia_fileName)

# recorded data
RECORDED = os.path.join(current_folder, "recorded_data")

# data reading
root_folder = os.path.dirname(current_folder)
DATA_PREVIOUS_TEST = "/home/annick/GUI/data_for_gui/print_data.F_sensor_temp_data_7.txt"
DATA_CSV = os.path.join(root_folder, "knee_eval_ws", "data.csv")
#DATA_CSV = "data_new.csv"


#landmarks of femur and tibia for grood & suntay calculations 
#knee model
FEMUR_LATERAL = np.array([110.0960693359375, -108.0730972290039, 1385.2410888671875])
FEMUR_MEDIAL = np.array([96.95680236816406,-164.77444458007812,1386.5252685546875])
FEMUR_PROXIMAL= np.array([75,-130,935])
FEMUR_DISTAL = np.array([83.05928802490234,-130.9730682373047,1373.7659912109375])
TIBIA_LATERAL = np.array([80,-105,1401.037])
TIBIA_MEDIAL= np.array([69.4353256225586,-142.4228515625,1407.4371337890625])
TIBIA_PROXIMAL = np.array([66.03421783447266,-120.49935913085938,1400.6976318359375])
TIBIA_DISTAL = np.array([56.0771484375,-104.6276626586914,1806.37841796875])

# femur definitions spheres for joint gap measurement
SURFACE_MEDIAL_1 = np.array([96.53736877441406,-173.56982421875,1377.37841796875])
SURFACE_MEDIAL_2 = np.array([70.72567749023438,-166.7306671142578,1398.4432373046875])
SURFACE_LATERAL_1 = np.array([112.36446380615234,-98.36043548583984,1382.3638916015625])
SURFACE_LATERAL_2 = np.array([85.2894058227539,-95.4275131225586,1400.6488037109375])
CENTER_AXIS_MEDIAL = np.array([68.44070434570312,-173.55142211914062,1368.13623046875])
CENTER_AXIS_LATERAL = np.array([89.07107543945312,-84.18157196044922,1376.0799560546875])
#CENTER_AXIS_LATERAL = np.array([84.7090072631836,-92.42295837402344,1350.3447265625,])
#CENTER_AXIS_MEDIAL = np.array([66.45207977294922,-155.77943420410156,1350.2264404296875])

TEST_POINT_MEDIAL = np.array([70,-160,1375])

#calculation torques - distances FT-Dose to Tibia Origin
DELTA_X = 0.077
DELTA_Y = 0.043
DELTA_Z = 0.226
#CENTER_FT = np.array([])


# 3D plot settings
AXIS_FACTOR = 0.5
AXIS_LINEWIDTH = 0.85
HISTORY_SIZE = 100
FORCE_MAX = 12
TORQUE_MAX = 3
ARROW_SIZE_FORCE = 6.0
ARROW_LENGTH_FACTOR_FORCE = 0.85
HEAD_SIZE_FACTOR_FORCE = 0.15
ARROW_SIZE_TORQUE = 6.0
ARROW_LENGTH_FACTOR_TORQUE = 5.0
HEAD_SIZE_FACTOR_TORQUE = 0.5
SHAFT_WIDTH = 2.0
DISTANCE_BONE_VIZ = 700
SCALE_FACTOR_ARROW = 20

#2D plot settings
X_LIM_VAL = 40
X_LIM_ROT = 50
Y_MAX_FLEX = 120
Y_MIN_FLEX = -10


# interpolation and other plot adjustments
BINS_ROT = 9
BINS_VAR = 3
FLEXION_BIN_SIZE = 0.5
INTERPOLATION_KIND = 'linear'
SMOOTHING_FACTOR = 2
MIN_POINTS_FOR_SMOOTHING = 2
MOVING_AVERAGE_WINDOW = 13
MOVING_AVERAGE_METHOD = 'weighted'
APPLY_MOVING_AVERAGE = True
WEIGHT_TYPE = 'gaussian'
SIGMA_FACTOR = 0.2

# colors
SALMON =  (0.980, 0.502, 0.447, 1.0)
LIMEGREEN = (0.196, 0.804, 0.196, 1.0)
DEEPSKYBLUE = (0.0, 0.749, 1.0, 1.0)
MEDIUMSLATEBLUE = (0.482, 0.408, 0.933, 1.0)
DODGERBLUE = (0.118, 0.565, 1.0, 1.0)

# buttons
BUTTON_HEIGHT = 60
BUTTON_HEIGHT_2 = 40

