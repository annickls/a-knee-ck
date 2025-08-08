import os
import glob

current_folder = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.dirname(current_folder)
test_path = glob.glob(root_folder + "/knee_eval_ws" + "/data*.csv")
DATA_CSV = os.path.join(root_folder, "knee_eval_ws", "data.csv")

print(f"Data_csv: {DATA_CSV}, root_folder:{root_folder}, current_folder: {current_folder}")
print(test_path)
