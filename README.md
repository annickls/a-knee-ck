GUI_with_bones_lets_go.py is the important file, main code visualizing real-time data.

to start the optitrack + FT data stream:
    cd knee_eval_ws
    ros2 launch pkg_launcher global_launcher.launch.py
    ros2 service call /start_csv_recording std_srvs/srv/Trigger {}

classes used in this program:
    constants.py
    mesh_utils.py
    plot_config1.py
    update_visualization.py



to test without Optitrack:
    both.py: starts the main program and the dummy csv-data generator update_csv_at_50hz.py.
    GUI_with_dummy_data: uses random dummy data to show functionality.
    data.csv: used to store and read dummy data.