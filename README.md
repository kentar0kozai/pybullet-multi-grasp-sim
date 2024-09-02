## Usage: 
The goal here is to get the most simple form of a grasp planner working first, then move on to more sophisticated methods

__To Use:__
* __Edit + run write_config.py__  
    - file_paths
        - __robot_path__ - relative path from grasper.py to robot hand URDF file
        - __object_path__ - path to object URDF file
        - __object_scale__ - scale of object in URDF file (to adjust to size of hand, should default to 1)
    - grasp_settings
        - __init_grasp_distance__ - how far from the origin should the hand be at the start (just needs to be beyond the length of the object) to attempt to find closest point
        - __speed_find_distance__ - speed the object moves toward the hand to find ideal grasp distance
        - __grasp_distance_margin__ - how far from touching do you want the palm to be when attempting grips
        - __max_grasp_force__ - max force allowed
        - __target_grasp_velocity__ - target velocity for joints when grasping
        - __grasp_time_limit__ - how long given to find a grasp
        - __active_grasp_joints__ - which joints in the hand to use - specified given joint number from Pybullet getJointInfo
        - __num_grasps_per_cycle__ - number of grasps attempted in each rotation around object (evenly spaced attempts)
        - __num_cycles_to_grasp__ - number of rotations (with angles spread between pi/2 and 0 - ie euler theta) 
        - __use_wrist_rotations__ - binary. allows for rotations around the wrist at each grasp attempt
        - __num_wrist_rotations__ - number of rotations around the wrist axis for each grasp location
    - eval_settings
        - __force_pyramid_sides__ - for grasp wrench space: how many sides does the pyramid approximating the friction cone have
        - __force_pyramid_radius__ - for grasp wrench space: what is the radius of the friction cone approximated by the pyramid
    - gui_settings
        - __use_gui__ - not functioning yet. should allow the whole system to work without visualizations to allow for speedups/parallelization
        - __debug_lines__ - shows axis lines for the hand
        - __debug_text__ - shows text on screen to update the user
       
* __Run grasper.py__
    - Should return a list of good grasps specified by a (Position, Orientation, Joint Angle) Tuple. Also returned is the final pose of the object and the grasp quality metrics (volume and epsilson). At the moment, only grasps that pass the gravity check threshold are returned. 
