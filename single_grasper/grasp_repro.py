from configparser import ConfigParser
import pybullet as p
import pybullet_data
import pandas as pd
import os
import ast


row_index = 4  # Change this to load a different row

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# GLOBAL VARIABLES - from config file
config = ConfigParser()
print(os.path.join(os.path.dirname(__file__), "bh_config.ini"))
config.read(os.path.join(os.path.dirname(__file__), "bh_config.ini"))

# Load settings
init_grasp_distance = config.getfloat("grasp_settings", "init_grasp_distance")
speed_find_distance = config.getfloat("grasp_settings", "speed_find_distance")
grasp_distance_margin = config.getfloat("grasp_settings", "grasp_distance_margin")

max_grasp_force = config.getfloat("grasp_settings", "max_grasp_force")
target_grasp_velocity = config.getfloat("grasp_settings", "target_grasp_velocity")
grasp_time_limit = config.getfloat("grasp_settings", "grasp_time_limit")
active_grasp_joints = [int(j.strip()) for j in config.get("grasp_settings", "active_grasp_joints").split(",")]
num_grasps_per_cycle = config.getint("grasp_settings", "num_grasps_per_cycle")
num_cycles_to_grasp = config.getint("grasp_settings", "num_cycles_to_grasp")
num_wrist_rotations = config.getint("grasp_settings", "num_wrist_rotations")
use_wrist_rotations = config.getboolean("grasp_settings", "use_wrist_rotations")

force_pyramid_sides = config.getint("eval_settings", "force_pyramid_sides")
force_pyramid_radius = config.getfloat("eval_settings", "force_pyramid_radius")

use_gui = config.getboolean("gui_settings", "use_gui")
debug_lines = config.getboolean("gui_settings", "debug_lines")
debug_text = config.getboolean("gui_settings", "debug_text")

robot_path = os.path.join(os.path.dirname(__file__), config.get("file_paths", "robot_path"))
object_path = os.path.join(os.path.dirname(__file__), config.get("file_paths", "object_path"))
object_scale = config.getfloat("file_paths", "object_scale")

# PyBullet initialization
if use_gui:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(fixedTimeStep=1 / 240.0, numSubSteps=4)

if use_gui:
    p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=135, cameraPitch=-20, cameraTargetPosition=[0.0, 0.0, 0.0])


# Utilities
def reset_hand(robot_pose):
    """Load and reset the robot hand."""
    return p.loadURDF(robot_path, basePosition=robot_pose[0], baseOrientation=robot_pose[1], useFixedBase=True)


def reset_object(object_pose):
    """Load and reset the object."""
    return p.loadURDF(object_path, basePosition=object_pose[0], baseOrientation=object_pose[1], globalScaling=object_scale, useFixedBase=False)


def load_csv(csv_path, row_index):
    """Load a specific row from the CSV file."""
    data = pd.read_csv(csv_path)
    row = data.iloc[row_index]
    robot_pose = ast.literal_eval(row["Robot Pose"])
    robot_joints = ast.literal_eval(row["Robot Joints"])
    object_pose = ast.literal_eval(row["Object Pose"])
    return robot_pose, robot_joints, object_pose


def grasp_with_feedback(oID, handId):
    """
    Actively control the robot hand to stably grasp the object.
    """
    mass = 0.1
    mag = 9.8 * mass
    pos, oren = p.getBasePositionAndOrientation(handId)
    # finish_time = time.time() + grasp_time_limit
    # p.setGravity(0, 0, -9.8)
    while True:
        p.stepSimulation()
        # p.applyExternalForce(oID, linkIndex=-1, forceObj=[0, 0, -mag], posObj=pos, flags=p.WORLD_FRAME)

        # Get feedback from contact points
        contact_points = p.getContactPoints(handId, oID)
        if len(contact_points) == 0:
            # Close fingers lightly if no contact
            for joint in active_grasp_joints:
                p.setJointMotorControl2(
                    bodyUniqueId=handId,
                    jointIndex=joint,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=target_grasp_velocity,
                    force=max_grasp_force,
                )
        else:
            # Adjust force based on contact feedback
            print("debug")
            p.applyExternalForce(oID, linkIndex=-1, forceObj=[0, 0, -mag], posObj=pos, flags=p.WORLD_FRAME)
            for point in contact_points:
                normal_force = point[9]  # Normal force
                contact_link = point[3]  # Link in contact
                if contact_link in active_grasp_joints:
                    desired_force = max_grasp_force - normal_force
                    p.setJointMotorControl2(
                        bodyUniqueId=handId,
                        jointIndex=contact_link,
                        controlMode=p.TORQUE_CONTROL,
                        force=desired_force,
                    )


def main():
    # Path to the CSV file
    csv_path = "good_grasps.csv"

    # Load a specific row from the CSV file
    robot_pose, robot_joints, object_pose = load_csv(csv_path, row_index)

    # Reset simulation
    p.resetSimulation()

    # Load robot and object
    rID = reset_hand(robot_pose)
    oID = reset_object(object_pose)

    # Set initial joint states
    for joint_index, joint_state in robot_joints.items():
        p.resetJointState(rID, joint_index, joint_state[0])

    grasp_with_feedback(oID, rID)


if __name__ == "__main__":
    main()
