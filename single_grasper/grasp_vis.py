from configparser import ConfigParser
import pybullet as p
import pybullet_data
import pandas as pd
import os

row_index = 10  # Change this to load a different row

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# GLOBAL VARIABLES - from config file
config = ConfigParser()
print(os.path.join(os.path.dirname(__file__), "bh_config.ini"))
config.read(os.path.join(os.path.dirname(__file__), "bh_config.ini"))

robot_path = config.get("file_paths", "robot_path")
robot_path = os.path.join(PROJECT_ROOT, robot_path)
object_path = config.get("file_paths", "object_path")
object_path = os.path.join(PROJECT_ROOT, object_path)
object_scale = config.getfloat("file_paths", "object_scale")

# PYBULLET INITIALIZATION
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(fixedTimeStep=1 / 240.0, numSubSteps=4)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, enable=0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, enable=0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, enable=0)
p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=135, cameraPitch=-20, cameraTargetPosition=[0.0, 0.0, 0.0])


# UTILITIES
def reset_hand(robot_pose):
    """Load and reset the robot hand."""
    rID = p.loadURDF(robot_path, basePosition=robot_pose[0], baseOrientation=robot_pose[1], useFixedBase=True)
    return rID


def reset_object(object_pose):
    """Load and reset the object."""
    oID = p.loadURDF(object_path, basePosition=object_pose[0], baseOrientation=object_pose[1], globalScaling=object_scale, useFixedBase=True)
    return oID


# LOAD CSV FILE
def load_csv(csv_path, row_index):
    """Load a specific row from the CSV file."""
    data = pd.read_csv(csv_path)
    row = data.iloc[row_index]
    robot_pose = eval(row["Robot Pose"])
    robot_joints = eval(row["Robot Joints"])
    object_pose = eval(row["Object Pose"])
    return robot_pose, robot_joints, object_pose


# MAIN
def main():
    # Path to the CSV file
    csv_path = "good_grasps.csv"

    # Load the robot pose, joints, and object pose from the CSV
    robot_pose, robot_joints, object_pose = load_csv(csv_path, row_index)

    # Reset simulation
    p.resetSimulation()

    # Load robot and object
    rID = reset_hand(robot_pose)
    oID = reset_object(object_pose)

    # Set robot joints
    for joint_index, joint_state in robot_joints.items():
        p.resetJointState(rID, joint_index, joint_state[0])

    # Run simulation
    while True:
        p.stepSimulation()


if __name__ == "__main__":
    main()
