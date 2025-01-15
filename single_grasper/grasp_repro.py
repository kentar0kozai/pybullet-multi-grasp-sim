from configparser import ConfigParser
import pybullet as p
import pybullet_data
from pyquaternion import Quaternion
import pandas as pd
from math import pi, sqrt
import os
import ast
from scipy.spatial import ConvexHull, distance
import numpy as np
import sys

row_index = 10  # Change this to load a different row

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


def get_obj_info(oID):  # TODO: what about not mesh objects?
    """
    get object data to figure out how far away the hand needs to be to make its approach
    """
    obj_data = p.getCollisionShapeData(oID, -1)[0]
    # geometry_type = obj_data[2]
    # print("geometry type: " + str(geometry_type))
    dimensions = obj_data[3]
    # print("dimensions: "+ str(dimensions))
    local_frame_pos = obj_data[5]
    # print("local frome position: " + str(local_frame_pos))
    # local_frame_orn = obj_data[6]
    # print("local frame oren: " + str(local_frame_orn))
    diagonal = sqrt(dimensions[0] ** 2 + dimensions[1] ** 2 + dimensions[2] ** 2)
    # print("diagonal: ", diagonal)
    max_radius = diagonal / 2
    return local_frame_pos, max_radius


def get_new_normals(force_vector, normal_force, sides, radius):
    return_vectors = []
    vector_to_cross = np.array((force_vector[0] + 1, force_vector[1] + 2, force_vector[2] + 3))
    orthg = np.cross(force_vector, vector_to_cross)
    orthg_vector = (orthg / np.linalg.norm(orthg)) * radius
    rot_angle = (2 * pi) / sides
    split_force = normal_force / sides

    for side_num in range(sides):
        rotated_orthg = Quaternion(axis=force_vector, angle=(rot_angle * side_num)).rotate(orthg_vector)
        new_vect = force_vector + np.array(rotated_orthg)
        norm_vect = (new_vect / np.linalg.norm(new_vect)) * split_force

        # 正規化して異常なベクトルをフィルタリング
        if np.linalg.norm(norm_vect) < 1e6:  # 適切なしきい値を設定
            return_vectors.append(norm_vect)

    return return_vectors


def gws_pyramid_extension(rID, oID, pyramid_sides=force_pyramid_sides, pyramid_radius=force_pyramid_radius):
    # often dont have enough contact points to create a qhull of the right dimensions, so create more that are very close to the existing ones
    local_frame_pos, max_radius = get_obj_info(oID)
    # sim uses center of mass as a reference for the Cartesian world transforms in getBasePositionAndOrientation
    obj_pos, obj_orn = p.getBasePositionAndOrientation(oID)
    force_torque = []
    contact_points = p.getContactPoints(rID, oID)
    for point in contact_points:
        contact_pos = point[6]
        normal_vector_on_obj = point[7]
        normal_force_on_obj = point[9]
        force_vector = np.array(normal_vector_on_obj) * normal_force_on_obj
        if np.linalg.norm(force_vector) > 0:
            new_vectors = get_new_normals(force_vector, normal_force_on_obj, pyramid_sides, pyramid_radius)

            radius_to_contact = np.array(contact_pos) - np.array(obj_pos)

            for pyramid_vector in new_vectors:
                torque_numerator = np.cross(radius_to_contact, pyramid_vector)
                torque_vector = torque_numerator / max_radius
                force_torque.append(np.concatenate([pyramid_vector, torque_vector]))

    return force_torque


def volume(force_torque):
    """
    Get qhull of the 6D vectors [fx, fy, fz, tx, ty, tz] created by GWS (from contact points).
    Get the volume. Return 0 if force_torque is empty or invalid.
    """
    if not force_torque or len(force_torque) < 6:  # 必要な点が足りない場合
        return 0.0

    try:
        vol = ConvexHull(points=force_torque, qhull_options="QJ")
        return vol.volume
    except Exception as e:
        print(f"ConvexHull error: {e}")
        return 0.0


def epsilon(force_torque):
    """
    Get qhull of the 6D vectors [fx, fy, fz, tx, ty, tz] created by GWS (from contact points).
    Get the distance from centroid of the hull to the closest vertex. Return 0 if invalid.
    """
    if not force_torque or len(force_torque) < 6:  # 必要な点が足りない場合
        return 0.0

    try:
        hull = ConvexHull(points=force_torque, qhull_options="QJ")
        centroid = []
        for dim in range(6):
            centroid.append(np.mean(hull.points[hull.vertices, dim]))
        shortest_distance = min(distance.euclidean(centroid, point) for point in hull.points[hull.vertices])
        return shortest_distance
    except Exception as e:
        print(f"Epsilon calculation error: {e}")
        return 0.0


def grip_qual(oID, rID):
    """
    evaluate the grasp quality
    """
    contact = p.getContactPoints(oID, rID)  # see if hand is still holding obj after gravity is applied
    if len(contact) > 0:
        force_torque = gws_pyramid_extension(rID, oID)
        # print("force_torque: ", force_torque)
        # print("force_torque shape: ", np.array(force_torque).shape)
        vol = volume(force_torque)
        # print("volume: ", vol)
        ep = epsilon(force_torque)
        # print("epsilon: ", ep)
    else:
        vol = None
        ep = None
    return vol, ep


def grasp_with_feedback(oID, rID, sliders):
    """
    Actively control the robot hand to stably grasp the object.
    """

    finger_pairs = {
        2: 3,
        3: 2,
        5: 6,
        6: 5,
        9: 10,
        10: 9,
    }

    p.setGravity(0, 0, -9.8)
    while True:
        p.stepSimulation()

        # スライダーで各ジョイントの目標位置を取得して設定
        for joint_index, slider in sliders:
            target_position = p.readUserDebugParameter(slider)
            p.setJointMotorControl2(
                bodyUniqueId=rID,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=max_grasp_force,  # モーターの出力を設定
            )

        # Get feedback from contact points
        contact_points = p.getContactPoints(rID, oID)
        if len(contact_points) == 0:
            # Close fingers lightly if no contact
            for joint in active_grasp_joints:
                p.setJointMotorControl2(
                    bodyUniqueId=rID,
                    jointIndex=joint,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=target_grasp_velocity,
                    force=max_grasp_force,
                )
        else:
            # Adjust force based on contact feedback
            for point in contact_points:
                normal_force = point[9]  # Normal force
                contact_link = point[3]  # Link in contact
                if contact_link in active_grasp_joints:
                    # print("Force Control Mode!")
                    desired_force = max_grasp_force - normal_force
                    # print(desired_force)
                    p.setJointMotorControl2(
                        bodyUniqueId=rID,
                        jointIndex=contact_link,
                        controlMode=p.TORQUE_CONTROL,
                        force=desired_force,
                    )
                paired_joint = finger_pairs.get(contact_link)
                if paired_joint is not None:
                    p.setJointMotorControl2(
                        bodyUniqueId=rID,
                        jointIndex=paired_joint,
                        controlMode=p.TORQUE_CONTROL,
                        force=desired_force,
                    )
        vol, ep = grip_qual(rID, oID)

        # tqdm風の出力
        if vol is not None and ep is not None:
            sys.stdout.write(f"\rEpsilon: {ep:.4f} | Volume: {vol:.4f}")
            sys.stdout.flush()
        else:
            sys.stdout.write("\rEpsilon: None | Volume: None")
            sys.stdout.flush()


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
    initial_joint_states = {}
    for joint_index, joint_state in robot_joints.items():
        p.resetJointState(rID, joint_index, joint_state[0])
        initial_joint_states[joint_index] = joint_state[0]

    # ジョイント情報を取得
    num_joints = p.getNumJoints(rID)
    print(f"Number of joints: {num_joints}")

    # スライダーを使ったデバッグパラメータを追加
    sliders = []
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(rID, joint_index)
        joint_name = joint_info[1].decode("utf-8")
        joint_type = joint_info[2]

        # Revoluteジョイントのみにスライダーを追加
        if joint_type == p.JOINT_REVOLUTE:
            initial_position = initial_joint_states.get(joint_index, 0)
            joint_lower_limit = joint_info[8] if joint_info[8] > -1e10 else -3.14
            joint_upper_limit = joint_info[9] if joint_info[9] < 1e10 else 3.14
            slider = p.addUserDebugParameter(joint_name, joint_lower_limit, joint_upper_limit, initial_position)
            sliders.append((joint_index, slider))

    grasp_with_feedback(oID, rID, sliders)


if __name__ == "__main__":
    main()
