import os
import sys
from configparser import ConfigParser
from math import pi, sqrt

import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull, distance

CSV_PATH = "good_grasps.csv"

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


def init_simulation(gravity=(0, 0, 0), timestep=1 / 240):
    """PyBullet を初期化して物理設定を行う"""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(*gravity)
    p.setTimeStep(timestep)
    p.setPhysicsEngineParameter(numSolverIterations=200, contactERP=0.9)


def load_entities():
    """ロボットとシリンダーをロードして ID を返す"""
    robot_id = p.loadURDF("../RobotURDFs/wbr_description/urdf/wbr_hand.urdf", baseOrientation=[0, -0.7068252, 0, 0.7073883], useFixedBase=True)
    cylinder_id = p.loadURDF("../ObjectURDFs/cylinder/cylinder.urdf", basePosition=[-0.05, 0, 0], useFixedBase=False)
    return robot_id, cylinder_id


def create_fixed_constraint(body_id):
    """シリンダーをワールドに固定する Constraint を作成"""
    pos, ori = p.getBasePositionAndOrientation(body_id)
    return p.createConstraint(
        parentBodyUniqueId=body_id,
        parentLinkIndex=-1,
        childBodyUniqueId=-1,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=pos,
        parentFrameOrientation=[0, 0, 0, 1],
        childFrameOrientation=ori,
    )


def reset_initial_positions(robot_id, init_positions):
    """関節の初期角度をリセット"""
    for joint_index, angle in init_positions.items():
        p.resetJointState(robot_id, joint_index, angle)


def create_sliders(robot_id, init_positions):
    """Revoluteジョイント用のスライダーを追加してリストで返す"""
    sliders = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        joint_type = info[2]
        if joint_type == p.JOINT_REVOLUTE:
            name = info[1].decode("utf-8")
            init = init_positions.get(i, 0.0)
            slider_id = p.addUserDebugParameter(name, -1.57, 1.57, init)
            sliders.append((i, slider_id))
    return sliders


def save_current_grasp(robot_id, object_id, csv_path):
    """現在の把持情報をCSVに保存／追記する"""
    # ロボットベース姿勢取得 (位置, クォータニオン)
    pos, ori = p.getBasePositionAndOrientation(robot_id)
    robot_pose = (tuple(pos), tuple(ori))

    # 各ジョイントの (位置, 速度) を取得
    joints = {}
    for i in range(p.getNumJoints(robot_id)):
        js = p.getJointState(robot_id, i)
        joints[i] = (js[0], js[1])

    # 物体ベース姿勢取得
    o_pos, o_ori = p.getBasePositionAndOrientation(object_id)
    object_pose = (tuple(o_pos), tuple(o_ori))

    # DataFrame に整形
    new_row = {
        "Robot Pose": repr(robot_pose),
        "Robot Joints": repr(joints),
        "Object Pose": repr(object_pose),
        "Quality Volume": None,  # 必要に応じて設定
        "Quality Epsilon": None,  # 必要に応じて設定
    }
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(csv_path, index=False)
    print(f"💾 把持情報を保存しました: {csv_path}")


def handle_keyboard_events(released, constraint_id, robot_id, object_id, sliders):
    """
    キー入力を処理する:
        - gキー: シリンダーの固定解除＋重力有効化
        - sキー: 現在の把持情報をCSVに保存
    """
    events = p.getKeyboardEvents()

    # gキー解除
    if not released and ord("g") in events and (events[ord("g")] & p.KEY_WAS_TRIGGERED):
        p.removeConstraint(constraint_id)
        p.setGravity(0, 0, -9.8)
        print("🎉 Cylinder released! Now gravity ON.")
        released = True

    # sキーでCSV保存
    if ord("s") in events and (events[ord("s")] & p.KEY_WAS_TRIGGERED):
        save_current_grasp(robot_id, object_id, CSV_PATH)

    # スライダー制御
    apply_joint_controls(robot_id, sliders)

    return released


def apply_joint_controls(robot_id, sliders, force=0.3):
    """スライダーの値を読み取ってジョイント制御"""
    for joint_index, slider_id in sliders:
        target = p.readUserDebugParameter(slider_id)
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            force=force,
        )


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


def main():
    # 1) 初期設定
    init_simulation(gravity=(0, 0, 0), timestep=1 / 240)

    # 2) エンティティのロード
    robot_id, cylinder_id = load_entities()
    fixed_const = create_fixed_constraint(cylinder_id)

    # 3) 初期ポーズ設定
    """ Number of joints: 13
        Base link index: -1 (base link)
        Link index: 0,  Link name: hand_base_link
        Link index: 1,  Link name: index_finger_1
        Link index: 2,  Link name: index_finger_2
        Link index: 3,  Link name: index_finger_3
        Link index: 4,  Link name: index_tip
        Link index: 5,  Link name: middle_finger_1
        Link index: 6,  Link name: middle_finger_2
        Link index: 7,  Link name: middle_finger_3
        Link index: 8,  Link name: thumb_1
        Link index: 9,  Link name: thumb_2
        Link index: 10, Link name: thumb_3
        Link index: 11, Link name: thumb_4
        Link index: 12, Link name: thumb_tip """
    init_positions = {8: 1.57}
    reset_initial_positions(robot_id, init_positions)

    # 4) デバッグ用スライダー作成
    sliders = create_sliders(robot_id, init_positions)

    # 5) メインループ
    released = False
    while True:
        released = handle_keyboard_events(released, fixed_const, robot_id, cylinder_id, sliders)
        vol, ep = grip_qual(robot_id, cylinder_id)

        # tqdm風の出力
        if vol is not None and ep is not None:
            sys.stdout.write(f"\rEpsilon: {ep:.4f} | Volume: {vol:.4f}")
            sys.stdout.flush()
        else:
            sys.stdout.write("\rEpsilon: None | Volume: None")
            sys.stdout.flush()
        p.stepSimulation()
        # time.sleep(1 / 240)


if __name__ == "__main__":
    main()
