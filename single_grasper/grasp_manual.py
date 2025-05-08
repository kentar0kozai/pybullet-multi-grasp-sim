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

# --- 定数 & 設定ファイルパス ---
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CONFIG_PATH = os.path.join(BASE_DIR, "bh_config.ini")
CSV_PATH = os.path.join(PROJECT_ROOT, "good_grasps.csv")

# --- 設定読み込み ---
config = ConfigParser()
config.read(CONFIG_PATH)

FORCE_PYRAMID_SIDES = config.getint("eval_settings", "force_pyramid_sides")
FORCE_PYRAMID_RADIUS = config.getfloat("eval_settings", "force_pyramid_radius")
ROBOT_URDF_PATH = os.path.join(BASE_DIR, config.get("file_paths", "robot_path"))
OBJECT_URDF_PATH = os.path.join(BASE_DIR, config.get("file_paths", "object_path"))
OBJECT_SCALE = config.getfloat("file_paths", "object_scale")


def init_simulation(gravity=(0, 0, 0), timestep=1 / 240):
    """PyBullet を初期化して物理設定を行う"""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(*gravity)
    p.setTimeStep(timestep)
    p.setPhysicsEngineParameter(numSolverIterations=200, contactERP=0.9)


def load_entities(init_positions):
    """ロボットハンドとオブジェクトをロードし、固定制約を作成"""
    hand_id = p.loadURDF(ROBOT_URDF_PATH, baseOrientation=[0, -0.7068252, 0, 0.7073883], useFixedBase=True)
    obj_id = p.loadURDF(OBJECT_URDF_PATH, basePosition=[-0.05, 0, 0], globalScaling=OBJECT_SCALE, useFixedBase=False)
    constraint_id = _create_fixed_constraint(obj_id)
    _reset_initial_positions(hand_id, init_positions)
    return hand_id, obj_id, constraint_id


def _create_fixed_constraint(body_id):
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


def _reset_initial_positions(body_id, init_positions):
    for joint_index, angle in init_positions.items():
        p.resetJointState(body_id, joint_index, angle)


def create_sliders(body_id, init_positions):
    """Revoluteジョイント用のスライダーを追加して返す"""
    sliders = []
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        if info[2] == p.JOINT_REVOLUTE:
            name = info[1].decode("utf-8")
            init = init_positions.get(i, 0.0)
            slider = p.addUserDebugParameter(name, -1.57, 1.57, init)
            sliders.append((i, slider))
    return sliders


def handle_input(released, constraint_id, hand_id, obj_id, sliders, vol, ep):
    """
    キー入力:
        gキー → シリンダー解放＋重力ON
        sキー → 把持情報をCSVに保存
        スライダー → ジョイント制御
    """
    events = p.getKeyboardEvents()
    if not released and p.KEY_WAS_TRIGGERED & events.get(ord("g"), 0):
        p.removeConstraint(constraint_id)
        p.setGravity(0, 0, -9.8)
        print("🎉 Cylinder released! Now gravity ON.")
        released = True

    if p.KEY_WAS_TRIGGERED & events.get(ord("s"), 0):
        save_grasp(hand_id, obj_id, CSV_PATH, sliders, vol, ep)

    _apply_joint_controls(hand_id, sliders)
    return released


def _apply_joint_controls(body_id, sliders, force=0.3):
    for j_idx, slider_id in sliders:
        tgt = p.readUserDebugParameter(slider_id)
        p.setJointMotorControl2(
            bodyUniqueId=body_id,
            jointIndex=j_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=tgt,
            force=force,
        )


def save_grasp(hand_id, obj_id, csv_path, sliders, vol, ep):
    """現在の把持状態をCSVに保存/追記"""

    def _pose_repr(body):
        pos, ori = p.getBasePositionAndOrientation(body)
        return repr((tuple(pos), tuple(ori)))

    robot_pose = _pose_repr(hand_id)
    object_pose = _pose_repr(obj_id)

    # joints = {i: p.getJointState(hand_id, i)[:2] for i in range(p.getNumJoints(hand_id))}

    joints = {}
    for joint_index, slider_id in sliders:
        target_angle = p.readUserDebugParameter(slider_id)
        # getJointState の戻り値は (position, velocity, …)
        velocity = p.getJointState(hand_id, joint_index)[1]
        joints[joint_index] = (target_angle, velocity)

    row = {
        "Robot Pose": robot_pose,
        "Robot Joints": repr(joints),
        "Object Pose": object_pose,
        "Quality Volume": vol,
        "Quality Epsilon": ep,
    }

    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"把持情報を保存しました: {csv_path}")


def get_object_info(obj_id):
    """オブジェクト形状から接近半径を計算"""
    data = p.getCollisionShapeData(obj_id, -1)[0]
    dims, frame_pos = data[3], data[5]
    radius = sqrt(sum(d**2 for d in dims)) / 2
    return frame_pos, radius


def _generate_force_normals(f_vec, n_force, sides, radius):
    base = np.cross(f_vec, f_vec + np.array((1, 2, 3)))
    base = (base / np.linalg.norm(base)) * radius
    angle = (2 * pi) / sides
    split = n_force / sides

    normals = []
    for i in range(sides):
        rot = Quaternion(axis=f_vec, angle=angle * i).rotate(base)
        vec = f_vec + rot
        norm_vec = vec / np.linalg.norm(vec) * split
        if np.linalg.norm(norm_vec) < 1e6:
            normals.append(norm_vec)
    return normals


def gws_pyramid(r_id, o_id, sides=FORCE_PYRAMID_SIDES, radius=FORCE_PYRAMID_RADIUS):
    frame_pos, max_r = get_object_info(o_id)
    obj_pos = p.getBasePositionAndOrientation(o_id)[0]

    ft = []
    for pt in p.getContactPoints(r_id, o_id):
        contact, normal, force = np.array(pt[6]), np.array(pt[7]), pt[9]
        f_vec = normal * force
        if np.linalg.norm(f_vec) <= 0:
            continue
        normals = _generate_force_normals(f_vec, force, sides, radius)
        r_vec = contact - obj_pos
        for nv in normals:
            torque = np.cross(r_vec, nv) / max_r
            ft.append(np.concatenate([nv, torque]))
    return ft


def volume(points):
    if len(points) < 6:
        return 0.0
    try:
        return ConvexHull(points, qhull_options="QJ").volume
    except Exception as e:
        print(f"ConvexHull error: {e}")
        return 0.0


def epsilon(points):
    if len(points) < 6:
        return 0.0
    try:
        hull = ConvexHull(points, qhull_options="QJ")
        verts = hull.points[hull.vertices]
        centroid = verts.mean(axis=0)
        return min(distance.euclidean(centroid, v) for v in verts)
    except Exception as e:
        print(f"Epsilon error: {e}")
        return 0.0


def evaluate_grasp(o_id, r_id):
    """把持品質を評価して (volume, epsilon) を返す"""
    if p.getContactPoints(o_id, r_id):
        ft = gws_pyramid(r_id, o_id)
        return volume(ft), epsilon(ft)
    return None, None


def main():
    init_simulation()
    init_positions = {8: 1.57}
    hand_id, obj_id, constraint_id = load_entities(init_positions)
    sliders = create_sliders(hand_id, init_positions)

    released = False
    while True:
        vol, ep = evaluate_grasp(obj_id, hand_id)
        released = handle_input(released, constraint_id, hand_id, obj_id, sliders, vol, ep)
        eps_str = f"{ep:.4f}" if ep is not None else "None"
        vol_str = f"{vol:.4f}" if vol is not None else "None"
        sys.stdout.write(f"\rEpsilon: {eps_str} | Volume: {vol_str}")
        sys.stdout.flush()
        p.stepSimulation()


if __name__ == "__main__":
    main()
