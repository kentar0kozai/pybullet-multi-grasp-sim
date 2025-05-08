#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リファクタリング版: 関数ベースで整理し、可読性を向上させました✨
"""
import ast
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


# 設定ロード
def load_config(config_file: str) -> dict:
    config = ConfigParser()
    config.read(config_file)
    base = os.path.dirname(config_file)
    return {
        "max_force": config.getfloat("grasp_settings", "max_grasp_force"),
        "target_vel": config.getfloat("grasp_settings", "target_grasp_velocity"),
        "active_joints": [int(x) for x in config.get("grasp_settings", "active_grasp_joints").split(",")],
        "pyramid_sides": config.getint("eval_settings", "force_pyramid_sides"),
        "pyramid_radius": config.getfloat("eval_settings", "force_pyramid_radius"),
        "use_gui": config.getboolean("gui_settings", "use_gui"),
        "robot_urdf": os.path.join(base, config.get("file_paths", "robot_path")),
        "object_urdf": os.path.join(base, config.get("file_paths", "object_path")),
        "object_scale": config.getfloat("file_paths", "object_scale"),
    }


# シミュレーション初期化
def init_simulation(settings: dict) -> None:
    mode = p.GUI if settings["use_gui"] else p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(fixedTimeStep=1 / 240.0, numSubSteps=4)
    if settings["use_gui"]:
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=135, cameraPitch=-20, cameraTargetPosition=[0, 0, 0])


# ロボットとオブジェクトのロード
def reset_hand(settings: dict, pose: tuple) -> int:
    return p.loadURDF(settings["robot_urdf"], basePosition=pose[0], baseOrientation=pose[1], useFixedBase=True)


def reset_object(settings: dict, pose: tuple) -> int:
    return p.loadURDF(
        settings["object_urdf"], basePosition=pose[0], baseOrientation=pose[1], globalScaling=settings["object_scale"], useFixedBase=False
    )


# CSVからグリップデータ読み込み
def load_grasp_data(csv_path: str, row_index: int):
    df = pd.read_csv(csv_path)
    row = df.iloc[row_index]
    return (ast.literal_eval(row["Robot Pose"]), ast.literal_eval(row["Robot Joints"]), ast.literal_eval(row["Object Pose"]))


# オブジェクト形状情報取得
def get_obj_info(oID: int) -> tuple:
    data = p.getCollisionShapeData(oID, -1)[0]
    dims, frame_pos = data[3], data[5]
    diag = sqrt(sum(d * d for d in dims))
    return frame_pos, diag / 2


# 法線ベクトルを分割生成
def compute_normals(force_vec, force_val, sides: int, radius: float) -> list:
    fv = np.array(force_vec) * force_val
    ref = fv + np.array([1, 2, 3])
    orth = np.cross(fv, ref)
    orth = (orth / np.linalg.norm(orth)) * radius
    angle_step = 2 * pi / sides
    results = []
    for i in range(sides):
        rot = Quaternion(axis=fv, angle=angle_step * i).rotate(orth)
        vec = fv + np.array(rot)
        norm = (vec / np.linalg.norm(vec)) * (force_val / sides)
        if np.linalg.norm(norm) < 1e6:
            results.append(norm)
    return results


# GWSピラミッド拡張
def gws_extension(rID: int, oID: int, sides: int, radius: float) -> list:
    _, max_r = get_obj_info(oID)
    obj_pos, _ = p.getBasePositionAndOrientation(oID)
    ft = []
    for pt in p.getContactPoints(rID, oID):
        fv = np.array(pt[7]) * pt[9]
        if np.linalg.norm(fv) == 0:
            continue
        normals = compute_normals(pt[7], pt[9], sides, radius)
        for n in normals:
            torque = np.cross(np.array(pt[6]) - np.array(obj_pos), n) / max_r
            ft.append(np.concatenate([n, torque]))
    return ft


# ボリューム計算
def volume(ft: list) -> float:
    if len(ft) < 6:
        return 0.0
    try:
        return ConvexHull(points=ft, qhull_options="QJ").volume
    except:
        return 0.0


# イプシロン計算
def epsilon(ft: list) -> float:
    if len(ft) < 6:
        return 0.0
    try:
        hull = ConvexHull(points=ft, qhull_options="QJ")
        cent = np.mean(hull.points[hull.vertices], axis=0)
        return min(distance.euclidean(cent, hull.points[v]) for v in hull.vertices)
    except:
        return 0.0


# グリップ品質評価
def grasp_quality(rID: int, oID: int, sides: int, radius: float) -> tuple:
    if p.getContactPoints(oID, rID):
        ft = gws_extension(rID, oID, sides, radius)
        return volume(ft), epsilon(ft)
    return None, None


# スライダーセットアップ
def setup_sliders(rID: int, init_states: dict) -> list:
    sliders = []
    for i in range(p.getNumJoints(rID)):
        info = p.getJointInfo(rID, i)
        if info[2] == p.JOINT_REVOLUTE:
            init = init_states.get(i, 0)
            low = info[8] if info[8] > -1e10 else -3.14
            high = info[9] if info[9] < 1e10 else 3.14
            sliders.append((i, p.addUserDebugParameter(info[1].decode(), low, high, init)))
    return sliders


# メインループ実行
def run_sim(csv_path: str, row_index: int, settings: dict) -> None:
    p.resetSimulation()
    rp, joints, op = load_grasp_data(csv_path, row_index)
    rID = reset_hand(settings, rp)
    oID = reset_object(settings, op)
    init = {i: st[0] for i, st in joints.items()}
    for i, val in init.items():
        p.resetJointState(rID, i, val)
    print(f"Joint count: {p.getNumJoints(rID)}")
    sliders = setup_sliders(rID, init)
    p.setGravity(0, 0, -9.8)
    while True:
        p.stepSimulation()
        for i, sl in sliders:
            tp = p.readUserDebugParameter(sl)
            p.setJointMotorControl2(bodyUniqueId=rID, jointIndex=i, controlMode=p.POSITION_CONTROL, targetPosition=tp, force=settings["max_force"])
        vol, ep = grasp_quality(rID, oID, settings["pyramid_sides"], settings["pyramid_radius"])
        sys.stdout.write(f"\rEpsilon: {ep or 0:.4f} | Volume: {vol or 0:.4f}")
        sys.stdout.flush()


# エントリポイント
def main() -> None:
    cfg = os.path.join(os.path.dirname(__file__), "bh_config.ini")
    settings = load_config(cfg)
    init_simulation(settings)
    run_sim("good_grasps.csv", row_index=0, settings=settings)


if __name__ == "__main__":
    main()
