import csv
import os
import random
from configparser import ConfigParser
from math import pi, sqrt
from time import sleep, time

import astropy.coordinates
import numpy as np
import pybullet as p
import pybullet_data
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull, distance
from transforms3d import euler

# プロジェクトルート
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
# 設定ファイル
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "bh_config.ini")

# 設定読み込み
config = ConfigParser()
config.read(CONFIG_PATH)

# ファイルパス
robot_path = os.path.join(PROJECT_ROOT, config.get("file_paths", "robot_path"))
object_path = os.path.join(PROJECT_ROOT, config.get("file_paths", "object_path"))
object_scale = config.getfloat("file_paths", "object_scale")

# 把持パラメータ
init_grasp_distance = config.getfloat("grasp_settings", "init_grasp_distance")
speed_find_distance = config.getfloat("grasp_settings", "speed_find_distance")
grasp_distance_margin = config.getfloat("grasp_settings", "grasp_distance_margin")
max_grasp_force = config.getfloat("grasp_settings", "max_grasp_force")
target_grasp_velocity = config.getfloat("grasp_settings", "target_grasp_velocity")
grasp_time_limit = config.getfloat("grasp_settings", "grasp_time_limit")
active_grasp_joints = [int(j) for j in config.get("grasp_settings", "active_grasp_joints").split(",")]
num_grasps_per_cycle = config.getint("grasp_settings", "num_grasps_per_cycle")
num_cycles_to_grasp = config.getint("grasp_settings", "num_cycles_to_grasp")
num_wrist_rotations = config.getint("grasp_settings", "num_wrist_rotations")
use_wrist_rotations = config.getboolean("grasp_settings", "use_wrist_rotations")

# 評価パラメータ
force_pyramid_sides = config.getint("eval_settings", "force_pyramid_sides")
force_pyramid_radius = config.getfloat("eval_settings", "force_pyramid_radius")

# GUI設定
use_gui = config.getboolean("gui_settings", "use_gui")
debug_lines = config.getboolean("gui_settings", "debug_lines")
debug_text = config.getboolean("gui_settings", "debug_text")

# PyBullet初期化
physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setPhysicsEngineParameter(fixedTimeStep=1 / 240.0, numSubSteps=4)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
p.resetDebugVisualizerCamera(0.5, 135, -20, [0, 0, 0])


# ユーティリティ関数
def rand_coord():
    return random.uniform(-pi / 2, pi / 2), random.uniform(-pi / 2, pi / 2)


def add_debug_lines(body_id, length=0.3, width=500):
    p.addUserDebugLine([0, 0, 0], [length, 0, 0], [1, 0, 0], body_id, -1, width)
    p.addUserDebugLine([0, 0, 0], [0, length, 0], [0, 1, 0], body_id, -1, width)
    p.addUserDebugLine([0, 0, 0], [0, 0, length], [0, 0, 1], body_id, -1, width)


# リセット関数


def reset_hand(rID=None, pos=(0, 0, -init_grasp_distance), orn=(0, 0, 0, 1), fixed=True):
    if rID is None:
        rID = p.loadURDF(robot_path, basePosition=(0, 0, 0), baseOrientation=(0, 0, 0, 1), useFixedBase=fixed)
        p.changeDynamics(rID, -1, mass=5.0)
    else:
        p.resetBasePositionAndOrientation(rID, pos, orn)
    if debug_lines:
        add_debug_lines(rID)
    return rID


def reset_ob(oID=None, pos=(0, 0, 0), fixed=True):
    if oID is not None:
        p.removeBody(oID)
    return p.loadURDF(object_path, pos, globalScaling=object_scale, useFixedBase=fixed)


# 距離計測・調整


def hand_dist(oID, rID, pos, orn):
    reset_hand(rID, pos, orn, True)
    relax(rID)
    force = -np.array(pos) * speed_find_distance
    while not p.getContactPoints(rID, oID):
        p.applyExternalForce(rID, -1, force, pos, p.WORLD_FRAME)
        p.stepSimulation()
    return p.getBasePositionAndOrientation(rID)[0]


def adjust_point_dist(theta, phi, rID, oID, coords, quat):
    touch = hand_dist(oID, rID, coords, quat)
    dist = distance.euclidean(touch, [0, 0, 0]) + grasp_distance_margin
    cart = astropy.coordinates.spherical_to_cartesian(dist, theta, phi)
    return -np.array(cart)


def get_given_point(dist, theta, phi, rID, oID):
    cart = astropy.coordinates.spherical_to_cartesian(dist, theta, phi)
    base = -np.array(cart)
    quat = euler.euler2quat(phi + pi, pi / 2 - theta, pi, axes="sxyz")
    adj = adjust_point_dist(theta, phi, rID, oID, base, quat)
    return adj, quat


# サンプリング関数


def manual_set():
    p1 = np.array([0.0647, 0.006, 0])
    q1 = np.array([4.32978028e-17, -0.707106781, 4.32978028e-17, 0.707106781])
    return [(p1, q1)]


def sphere_set(rID, oID):
    poses = []
    for ti in range(num_cycles_to_grasp + 1):
        for pj in range(num_grasps_per_cycle):
            theta = -pi / 2 + (pi / 2) * ti / num_cycles_to_grasp
            phi = -pi + 2 * pi * pj / num_grasps_per_cycle
            poses.append(get_given_point(init_grasp_distance, theta, phi, rID, oID))
    return poses


def rand_set(rID, oID, n=10):
    return [get_given_point(init_grasp_distance, *rand_coord(), rID, oID) for _ in range(n)]


# ジョイント操作


def reset_initial_positions(robot_id, init_positions):
    for idx, ang in init_positions.items():
        p.resetJointState(robot_id, idx, ang)


def wrist_rotations(pose):
    poses = []
    p0, q0 = pose
    q = Quaternion(q0[3], q0[0], q0[1], q0[2])
    for i in range(num_wrist_rotations):
        rot = Quaternion(axis=-np.array(p0), radians=pi / 2 + 2 * pi * i / num_wrist_rotations)
        dq = rot * q
        poses.append((p0, (dq[1], dq[2], dq[3], dq[0])))
    return poses


# グリップ


def grasp(handId):
    end = time() + grasp_time_limit
    while time() < end:
        p.stepSimulation()
        for j in active_grasp_joints:
            p.setJointMotorControl2(handId, j, p.VELOCITY_CONTROL, targetVelocity=target_grasp_velocity, force=max_grasp_force)


def relax(rID):
    for j in range(p.getNumJoints(rID)):
        p.resetJointState(rID, j, 0.0)


# 評価用クラス
class Grasp:
    def __init__(self, r_pose, r_joints, o_pose, vol, ep):
        self.robot_pose = r_pose
        self.robot_joints = r_joints
        self.final_object_pose = o_pose
        self.vol = vol
        self.ep = ep

    def __repr__(self):
        return f"Grasp(r_pose={self.robot_pose}, joints={self.robot_joints}, o_pose={self.final_object_pose}, vol={self.vol}, ep={self.ep})"


def get_robot_config(rID, oID):
    rp, ro = p.getBasePositionAndOrientation(rID)
    joints = {i: p.getJointState(rID, i) for i in range(p.getNumJoints(rID))}
    op, oo = p.getBasePositionAndOrientation(oID)
    vol, ep = grip_qual(oID, rID)
    return Grasp((rp, ro), joints, (op, oo), vol, ep)


# 評価関数


def check_grip(oID, rID):
    p.addUserDebugText("Grav Check!", [-0.07] * 3, textSize=1)
    p.setGravity(0, 0, -9.8)
    t = time() + 2
    lost = False
    while time() < t:
        p.stepSimulation()
        if not p.getContactPoints(oID, rID):
            lost = True
            break
    p.setGravity(0, 0, 0)
    p.removeAllUserDebugItems()
    if lost:
        print("Grav Check Failed")
        sleep(0.2)
        return None
    print("Grav Check Passed")
    sleep(0.2)
    return get_robot_config(rID, oID)


def grip_qual(oID, rID):
    pts = p.getContactPoints(oID, rID)
    if not pts:
        return None, None
    ft = gws_pyramid_extension(rID, oID)
    return volume(ft), epsilon(ft)


# 接触力／トルク計算


def get_obj_info(oID):
    data = p.getCollisionShapeData(oID, -1)[0]
    dims = data[3]
    radius = sqrt(sum(d * d for d in dims)) / 2
    return data[5], radius


def get_new_normals(vec, f, n_sides, radius):
    base = np.array((vec[0] + 1, vec[1] + 2, vec[2] + 3))
    ort = np.cross(vec, base)
    ort = ort / np.linalg.norm(ort) * radius
    angle = 2 * pi / n_sides
    for i in range(n_sides):
        r = Quaternion(axis=vec, angle=i * angle).rotate(ort)
        yield (vec + r) / np.linalg.norm(vec + r) * (f / n_sides)


def gws_pyramid_extension(rID, oID):
    _, rad = get_obj_info(oID)
    op, _ = p.getBasePositionAndOrientation(oID)
    result = []
    for pt in p.getContactPoints(rID, oID):
        vec = np.array(pt[7]) * pt[9]
        if np.linalg.norm(vec) == 0:
            continue
        for nvec in get_new_normals(vec, pt[9], force_pyramid_sides, force_pyramid_radius):
            torque = np.cross(np.array(pt[6]) - np.array(op), nvec) / rad
            result.append(np.concatenate([nvec, torque]))
    return result


def volume(ft):
    if len(ft) < 6:
        return 0.0
    try:
        return ConvexHull(ft, qhull_options="QJ").volume
    except:
        return 0.0


def epsilon(ft):
    if len(ft) < 6:
        return 0.0
    hull = ConvexHull(ft, qhull_options="QJ")
    verts = hull.points[hull.vertices]
    cen = np.mean(verts, axis=0)
    return min(distance.euclidean(cen, v) for v in verts)


# データ丸め


def round_grip_data(grip, dec):
    rp = tuple(round(x, dec) for x in grip.robot_pose[0]), tuple(round(q, dec) for q in grip.robot_pose[1])
    jn = {i: (round(s[0], dec), round(s[1], dec)) for i, s in grip.robot_joints.items()}
    op = tuple(round(x, dec) for x in grip.final_object_pose[0]), tuple(round(q, dec) for q in grip.final_object_pose[1])
    return rp, jn, op, round(grip.vol, dec), round(grip.ep, dec)


# メイン処理
rID = reset_hand()
oID = reset_ob()
hand_set = manual_set()  # sphere_set(), rand_set() に切替可
p.changeDynamics(rID, -1, mass=0.0)
oID = reset_ob(oID)
good_grasps = []
pos = 0
init_positions = {8: 1.57}
for base_pose in hand_set:
    poses = [base_pose]
    if use_wrist_rotations:
        poses += wrist_rotations(base_pose)
    for p0 in poses:
        print("\nPose #", pos)
        relax(rID)
        p.removeAllUserDebugItems()
        p.resetBasePositionAndOrientation(rID, p0[0], p0[1])
        if debug_lines:
            add_debug_lines(rID)
        oID = reset_ob(oID, (0, 0, 0), fixed=False)
        reset_initial_positions(rID, init_positions)
        grasp(rID)
        print("Volume:", grip_qual(oID, rID)[0], "Epsilon:", grip_qual(oID, rID)[1])
        good_grasps.append(check_grip(oID, rID))
        pos += 1

print("Num Good Grips:", len(good_grasps))
with open("good_grasps.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Robot Pose", "Robot Joints", "Object Pose", "Quality Volume", "Quality Epsilon"])
    for g in good_grasps:
        if g:
            w.writerow(round_grip_data(g, 5))
