import csv
import os
import random
import threading
from configparser import ConfigParser
from math import pi, sqrt
from time import sleep, time

import astropy.coordinates
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull, distance
from transforms3d import euler

multi = 4
clients = []

# シミュレーションクライアントを動的に生成
for i in range(multi):
    if i == 0:
        client = bc.BulletClient(connection_mode=p.GUI)
    else:
        client = bc.BulletClient(connection_mode=p.DIRECT)
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    clients.append(client)


def setup_simulation(p_client):
    p_client.setPhysicsEngineParameter(fixedTimeStep=1 / 60.0, numSubSteps=4)
    p_client.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, enable=0)
    p_client.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, enable=0)
    p_client.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, enable=0)
    p_client.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=135, cameraPitch=-20, cameraTargetPosition=[0.0, 0.0, 0.0])
    p_client.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally


for client in clients:
    setup_simulation(client)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


# GLOBAL VARIABLES  - from config file
config = ConfigParser()
print(os.path.join(os.path.dirname(__file__), "bh_config_multi.ini"))
config.read(os.path.join(os.path.dirname(__file__), "bh_config_multi.ini"))

robot_path = config.get("file_paths", "robot_path")
robot_path = os.path.join(PROJECT_ROOT, robot_path)
object_path_0 = config.get("file_paths", "object_path_0")
object_path_0 = os.path.join(PROJECT_ROOT, object_path_0)
object_scale_0 = config.getfloat("file_paths", "object_scale_0")
object_path_1 = config.get("file_paths", "object_path_1")
object_path_1 = os.path.join(PROJECT_ROOT, object_path_1)
object_scale_1 = config.getfloat("file_paths", "object_scale_1")
object_path_2 = config.get("file_paths", "object_path_2")
object_path_2 = os.path.join(PROJECT_ROOT, object_path_2)
object_scale_2 = config.getfloat("file_paths", "object_scale_2")
object_path_3 = config.get("file_paths", "object_path_3")
object_path_3 = os.path.join(PROJECT_ROOT, object_path_3)
object_scale_3 = config.getfloat("file_paths", "object_scale_3")

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


# UTILIES
def extract_filename(path):
    filename_with_ext = os.path.basename(path)
    filename_without_ext = os.path.splitext(filename_with_ext)[0]

    return filename_without_ext


def rand_coord():
    rand_theta = random.uniform(-pi / 2, pi / 2)
    rand_phi = random.uniform(-pi / 2, pi / 2)
    return rand_theta, rand_phi


def add_debug_lines(p_client, rID, line_dist=0.3, line_width=500):
    """
    Use pybullet's built-in line functionality to see the z/y/z coords of the hand
    """
    p_client.addUserDebugLine([0, 0, 0], [line_dist, 0, 0], [1, 0, 0], parentObjectUniqueId=rID, parentLinkIndex=-1, lineWidth=line_width)
    p_client.addUserDebugLine([0, 0, 0], [0, line_dist, 0], [0, 1, 0], parentObjectUniqueId=rID, parentLinkIndex=-1, lineWidth=line_width)
    p_client.addUserDebugLine([0, 0, 0], [0, 0, line_dist], [0, 0, 1], parentObjectUniqueId=rID, parentLinkIndex=-1, lineWidth=line_width)


def reset_hand(p_client, rID=None, rPos=(0, 0, -init_grasp_distance), rOr=(0, 0, 0, 1), fixed=True):
    """
    move the hand back to the starting pos
    """

    if rID is None:
        rID = p_client.loadURDF(robot_path, basePosition=(0, 0, 0), baseOrientation=(0, 0, 0, 1), useFixedBase=fixed, globalScaling=1)
        p_client.changeDynamics(rID, -1, mass=5.0)
    else:
        p_client.resetBasePositionAndOrientation(rID, rPos, rOr)
    if debug_lines:
        add_debug_lines(p_client, rID)
    return rID


def reset_ob(p_client, oID=None, oPos=(0, 0, 0), fixed=True):
    """
    reset by deleting
    """
    if oID is not None:
        p_client.removeBody(oID)

    client_id = p_client._client

    object_path_var = f"object_path_{client_id}"
    object_scale_var = f"object_scale_{client_id}"

    object_path = globals().get(object_path_var)
    object_scale = globals().get(object_scale_var)

    if object_path is None or object_scale is None:
        raise ValueError(f"Invalid client ID or missing variable for client ID: {client_id}")

    oID = p_client.loadURDF(object_path, oPos, globalScaling=object_scale, useFixedBase=fixed)

    return oID


"""#####################################################################################################################
                                            HAND ORIENTATION + LOCATION
#####################################################################################################################"""


# TODO: make these more user specifiable (range of angles w reasonable defaults)


def hand_dist(p_client, oID, rID, pos, oren):
    """
    actually does tthe movement to have hand touch object to judge distance

    returns position of the hand when it touches the object
    """
    # print("reset hand for non-fixed base")
    reset_hand(p_client, rID, rPos=pos, rOr=oren, fixed=False)
    relax(p_client, rID)  # want fingers splayed to get distance
    force_vector = np.array(pos) * -speed_find_distance

    has_contact = 0
    while not has_contact:  # while still distance between hand/object
        p_client.applyExternalForce(rID, -1, force_vector, pos, p.WORLD_FRAME)
        p_client.stepSimulation()
        has_contact = len(p_client.getContactPoints(rID, oID))
    t_pos, t_oren = p_client.getBasePositionAndOrientation(rID)
    return t_pos  # only need the position of the object


def adjust_point_dist(p_client, theta_rad, phi_rad, rID, oID, carts, quat):
    """
    move the hand w/fingers splayed until it touches the object
    should touch in center/palm - this should be the best for an initial grasp

    returns set of position coordinates representing how far from the object the hand should be (touching + a margin)
    """

    t_pos = hand_dist(p_client, oID, rID, carts, quat)
    t_dist = distance.euclidean(t_pos, [0, 0, 0])
    m_dist = t_dist + grasp_distance_margin
    carts = astropy.coordinates.spherical_to_cartesian(m_dist, theta_rad, phi_rad)
    flip_carts = np.array(carts) * -1  # adjust to face obj

    return flip_carts


def get_given_point(p_client, dist, theta_rad, phi_rad, rID, oID):
    """
    For the transform3d euler to quat: (seems like their z is our x, their y is our y, their x is our z)
    Rotate about the current z-axis by ϕ. Then, rotate about the new y-axis by θ
    """
    carts = astropy.coordinates.spherical_to_cartesian(dist, theta_rad, phi_rad)
    flip_carts = np.array(carts) * -1  # adjust to face obj
    quat = euler.euler2quat(phi_rad + pi, pi / 2 - theta_rad, pi, axes="sxyz")  # pi in the z to face "up"
    close_carts = adjust_point_dist(p_client, theta_rad, phi_rad, rID, oID, flip_carts, quat)  # find dist to grasp

    return (close_carts, quat)


def sphere_set(p_client, rID, oID, phi_init=pi, phi_span=2 * pi, theta_init=pi / 2, theta_span=pi / 2):
    """
    move the hand around the object in a reasonable way
    returns an array of (position, orientation) pairs
    """
    set = []

    phi = phi_init
    increment_phi = phi_span / num_grasps_per_cycle

    theta = theta_init
    increment_theta = theta_span / num_cycles_to_grasp

    for theta_i in range(0, (num_cycles_to_grasp + 1)):
        for phi_i in range(0, (num_grasps_per_cycle)):
            point = get_given_point(
                p_client,
                dist=init_grasp_distance,
                theta_rad=(-theta) + increment_theta * theta_i,
                phi_rad=(-phi) + increment_phi * phi_i,
                rID=rID,
                oID=oID,
            )
            set.append(point)

    return set


def rand_set(rID, oID, dist=init_grasp_distance, n=10):
    """
    get n pairs for the hand randomly distributed dist away from the origin
    returns an array of (position, orientation) pairs
    TODO: make this more uniformly distributed https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    """
    set = []

    for i in range(n):
        theta_rad, phi_rad = rand_coord()
        set.append(get_given_point(dist, theta_rad, phi_rad, rID, oID))

    return set


def wrist_rotations(pose):
    """
    rotates the wrist of hand in place, increasing number of grasp possibilities for one position
    """

    rotated_poses = []
    point = pose[0]
    quat = pose[1]
    # change to from xyzw to wxyz
    quat_w = quat[3]
    quat_x = quat[0]
    quat_y = quat[1]
    quat_z = quat[2]
    current_quat = Quaternion(quat_w, quat_x, quat_y, quat_z)
    rot_iter = (2 * pi) / num_wrist_rotations
    for i in range(0, num_wrist_rotations):
        rot_quat = Quaternion(axis=np.array(point) * -1, radians=(pi / 2) + (rot_iter * i))
        delta_quat = rot_quat * current_quat
        pyb_quat = (delta_quat[1], delta_quat[2], delta_quat[3], delta_quat[0])
        rotated_poses.append((point, pyb_quat))
    return rotated_poses


"""#####################################################################################################################
                                            GRIPPER FUNCTIONS/MOVEMENT
#####################################################################################################################"""


def grasp(p_client, handId):
    """
    closes the gripper uniformly + attempts to find a grasp
    this is based on time + not contact points because contact points could just be a finger poking the object
    relies on grip_joints - specified by user/config file which joints should close
    """
    finish_time = time() + grasp_time_limit
    while time() < finish_time:
        p_client.stepSimulation()
        for joint in active_grasp_joints:
            if joint == 3:
                p_client.setJointMotorControl2(
                    bodyUniqueId=handId,
                    jointIndex=joint,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=target_grasp_velocity,
                    force=max_grasp_force,
                )
            else:
                p_client.setJointMotorControl2(
                    bodyUniqueId=handId,
                    jointIndex=joint,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=target_grasp_velocity,
                    force=max_grasp_force / 2.0,
                )


def relax(p_client, rID):
    """
    return all joints to neutral/furthest extended, based on urdf specification
    """
    joint = 0
    num = p_client.getNumJoints(rID)
    while joint < num:
        p_client.resetJointState(rID, jointIndex=joint, targetValue=0.0)
        joint = joint + 1


"""#####################################################################################################################
                                        POSITION/ORIENTATION DATA - GRAP MEMORY
#####################################################################################################################"""


class Grasp:

    def __init__(self, robot_position, robot_orientation, robot_joints, final_object_position, final_object_orientation, vol, ep):
        self.robot_pose = (robot_position, robot_orientation)
        self.robot_joints = robot_joints
        self.final_object_pose = (final_object_position, final_object_orientation)
        self.vol = vol
        self.ep = ep

    def __repr__(self):
        return (
            "ROBOT: Pose: "
            + str(self.robot_pose)
            + " , Joints: "
            + str(self.robot_joints)
            + " OBJECT: Pose: "
            + str(self.final_object_pose)
            + " QUALITY: Volume"
            + str(self.vol)
            + " , Epsilon: "
            + str(self.vol)
            + " "
        )

    def __str__(self):
        return (
            "ROBOT: Pose: "
            + str(self.robot_pose)
            + " , Joints: "
            + str(self.robot_joints)
            + " OBJECT: Pose: "
            + str(self.final_object_pose)
            + " QUALITY: Volume"
            + str(self.vol)
            + " , Epsilon: "
            + str(self.vol)
            + " "
        )


def get_robot_config(p_client, rID, oID):
    r_pos, r_oren = p_client.getBasePositionAndOrientation(rID)
    joints = {}
    num = p_client.getNumJoints(rID)
    for joint in range(0, num):
        joints[joint] = p_client.getJointState(rID, joint)
    o_pos, o_oren = p_client.getBasePositionAndOrientation(oID)
    vol, ep = grip_qual(p_client, rID, oID)
    return Grasp(r_pos, r_oren, joints, o_pos, o_oren, vol, ep)


"""#####################################################################################################################
                                            GRASP EVALUATION FUNCTIONS
#####################################################################################################################"""


def check_grip(p_client, oID, rID):
    """
    check grip by adding in gravity
    """
    # print("checking strength of current grip")
    mass = 0.1
    mag = 9.8 * mass
    pos, oren = p_client.getBasePositionAndOrientation(rID)
    time_limit = 0.5
    finish_time = time() + time_limit
    p_client.addUserDebugText("Grav Check!", [-0.07, 0.07, 0.07], textColorRGB=[0, 0, 1], textSize=1)
    while time() < finish_time:
        p_client.stepSimulation()
        p_client.applyExternalForce(oID, linkIndex=-1, forceObj=[0, 0, -mag], posObj=pos, flags=p.WORLD_FRAME)
    contact = p_client.getContactPoints(oID, rID)  # see if hand is still holding obj after gravity is applied
    if len(contact) > 0:
        p_client.removeAllUserDebugItems()
        p_client.addUserDebugText("Grav Check Passed!", [-0.07, 0.07, 0.07], textColorRGB=[0, 1, 0], textSize=1)
        print("Grav Check Passed")
        sleep(0.2)
        return get_robot_config(p_client, rID, oID)
    else:
        p_client.removeAllUserDebugItems()
        p_client.addUserDebugText("Grav Check Failed!", [-0.07, 0.07, 0.07], textColorRGB=[1, 0, 0], textSize=1)
        print("Grav Check Failed")
        sleep(0.2)
        return None


def grip_qual(p_client, oID, rID):
    """
    evaluate the grasp quality
    """
    contact = p_client.getContactPoints(oID, rID)  # see if hand is still holding obj after gravity is applied
    if len(contact) > 0:
        force_torque = gws_pyramid_extension(p_client, rID, oID)
        if not force_torque:
            vol = None
            ep = None
        else:
            force_torque = np.array(force_torque) + np.random.normal(scale=1e-6, size=np.array(force_torque).shape)
            vol = volume(force_torque)
            ep = eplison(force_torque)
    else:
        vol = None
        ep = None
    return vol, ep


def get_obj_info(p_client, oID):  # TODO: what about not mesh objects?
    """
    get object data to figure out how far away the hand needs to be to make its approach
    """
    obj_data = p_client.getCollisionShapeData(oID, -1)[0]
    geometry_type = obj_data[2]
    # print("geometry type: " + str(geometry_type))
    dimensions = obj_data[3]
    # print("dimensions: "+ str(dimensions))
    local_frame_pos = obj_data[5]
    # print("local frome position: " + str(local_frame_pos))
    local_frame_orn = obj_data[6]
    # print("local frame oren: " + str(local_frame_orn))
    diagonal = sqrt(dimensions[0] ** 2 + dimensions[1] ** 2 + dimensions[2] ** 2)
    # print("diagonal: ", diagonal)
    max_radius = diagonal / 2
    return local_frame_pos, max_radius


def gws(p_client, rID, oID):
    """
    calculate force/torque vectors for use in evaluation

    """
    print("eval gws")
    local_frame_pos, max_radius = get_obj_info(p_client, oID)
    # sim uses center of mass as a reference for the Cartesian world transforms in getBasePositionAndOrientation
    obj_pos, obj_orn = p_client.getBasePositionAndOrientation(oID)
    force_torque = []
    contact_points = p_client.getContactPoints(rID, oID)
    for point in contact_points:
        contact_pos = point[6]
        normal_vector_on_obj = point[7]
        normal_force_on_obj = point[9]
        force_vector = np.array(normal_vector_on_obj) * normal_force_on_obj

        radius_to_contact = np.array(contact_pos) - np.array(obj_pos)
        torque_numerator = np.cross(radius_to_contact, force_vector)
        torque_vector = torque_numerator / max_radius

        force_torque.append(np.concatenate([force_vector, torque_vector]))

    return force_torque


def get_new_normals(force_vector, normal_force, sides, radius):
    """
    utility function to help with GWS/pyramid extension for contact points
    """
    return_vectors = []
    # get arbitrary vector to get cross product which should be orthogonal to both
    vector_to_cross = np.array((force_vector[0] + 1, force_vector[1] + 2, force_vector[2] + 3))
    orthg = np.cross(force_vector, vector_to_cross)
    orthg_vector = (orthg / np.linalg.norm(orthg)) * radius
    rot_angle = (2 * pi) / sides
    split_force = normal_force / sides

    for side_num in range(0, sides):
        rotated_orthg = Quaternion(axis=force_vector, angle=(rot_angle * side_num)).rotate(orthg_vector)
        new_vect = force_vector + np.array(rotated_orthg)
        norm_vect = (new_vect / np.linalg.norm(new_vect)) * split_force
        return_vectors.append(norm_vect)

    return return_vectors


def gws_pyramid_extension(p_client, rID, oID, pyramid_sides=force_pyramid_sides, pyramid_radius=force_pyramid_radius):
    # often dont have enough contact points to create a qhull of the right dimensions, so create more that are very close to the existing ones
    local_frame_pos, max_radius = get_obj_info(p_client, oID)
    # sim uses center of mass as a reference for the Cartesian world transforms in getBasePositionAndOrientation
    obj_pos, obj_orn = p_client.getBasePositionAndOrientation(oID)
    force_torque = []
    contact_points = p_client.getContactPoints(rID, oID)
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
    get qhull of the 6 dim vectors [fx, fy, fz, tx, ty, tz] created by gws (from contact points)
    get the volume
    """
    if len(force_torque) < 3:
        print("Error: force_torque does not have enough points for ConvexHull")
        print("force_torque:", force_torque)
        return None  # または、適切なエラーハンドリングを追加

    try:
        vol = ConvexHull(points=force_torque)
    except Exception as e:
        print("ConvexHull computation failed:", e)
        print("force_torque:", force_torque)
        return None

    return vol.volume


def eplison(force_torque):
    """
    get qhull of the 6 dim vectors [fx, fy, fz, tx, ty, tz] created by gws (from contact points)
    get the distance from centroid of the hull to the closest vertex
    """
    hull = ConvexHull(points=force_torque)
    centroid = []
    for dim in range(0, 6):
        centroid.append(np.mean(hull.points[hull.vertices, dim]))
    shortest_distance = 500000000
    closest_point = None
    for point in force_torque:
        point_dist = distance.euclidean(centroid, point)
        if point_dist < shortest_distance:
            shortest_distance = point_dist
            closest_point = point

    return shortest_distance


def round_grip_data(grip, decimal_places):
    rounded_robot_pose = (
        tuple(round(coord, decimal_places) for coord in grip.robot_pose[0]),
        tuple(round(quat, decimal_places) for quat in grip.robot_pose[1]),
    )

    rounded_robot_joints = {joint: (round(state[0], decimal_places), round(state[1], decimal_places)) for joint, state in grip.robot_joints.items()}

    rounded_object_pose = (
        tuple(round(coord, decimal_places) for coord in grip.final_object_pose[0]),
        tuple(round(quat, decimal_places) for quat in grip.final_object_pose[1]),
    )

    rounded_vol = round(grip.vol, decimal_places) if grip.vol is not None else None
    rounded_ep = round(grip.ep, decimal_places) if grip.ep is not None else None

    return rounded_robot_pose, rounded_robot_joints, rounded_object_pose, rounded_vol, rounded_ep


"""#####################################################################################################################
                                        MAIN MAIN MAIN MAIN MAIN MAIN
#####################################################################################################################"""


def run_simulation(client):

    rID = reset_hand(client)
    oID = reset_ob(client)

    hand_set = sphere_set(client, rID=rID, oID=oID)

    client.changeDynamics(rID, -1, mass=0.0)
    oID = reset_ob(client, oID, [0, 0, 0])

    good_grips = []

    pos = 0
    for pose in hand_set:
        poses = []
        poses.append(pose)
        if use_wrist_rotations:
            rotated_poses = wrist_rotations(pose)
            poses = poses + rotated_poses

        for pose in poses:
            print(" ")
            print("Pose #: ", pos)
            relax(client, rID)
            client.removeAllUserDebugItems()
            client.resetBasePositionAndOrientation(rID, pose[0], pose[1])
            if debug_lines:
                add_debug_lines(client, rID)
            oID = reset_ob(client, oID, [0, 0, 0], fixed=False)
            grasp(client, rID)
            vol, ep = grip_qual(client, oID, rID)
            print("Volume: ", vol)
            print("Epslion: ", ep)
            good_grips.append(check_grip(client, oID, rID))
            pos += 1

    print("Num Good Grips: ", len(good_grips))
    print("Grips:")
    for grip in good_grips:
        print(grip)

    decimal_places = 5

    client_id = client._client
    object_path_var = f"object_path_{client_id}"
    object_path = globals().get(object_path_var)
    filename = extract_filename(object_path)

    # CSVファイルに保存するコード
    with open(f"good_grasps_{filename}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Robot Pose", "Robot Joints", "Object Pose", "Quality Volume", "Quality Epsilon"])  # ヘッダー

        for grip in good_grips:
            if grip is not None:
                rounded_robot_pose, rounded_robot_joints, rounded_object_pose, rounded_vol, rounded_ep = round_grip_data(grip, decimal_places)
                writer.writerow([rounded_robot_pose, rounded_robot_joints, rounded_object_pose, rounded_vol, rounded_ep])
            print(grip)


threads = []
for client in clients:
    thread = threading.Thread(target=run_simulation, args=(client,))
    threads.append(thread)
    thread.start()

# 全スレッドの終了を待つ
for thread in threads:
    thread.join()
