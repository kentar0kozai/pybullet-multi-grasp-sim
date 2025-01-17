import pybullet as p
import pybullet_data

# PyBullet初期化
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# URDFをロード
robot_id = p.loadURDF("RobotURDFs/wbr_description/urdf/wbr_hand_with_cap.urdf", baseOrientation=[0, -0.7068252, 0, 0.7073883], useFixedBase=True)
cap_id = p.loadURDF("ObjectURDFs/cylinder/cap.urdf", useFixedBase=False)
cylinder_id = p.loadURDF("ObjectURDFs/cylinder/cylinder.urdf", basePosition=[-0.05, 0, 0], useFixedBase=True)
# robot_id = p.loadURDF("RobotURDFs/finger_description/urdf/RH8D.urdf", useFixedBase=True)
# robot_id = p.loadURDF("RobotURDFs/barrett_hand_description/urdf/bh.urdf")

# シミュレーション設定
# p.setGravity(0, 0, -9.8)
p.setGravity(0, 0, 0)
p.setTimeStep(1.0 / 240.0)

# ジョイント情報を取得
num_joints = p.getNumJoints(robot_id)
print(f"Number of joints: {num_joints}")

print(f"Base link index: -1 (base link)")

for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    link_name = joint_info[12].decode("utf-8")
    print(f"Link index: {i}, Link name: {link_name}")


p.createConstraint(
    parentBodyUniqueId=cap_id,
    parentLinkIndex=-1,  # thumb_4 のリンク番号
    childBodyUniqueId=robot_id,
    childLinkIndex=12,  # cap のリンク番号
    jointType=p.JOINT_FIXED,
    jointAxis=[1, 1, 1],
    parentFramePosition=[-0.017, 0.0, 0],
    parentFrameOrientation=[-0.7071067, 0, 0, 0.7071069],
    childFramePosition=[0.0, 0.0, 0],
)

p.createConstraint(
    parentBodyUniqueId=cap_id,
    parentLinkIndex=-1,  # thumb_4 のリンク番号
    childBodyUniqueId=robot_id,
    childLinkIndex=4,  # cap のリンク番号
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0.018, 0.0, -0.0],
    parentFrameOrientation=[0, -0.7071067, 0, 0.7071069],
    childFramePosition=[0.0, 0.0, 0],
)

# p.createConstraint(
#     parentBodyUniqueId=cap_id,
#     parentLinkIndex=1,  # thumb_4 のリンク番号
#     childBodyUniqueId=robot_id,
#     childLinkIndex=4,  # cap のリンク番号
#     jointType=p.JOINT_FIXED,
#     jointAxis=[0, 0, 1],
#     parentFramePosition=[0.0, 0.0, 0],
#     childFramePosition=[0.0, 0.0, 0],
# )

# スライダーを使ったデバッグパラメータを追加
sliders = []
for joint_index in range(num_joints):
    joint_info = p.getJointInfo(robot_id, joint_index)
    joint_name = joint_info[1].decode("utf-8")
    joint_type = joint_info[2]

    # Revoluteジョイントのみにスライダーを追加
    if joint_type == p.JOINT_REVOLUTE:
        slider = p.addUserDebugParameter(joint_name, -1.57, 1.57, 0)
        sliders.append((joint_index, slider))

line_ids = []

# シミュレーションループ
while True:

    for joint_index, slider in sliders:
        target_position = p.readUserDebugParameter(slider)
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_position,
            force=0.3,  # モーターの出力を設定
        )

    # p.removeAllUserDebugItems()
    # # リンクの情報を取得
    # num_joints = p.getNumJoints(cap_id)
    # for i in range(num_joints):
    #     # 各リンクのワールド座標系での位置と姿勢を取得
    #     joint_state = p.getLinkState(cap_id, i, computeForwardKinematics=True)
    #     joint_pos = joint_state[4]  # ワールド座標系でのリンクの位置
    #     joint_orn = joint_state[5]  # ワールド座標系でのリンクのクォータニオン

    #     # 回転を行列表現に変換 (回転行列を逆向きに適用する)
    #     joint_rot_matrix = p.getMatrixFromQuaternion(joint_orn)

    #     # 転置行列を計算（逆回転を適用するために転置を使う）
    #     joint_rot_matrix_transposed = [
    #         joint_rot_matrix[0],
    #         joint_rot_matrix[3],
    #         joint_rot_matrix[6],
    #         joint_rot_matrix[1],
    #         joint_rot_matrix[4],
    #         joint_rot_matrix[7],
    #         joint_rot_matrix[2],
    #         joint_rot_matrix[5],
    #         joint_rot_matrix[8],
    #     ]

    #     # 各軸の方向を計算
    #     x_axis = [joint_rot_matrix_transposed[0], joint_rot_matrix_transposed[1], joint_rot_matrix_transposed[2]]
    #     y_axis = [joint_rot_matrix_transposed[3], joint_rot_matrix_transposed[4], joint_rot_matrix_transposed[5]]
    #     z_axis = [joint_rot_matrix_transposed[6], joint_rot_matrix_transposed[7], joint_rot_matrix_transposed[8]]

    #     # スケール
    #     axis_length = 0.1

    #     # 各軸を描画
    #     p.addUserDebugLine(
    #         joint_pos,
    #         [joint_pos[0] + axis_length * x_axis[0], joint_pos[1] + axis_length * x_axis[1], joint_pos[2] + axis_length * x_axis[2]],
    #         [1, 0, 0],  # X軸（赤）
    #         2,
    #     )
    #     p.addUserDebugLine(
    #         joint_pos,
    #         [joint_pos[0] + axis_length * y_axis[0], joint_pos[1] + axis_length * y_axis[1], joint_pos[2] + axis_length * y_axis[2]],
    #         [0, 1, 0],  # Y軸（緑）
    #         2,
    #     )
    #     p.addUserDebugLine(
    #         joint_pos,
    #         [joint_pos[0] + axis_length * z_axis[0], joint_pos[1] + axis_length * z_axis[1], joint_pos[2] + axis_length * z_axis[2]],
    #         [0, 0, 1],  # Z軸（青）
    #         2,
    #     )

    p.stepSimulation()


# import mujoco
# import mujoco_viewer

# # モデルを読み込む
# model = mujoco.MjModel.from_xml_path("single_hand.xml")
# data = mujoco.MjData(model)

# # ビューアーで表示
# viewer = mujoco_viewer.MujocoViewer(model, data)
# while viewer.is_alive:
#     mujoco.mj_step(model, data)
#     viewer.render()
# viewer.close()
