import pybullet as p
import pybullet_data

# PyBullet初期化
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# URDFをロード
robot_id = p.loadURDF("RobotURDFs/wbr_description/urdf/wbr_hand.urdf")

# シミュレーション設定
# p.setGravity(0, 0, -9.8)
p.setGravity(0, 0, 0)
p.setTimeStep(1.0 / 240.0)

# ジョイント情報を取得
num_joints = p.getNumJoints(robot_id)
print(f"Number of joints: {num_joints}")

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

# シミュレーションループ
while True:
    for joint_index, slider in sliders:
        target_position = p.readUserDebugParameter(slider)
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_position,
            force=500,  # モーターの出力を設定
        )
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
