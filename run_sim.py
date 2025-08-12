import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import genesis as gs
import numpy as np
import pandas as pd

VIDEO_PATH = "videos"
FILE_NAME = "joint_angles_XZY_Moh"
FILE_PATH = "angle_arrays/joint_angles_XZY_Moh.csv"
URDF = "Wiki-GRx-Models-master/GRX/GR1/GR1T2/urdf/GR1T2_nohand.urdf"
gs.init(backend=gs.gpu)

# sim_opts = gs.options.SimOptions(gravity=(0.0, -9.81, 0.0))

scene = gs.Scene(
    # sim_options=sim_opts,
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(-3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=120,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer()
)

robot = scene.add_entity(
    gs.morphs.URDF(
        file=URDF,
        fixed=False,
        visualization=True,
        collision=True,
        requires_jac_and_IK=False,
        scale=1.0,
    )
)

cam = scene.add_camera(
    res=(1280, 960),
    pos=(5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=30,
    GUI=False
)

plane = scene.add_entity(gs.morphs.Plane(), vis_mode="collision")

scene.build()

df = pd.read_csv(FILE_PATH)

dof_names = [
      'left_hip_roll_joint', 'right_hip_roll_joint',
      'waist_yaw_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint',
      'waist_pitch_joint', 'left_hip_pitch_joint', 'right_hip_pitch_joint',
      'waist_roll_joint',
      'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
      'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
      'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
      'left_elbow_pitch_joint', 'right_elbow_pitch_joint'
]

dof_idx_map = {name: i for i, name in enumerate(dof_names)}

cam.start_recording()
robot.set_pos([0, 0, 0])
qpos = np.zeros(robot.get_qpos().shape[0])
y_offset = 0.75
qpos[0:3]   = [0.0, 0.0, 0.9]
qpos[3:7]   = [-1, 0.0, 0, 0]
# qpos[3:7] = [1, 0, 0, 0]
# for joint in robot.joints: 
#     print(f"name: {joint.name}, dof_start: {joint.dof_start}, dof_end: {joint.dof_end}, {joint.idx}, type: {joint.type}")
# for _, row in df.iterrows():
#     # qpos[0:3] = row[['x', 'y', 'z']].values
#     # qpos[3:7] = row[['qw', 'qx', 'qy', 'qz']].values
#     # qpos[2]  += y_offset
#     for dof_name, dof_idx in dof_idx_map.items():
#         if dof_name in row:
#             # print(f"dof_idx {dof_idx}, dof_name: {dof_name}")
#             qpos[dof_idx + 7] = row[dof_name] 
#         else: 
#             print("WARNING: dof_name not found in row:", dof_name)

#     print("row number", _)

#     print(qpos)

#     robot.set_qpos(qpos)
#     scene.step()
#     cam.render()

# cam.stop_recording(save_to_filename=f'{VIDEO_PATH}/{FILE_NAME}.mp4', fps=120)
MAX_ROWS_PER_RECORDING = 15000
segment_count = 1
row_count = 0
for _, row in df.iterrows():
    # qpos[0:3] = row[['x', 'y', 'z']].values
    # qpos[3:7] = row[['qw', 'qx', 'qy', 'qz']].values
    # qpos[2]  += y_offset
    for dof_name, dof_idx in dof_idx_map.items():
        if dof_name in row:
            qpos[dof_idx + 7] = row[dof_name]
        else:
            print("WARNING: dof_name not found in row:", dof_name)

    robot.set_qpos(qpos)
    scene.step()
    cam.render()

    row_count += 1

    if row_count >= MAX_ROWS_PER_RECORDING:
        # Stop and save the current recording
        cam.stop_recording(save_to_filename=f'{VIDEO_PATH}/{FILE_NAME}_segment_{segment_count}.mp4', fps=120)
        segment_count += 1
        row_count = 0

        # Start a new recording
        cam.start_recording()

# Stop and save the final recording if there are remaining rows
if row_count > 0:
    cam.stop_recording(save_to_filename=f'{VIDEO_PATH}/{FILE_NAME}_segment_{segment_count}.mp4', fps=120)