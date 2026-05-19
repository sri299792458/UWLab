# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch UWLab UR5e/OmniReset visual inspection views.

The default view creates the camera-alignment task scene that instantiates
runtime camera sensors, including:

    /World/envs/env_0/Robot/robotiq_base_link/rgb_wrist_camera

The same script can also spawn only the robot or only the OmniReset ``table``
asset for isolated USD inspection.

It can also spawn the lab Vention v60 CAD with the UWLab simulated UR5e placed
on the right-arm station from the CAD.

Example WebRTC/private-network launch:

    CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 LIVESTREAM=2 ENABLE_CAMERAS=1 \
        ./uwlab.sh -p scripts_v2/tools/view_ur5e_camera_scene.py --livestream 2

Example table/Vention-only view:

    CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 LIVESTREAM=2 ENABLE_CAMERAS=1 \
        ./uwlab.sh -p scripts_v2/tools/view_ur5e_camera_scene.py --view table --livestream 2

Example lab right-arm view:

    CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 LIVESTREAM=2 ENABLE_CAMERAS=1 \
        ./uwlab.sh -p scripts_v2/tools/view_ur5e_camera_scene.py --view lab-right-arm --livestream 2
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View UWLab UR5e/OmniReset assets and camera-alignment scenes.")
parser.add_argument(
    "--view",
    choices=("scene", "robot", "table", "lab-right-arm"),
    default="scene",
    help="What to show: full camera scene, robot only, table/Vention support only, or lab right-arm layout.",
)
parser.add_argument(
    "--camera",
    choices=("front_camera", "side_camera", "wrist_camera"),
    default="wrist_camera",
    help="Camera to highlight in the startup summary.",
)
parser.add_argument(
    "--scene-layout",
    choices=("default", "lab-right-arm"),
    default="default",
    help="Scene layout to use with --view scene.",
)
parser.add_argument(
    "--joint-angles",
    type=float,
    nargs=6,
    default=[2.28, -95.58, 99.07, -93.36, -86.57, 4.33],
    help="Initial UR5e arm joint angles in degrees.",
)
parser.add_argument("--gripper-pos", type=float, default=1.0, help="Gripper command, 0=closed and 1=open.")
parser.add_argument("--warmup-steps", type=int, default=30, help="Simulation warmup steps before holding the scene.")
parser.add_argument(
    "--robot-config",
    choices=("base", "implicit", "explicit"),
    default="implicit",
    help="Robot config for --view robot.",
)
parser.add_argument(
    "--prim-path",
    default=None,
    help="Prim path for isolated --view robot/table. Defaults to /World/Robot or /World/Table.",
)
parser.add_argument(
    "--pos",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Optional spawn translation for isolated --view robot/table.",
)
parser.add_argument(
    "--rot",
    type=float,
    nargs=4,
    default=None,
    metavar=("W", "X", "Y", "Z"),
    help="Optional spawn quaternion in wxyz order for isolated --view robot/table.",
)
parser.add_argument(
    "--show-ground",
    action="store_true",
    help="Spawn a ground plane in isolated --view robot/table/lab-right-arm modes.",
)
parser.add_argument(
    "--lab-vention-usd",
    default=None,
    help="Converted lab Vention v60 USD used by --view lab-right-arm. Defaults to lab_layout_cfg.py.",
)
parser.add_argument(
    "--show-cad-arms",
    action="store_true",
    help="Deprecated: the processed lab Vention asset has the CAD UR5 arms removed.",
)
parser.add_argument(
    "--clean-shutdown",
    action="store_true",
    help="Use Isaac Sim's full shutdown on Ctrl+C. By default this viewer exits immediately.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_shutdown_signal: int | None = None


def _request_shutdown(signum: int, _frame):
    global _shutdown_signal
    _shutdown_signal = signum
    signame = signal.Signals(signum).name
    if not args_cli.clean_shutdown:
        print(f"\n[INFO]: Received {signame}; hard exiting camera scene.", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(128 + signum)
    print(f"\n[INFO]: Received {signame}; cleanly shutting down camera scene.", flush=True)
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _request_shutdown)
signal.signal(signal.SIGTERM, _request_shutdown)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR  # noqa: E402
from uwlab_assets.robots.ur5e_robotiq_gripper import (  # noqa: E402
    EXPLICIT_UR5E_ROBOTIQ_2F85,
    IMPLICIT_UR5E_ROBOTIQ_2F85,
    UR5E_ARTICULATION,
)
import uwlab_tasks  # noqa: E402, F401
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.camera_align_cfg import (  # noqa: E402
    CameraAlignEnvCfg,
    LabRightArmCameraAlignEnvCfg,
)
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.lab_layout_cfg import (  # noqa: E402
    LAB_RIGHT_ARM_ROBOT_POS,
    LAB_RIGHT_ARM_ROBOT_ROT,
    LAB_TABLETOP_TOP_Z,
    LAB_VENTION_POS,
    LAB_VENTION_ROT,
    LAB_VENTION_SCALE,
    LAB_VENTION_USD_PATH,
)


def _make_env():
    if args_cli.scene_layout == "lab-right-arm":
        env_cfg = LabRightArmCameraAlignEnvCfg()
        env_id = "OmniReset-Ur5eRobotiq2f85-CameraAlign-LabRightArm-v0"
    else:
        env_cfg = CameraAlignEnvCfg()
        env_id = "OmniReset-Ur5eRobotiq2f85-CameraAlign-v0"

    joint_names = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    for name, angle_deg in zip(joint_names, args_cli.joint_angles):
        env_cfg.scene.robot.init_state.joint_pos[name] = float(np.deg2rad(angle_deg))

    return gym.make(env_id, cfg=env_cfg)


def _make_hold_action(env):
    action_dim = env.unwrapped.action_manager.total_action_dim
    action = torch.zeros(1, action_dim, device=env.unwrapped.device)
    if action_dim > 0:
        action[0, -1] = args_cli.gripper_pos
    return action


def _print_scene_summary(env):
    stage = omni.usd.get_context().get_stage()

    print("\n[INFO]: Runtime camera prims:", flush=True)
    for name, sensor in env.unwrapped.scene._sensors.items():
        prim_paths = [str(prim.GetPath()) for prim in sensor._sensor_prims]
        marker = " [selected]" if name == args_cli.camera else ""
        print(f"  - {name}{marker}: {prim_paths}", flush=True)

    print("\n[INFO]: D415 mount prims in this scene:", flush=True)
    for path in (
        "/World/envs/env_0/Robot/robotiq_base_link/visuals/D415_to_Robotiq_Mount",
        "/World/envs/env_0/Robot/robotiq_base_link/collisions/D415_to_Robotiq_Mount",
    ):
        prim = stage.GetPrimAtPath(path)
        print(f"  - {path}: {'found' if prim.IsValid() else 'missing'}", flush=True)

    print("\n[INFO]: Scene is running. Inspect /World/envs/env_0/Robot in the Stage tree.", flush=True)


def _run_camera_scene():
    env = _make_env()
    try:
        action = _make_hold_action(env)
        print(f"[INFO]: Warming up camera scene for {args_cli.warmup_steps} steps...", flush=True)
        env.reset()
        for _ in range(args_cli.warmup_steps):
            env.step(action)

        _print_scene_summary(env)

        while simulation_app.is_running():
            env.step(action)
    finally:
        env.close()


def _selected_prim_path(default: str) -> str:
    return args_cli.prim_path if args_cli.prim_path is not None else default


def _spawn_common_lightweight_scene():
    if args_cli.show_ground:
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)


def _select_robot_cfg():
    if args_cli.robot_config == "base":
        return UR5E_ARTICULATION.copy()
    if args_cli.robot_config == "explicit":
        return EXPLICIT_UR5E_ROBOTIQ_2F85.copy()
    return IMPLICIT_UR5E_ROBOTIQ_2F85.copy()


def _spawn_robot_only() -> Articulation:
    robot_cfg = _select_robot_cfg()
    robot_cfg.prim_path = _selected_prim_path("/World/Robot")
    if args_cli.pos is not None:
        robot_cfg.init_state.pos = tuple(args_cli.pos)
    if args_cli.rot is not None:
        robot_cfg.init_state.rot = tuple(args_cli.rot)

    print(f"[INFO]: Spawning robot config: {args_cli.robot_config}", flush=True)
    print(f"[INFO]: Robot prim path: {robot_cfg.prim_path}", flush=True)
    print(f"[INFO]: Robot USD path: {robot_cfg.spawn.usd_path}", flush=True)
    print(f"[INFO]: pos={robot_cfg.init_state.pos} rot={robot_cfg.init_state.rot}", flush=True)

    return Articulation(cfg=robot_cfg)


def _initialize_robot(robot: Articulation, sim: SimulationContext):
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    robot.update(sim.get_physics_dt())

    print(f"[INFO]: Joint names: {robot.joint_names}", flush=True)
    print(f"[INFO]: Body names: {robot.body_names}", flush=True)


def _spawn_table_only():
    table_pos = tuple(args_cli.pos) if args_cli.pos is not None else (0.4, 0.0, -0.881)
    table_rot = tuple(args_cli.rot) if args_cli.rot is not None else (0.707, 0.0, 0.0, -0.707)
    table_usd_path = f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention/pat_vention.usd"
    table_cfg = sim_utils.UsdFileCfg(
        usd_path=table_usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    prim_path = _selected_prim_path("/World/Table")
    table_cfg.func(prim_path, table_cfg, translation=table_pos, orientation=table_rot)

    print(f"[INFO]: Spawned OmniReset table/Vention support at: {prim_path}", flush=True)
    print(f"[INFO]: USD path: {table_usd_path}", flush=True)
    print(f"[INFO]: pos={table_pos} rot={table_rot}", flush=True)


def _spawn_lab_vention_frame():
    lab_vention_usd = args_cli.lab_vention_usd or LAB_VENTION_USD_PATH
    table_cfg = sim_utils.UsdFileCfg(
        usd_path=lab_vention_usd,
        scale=LAB_VENTION_SCALE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    table_cfg.func("/World/Table", table_cfg, translation=LAB_VENTION_POS, orientation=LAB_VENTION_ROT)

    print("[INFO]: Spawned lab Vention v60 frame at /World/Table", flush=True)
    print(f"[INFO]: USD path: {lab_vention_usd}", flush=True)
    print(f"[INFO]: pos={LAB_VENTION_POS} rot={LAB_VENTION_ROT} scale={LAB_VENTION_SCALE}", flush=True)
    print(f"[INFO]: tabletop top z ~= {LAB_TABLETOP_TOP_Z:.5f} m", flush=True)


def _spawn_lab_right_arm_robot() -> Articulation:
    robot_cfg = _select_robot_cfg()
    robot_cfg.prim_path = _selected_prim_path("/World/Robot")
    robot_cfg.init_state.pos = LAB_RIGHT_ARM_ROBOT_POS if args_cli.pos is None else tuple(args_cli.pos)
    robot_cfg.init_state.rot = LAB_RIGHT_ARM_ROBOT_ROT if args_cli.rot is None else tuple(args_cli.rot)

    print(f"[INFO]: Spawning simulated UWLab robot at lab right-arm station: {args_cli.robot_config}", flush=True)
    print(f"[INFO]: Robot prim path: {robot_cfg.prim_path}", flush=True)
    print(f"[INFO]: Robot USD path: {robot_cfg.spawn.usd_path}", flush=True)
    print(f"[INFO]: pos={robot_cfg.init_state.pos} rot={robot_cfg.init_state.rot}", flush=True)

    return Articulation(cfg=robot_cfg)


def _run_lightweight_view():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    robot = None
    _spawn_common_lightweight_scene()
    if args_cli.view == "table":
        sim.set_camera_view(eye=[1.6, -1.4, 0.5], target=[0.4, 0.0, -0.45])
        _spawn_table_only()
    elif args_cli.view == "lab-right-arm":
        sim.set_camera_view(eye=[1.5, -2.2, 1.55], target=[0.05, 0.25, 1.05])
        _spawn_lab_vention_frame()
        robot = _spawn_lab_right_arm_robot()
    else:
        sim.set_camera_view(eye=[1.2, -1.2, 1.2], target=[0.0, 0.0, 0.35])
        robot = _spawn_robot_only()

    sim.reset()
    if robot is not None:
        _initialize_robot(robot, sim)

    print("[INFO]: Setup complete. Leave this process running while using the WebRTC client.", flush=True)
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        if robot is not None:
            robot.write_data_to_sim()
        sim.step()
        if robot is not None:
            robot.update(sim_dt)


def main():
    if args_cli.view == "scene":
        _run_camera_scene()
    else:
        _run_lightweight_view()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        exit_code = 128 + (_shutdown_signal or signal.SIGINT)
    finally:
        simulation_app.close()
    sys.exit(exit_code)
