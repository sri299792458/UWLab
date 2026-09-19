"""Render saved simulated joint trajectories in the same Thunder lab scene."""
import argparse
import copy
import json
from pathlib import Path
import traceback

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--preview", required=True, type=Path)
parser.add_argument("--candidate", required=True, type=int)
parser.add_argument("--variant", default="UW_reference_dynamics_delay_4ms")
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=960)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args, multi_gpu=False).app

import gymnasium as gym
import imageio.v2 as imageio
import uwlab_tasks
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_sim2real_cfg import ThunderSysidCfg
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_training_cfg import UmiCubeTrainCfg


def main():
    manifest = json.loads((args.preview / "preview.json").read_text())
    source = np.load(args.preview / "trajectories.npz")
    row = next(i for i, r in enumerate(manifest["rows"]) if r["candidate"] == args.candidate and
               r["variant"] == args.variant)
    q = source["full_joint_positions"][:, row]
    recorded_body = source["body_poses"][:, row].copy()
    recorded_body[..., :3] -= source["env_origins"][row]
    cfg, lab = ThunderSysidCfg(), UmiCubeTrainCfg()
    cfg.scene.robot.init_state.pos = lab.scene.robot.init_state.pos
    cfg.scene.robot.init_state.rot = lab.scene.robot.init_state.rot
    cfg.scene.table = copy.deepcopy(lab.scene.table)
    cfg.scene.ground = copy.deepcopy(lab.scene.ground)
    cfg.scene.num_envs = 1; cfg.sim.device = args.device; cfg.seed = 42
    cfg.viewer.resolution = (args.width, args.height)
    cfg.viewer.eye = (1.45, -1.75, 2.0); cfg.viewer.lookat = (.05, -.03, 1.2)
    env = gym.make("OmniReset-Thunder-UMI-Sysid-v0", cfg=cfg, render_mode="rgb_array").unwrapped
    try:
        env.reset()
        robot = env.scene["robot"]
        assert robot.joint_names == manifest["joint_names"]
        def frame_at(index):
            joints = torch.tensor(q[index], device=env.device)[None]
            robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
            env.sim.forward(); robot.update(0.)
            frame = env.render()
            error = float(np.max(np.abs(robot.data.body_pos_w[0].cpu().numpy()-recorded_body[index, :, :3])))
            return frame, error
        for _ in range(15):
            frame, _ = frame_at(0)
        imageio.imwrite(args.output / "start_pose.png", frame)
        fps = args.fps
        indices = np.round(np.arange(0, 8, 1/fps)/manifest["dt"]).astype(int)
        max_error = 0.
        with imageio.get_writer(args.output / "collection_sweep.mp4", fps=fps, codec="libx264", quality=8) as writer:
            for number, index in enumerate(indices):
                frame, error = frame_at(int(index)); max_error = max(max_error, error)
                writer.append_data(frame)
                if number % 60 == 0:
                    print(f"Rendered {number}/{len(indices)} frames", flush=True)
        env.sim.set_camera_view((-1.1, -1.35, 1.85), (.05, -.03, 1.25))
        for _ in range(12):
            frame, _ = frame_at(0)
        imageio.imwrite(args.output / "start_pose_second_view.png", frame)
        details = {"source": str(args.preview), "candidate": args.candidate, "row": row,
                   "variant": manifest["rows"][row]["variant"], "fps": fps, "frames": len(indices),
                   "rendered_duration_s": 8, "new_physics_steps": 0,
                   "max_render_body_position_difference_m": max_error,
                   "method": "Replay the saved full joint states for visualization; use the original 500 Hz body states for collision measurements"}
        (args.output / "render_report.json").write_text(json.dumps(details, indent=2)+"\n")
        print(json.dumps(details), flush=True)
    except BaseException:
        traceback.print_exc(); raise
    finally:
        env.close()


try:
    main()
finally:
    app.close()
