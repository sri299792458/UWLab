"""Read back the corrected native mounting pose and render the unchanged lab."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args, multi_gpu=False).app

import gymnasium as gym
import imageio.v2 as imageio
import torch
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation
import uwlab_tasks
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85 import umi_reset_cfg as hardware
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_sim2real_cfg import ThunderSysidCfg
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.umi_training_cfg import UmiCubeTrainCfg

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO/'scripts_v2/tools/curobo_umi'))
from reachability_geometry import ROOT, ARM_NAMES, batch_fk, pose_matrix


def main():
    atlas = ROOT/'49_clearance_and_placement/atlas'
    cfg_map = json.loads((atlas/'config.json').read_text())
    audit = json.loads((ROOT/'15_table_reachability/model/export_audit.json').read_text())
    # Ensure an old map fails during task configuration, before spawning a scene.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'atlas/config.json'
        path.parent.mkdir()
        stale = dict(cfg_map, robot_base_quaternion_world_wxyz=[.7071068, 0, .7071068, 0])
        path.write_text(json.dumps(stale))
        saved = hardware.DATASET_DIR
        hardware.DATASET_DIR = str(Path(tmp)/'OmniReset')
        try:
            try:
                UmiCubeTrainCfg()
            except ValueError as error:
                assert 'mounting pose does not match' in str(error)
            else:
                raise AssertionError('Old-mount map was not rejected')
        finally:
            hardware.DATASET_DIR = saved

    hi = int(np.argmin(np.abs(np.asarray(cfg_map['height_above_table_m'])-.30)))
    with np.load(atlas/'pilot'/f'h{hi:02d}_o000.npz') as data:
        ys, xs = np.where(data['status'] == 4)
        distance = (np.asarray(cfg_map['x_world_m'])[xs]-.10)**2 + (np.asarray(cfg_map['y_world_m'])[ys]+.12)**2
        selected = int(np.argmin(distance))
        q = data['chosen_q'][ys[selected], xs[selected]]
    lab, cfg = UmiCubeTrainCfg(), ThunderSysidCfg()
    cfg.scene.robot.init_state.pos = lab.scene.robot.init_state.pos
    cfg.scene.robot.init_state.rot = lab.scene.robot.init_state.rot
    cfg.scene.table = copy.deepcopy(lab.scene.table)
    cfg.scene.ground = copy.deepcopy(lab.scene.ground)
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = 42
    cfg.viewer.resolution = (1280, 960)
    cfg.viewer.eye = (1.65, -2.1, 1.95)
    cfg.viewer.lookat = (-.05, .02, .94)
    env = gym.make('OmniReset-Thunder-UMI-Sysid-v0', cfg=cfg, render_mode='rgb_array').unwrapped
    try:
        env.reset()
        robot = env.scene['robot']
        joints = torch.zeros_like(robot.data.joint_pos)
        for name, value in zip(ARM_NAMES, q):
            joints[:, robot.joint_names.index(name)] = float(value)
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        env.sim.forward()
        robot.update(0.)
        poses = robot.data.body_pose_w[0].cpu().numpy()
        base = poses[robot.body_names.index('base_link')]
        expected_base = pose_matrix(hardware.ROBOT_POS, hardware.ROBOT_ROT)
        native_base = pose_matrix(base[:3], base[3:])
        assert np.max(np.abs(native_base-expected_base)) < 2e-6
        fk = batch_fk(audit, q[None])
        p_error, r_error = [], []
        for name in robot.body_names:
            native = poses[robot.body_names.index(name)]
            expected = expected_base @ fk[name][0]
            p_error.append(float(np.linalg.norm(native[:3]-expected[:3,3])))
            r_error.append(float((Rotation.from_matrix(expected[:3,:3]).inv()*
                                 Rotation.from_quat(native[[4,5,6,3]])).magnitude()))
        assert max(p_error) < 2e-5 and max(r_error) < 2e-5
        table_prim = next(p for p in env.sim.stage.Traverse() if p.GetName() == 'vention_mat')
        vertices = np.asarray(UsdGeom.Mesh(table_prim).GetPointsAttr().Get())
        transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(table_prim))
        world = (np.c_[vertices, np.ones(len(vertices))] @ transform)[:,:3]
        bounds = np.array([world.min(0), world.max(0)])
        expected_bounds = np.array([cfg_map['table_bounds_world_m'][key] for key in ('min','max')])
        assert np.max(np.abs(bounds-expected_bounds)) < 2e-6
        for _ in range(20):
            frame = env.render()
        imageio.imwrite(args.output/'corrected_mount_front.png', frame)
        env.sim.set_camera_view((-1.65, -1.85, 1.9), (-.05, .02, .94))
        for _ in range(15):
            frame = env.render()
        imageio.imwrite(args.output/'corrected_mount_other_angle.png', frame)
        report = dict(passed=True, old_mount_map_rejected=True,
            robot_base_position_world_m=base[:3].tolist(), robot_base_quaternion_world_wxyz=base[3:].tolist(),
            joint_positions_rad=q.tolist(), max_native_fk_position_error_m=max(p_error),
            max_native_fk_rotation_error_rad=max(r_error), tabletop_native_world_bounds_m=bounds.tolist(),
            table_bounds_error_m=float(np.max(np.abs(bounds-expected_bounds))),
            scope='Native Isaac pose/geometry readback and static render. Pilot atlas endpoint for illustration; not a collection-motion validation.')
        (args.output/'mounting_validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report), flush=True)
    finally:
        env.close()


try:
    main()
finally:
    app.close()
