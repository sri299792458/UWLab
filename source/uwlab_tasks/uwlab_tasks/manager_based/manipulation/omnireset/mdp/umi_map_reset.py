"""UMI reset events using the recorded atlas.

Free-hand proposals use continuous tool positions and stock hand orientations,
screened against position-only coverage. Cube XY proposals use that workspace
and edge filtering while retaining upstream height/orientation samples. Grasp
selection and perturbations remain upstream.
All targets use atlas-seeded IK; exact-pose errors are diagnostic only.
"""
import json
import numpy as np
import torch

import isaaclab.utils.math as math_utils
from pathlib import Path
import yaml
from .events import (reset_end_effector_round_fixed_asset, reset_end_effector_from_grasp_dataset,
                     reset_root_states_uniform)
from .terminations import check_reset_state_success
from .umi_sphere_geometry import UmiSphereGeometry, ROOT, ATLAS

LOOKUP = ROOT / '49_clearance_and_placement/lookup'


class AtlasSeeds:
    def __init__(self, env):
        self.cfg = json.loads((ATLAS / 'config.json').read_text())
        self.q = torch.from_numpy(np.load(LOOKUP / 'atlas_joint_seeds.npy')).to(env.device)
        self.q = self.q.reshape(-1, self.q.shape[-2], 6)
        self.nearest = torch.from_numpy(np.load(LOOKUP / 'nearest_occupied_cell.npy')).to(env.device).flatten()
        self.axes = [torch.tensor(self.cfg[k], device=env.device) for k in
                     ('height_above_table_m', 'y_world_m', 'x_world_m')]
        self.orientations = torch.tensor([o['quaternion_world_wxyz'] for o in self.cfg['orientations']], device=env.device)
        self.tool_p = torch.tensor(self.cfg['tool_offset']['pos'], device=env.device)
        self.tool_q = torch.tensor(self.cfg['tool_offset']['quat'], device=env.device)

    def covers_positions(self, positions):
        """Position-only coverage: some recorded arm orientation exists here.

        For cube centers this is an approximate workspace screen. It does not
        certify a grasp or constrain the sampled cube/hand orientation.
        """
        coordinates = [positions[:, 2] - self.cfg['tabletop_z_m'], positions[:, 1], positions[:, 0]]
        inside = torch.isfinite(positions).all(dim=1)
        idx = []
        for axis, value in zip(self.axes, coordinates):
            inside &= (value >= axis[0]) & (value <= axis[-1])
            idx.append((axis[None] - value[:, None]).abs().argmin(dim=1))
        cell = (idx[0] * len(self.axes[1]) + idx[1]) * len(self.axes[2]) + idx[2]
        # Occupied cells map to themselves in the validated nearest-cell table.
        return inside & (self.nearest[cell] == cell)

    def cube_edge_clear(self, cube_p, cube_q):
        """50 mm from a rotated 60 mm cube's bounding footprint to table edges."""
        half = .03 * math_utils.matrix_from_quat(cube_q).abs().sum(dim=2)[:, :2]
        bounds = self.cfg['table_bounds_world_m']
        lo = torch.tensor(bounds['min'][:2], device=cube_p.device) + .05
        hi = torch.tensor(bounds['max'][:2], device=cube_p.device) - .05
        return ((cube_p[:, :2] - half >= lo) & (cube_p[:, :2] + half <= hi)).all(dim=1)

    def query(self, wrist_p, wrist_q):
        n = len(wrist_p)
        p, q = math_utils.combine_frame_transforms(wrist_p, wrist_q,
            self.tool_p.expand(n, -1), self.tool_q.expand(n, -1))
        coordinates = [p[:, 2] - self.cfg['tabletop_z_m'], p[:, 1], p[:, 0]]
        idx = [(axis[None] - value[:, None]).abs().argmin(dim=1)
               for axis, value in zip(self.axes, coordinates)]
        cell = (idx[0] * len(self.axes[1]) + idx[1]) * len(self.axes[2]) + idx[2]
        candidates = self.q[self.nearest[cell]]
        score = (q @ self.orientations.T).abs()
        score.masked_fill_(~torch.isfinite(candidates).all(dim=-1), -1.)
        orientation = score.argmax(dim=1)
        return candidates[torch.arange(n, device=p.device), orientation]

    def covers_tool_poses(self, tool_p, tool_q):
        """Screen the nearest recorded pose without an occupied-cell fallback.

        Inputs are world-axis poses relative to the environment origin. This is
        finite-map coverage, not an exact-target IK or collision certificate.
        """
        coordinates = [tool_p[:, 2] - self.cfg['tabletop_z_m'], tool_p[:, 1], tool_p[:, 0]]
        inside = torch.isfinite(tool_q).all(dim=1)
        idx = []
        for axis, value in zip(self.axes, coordinates):
            inside &= (value >= axis[0]) & (value <= axis[-1])
            idx.append((axis[None] - value[:, None]).abs().argmin(dim=1))
        cell = (idx[0] * len(self.axes[1]) + idx[1]) * len(self.axes[2]) + idx[2]
        orientation = (tool_q @ self.orientations.T).abs().argmax(dim=1)
        return inside & torch.isfinite(self.q[cell, orientation]).all(dim=1)

    def sample_free_hand_poses(self, count, orientation_ranges):
        """Choose covered XYZ positions; independently retain stock hand angles."""
        device = self.q.device
        angles = math_utils.sample_uniform(orientation_ranges[:, 0], orientation_ranges[:, 1],
                                           (count, 3), device=device)
        hand_q = math_utils.quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
        tool_p = torch.empty((count, 3), device=device)
        lower = torch.stack([self.axes[2][0], self.axes[1][0], self.axes[0][0]])
        upper = torch.stack([self.axes[2][-1], self.axes[1][-1], self.axes[0][-1]])
        filled = 0
        for _ in range(128):
            remaining = count - filled
            if remaining == 0:
                break
            batch = max(64, 2 * remaining)
            p = math_utils.sample_uniform(lower, upper, (batch, 3), device=device)
            p[:, 2] += self.cfg['tabletop_z_m']
            ids = torch.where(self.covers_positions(p))[0][:remaining]
            tool_p[filled:filled + len(ids)] = p[ids]
            filled += len(ids)
        if filled != count:
            raise RuntimeError(f'Map position sampler filled {filled}/{count} poses after 128 batches.')
        base_p = tool_p - math_utils.quat_apply(hand_q, self.tool_p.expand(count, -1))
        return base_p, hand_q


class _AtlasSolver:
    """Add map seeds at the existing solver's process_actions boundary."""
    def __init__(self, solver, owner, env):
        self.solver, self.owner, self.env = solver, owner, env
        if not hasattr(env, '_umi_atlas_seeds'):
            env._umi_atlas_seeds = AtlasSeeds(env)
        if not hasattr(env, '_umi_map_ik_valid'):
            env._umi_map_ik_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            env._umi_map_position_error = torch.full((env.num_envs,), float('inf'), device=env.device)
            env._umi_map_rotation_error = torch.full((env.num_envs,), float('inf'), device=env.device)
            env._umi_map_stats = dict(proposed=0, ik_converged=0, upstream_accepted=0,
                                      geometry_checked=0, self_rejected=0, world_rejected=0,
                                      placement_rejected=0, accepted=0)
            arm = env.cfg.actions.arm
            print('MAP_RESET_CONFIG ' + json.dumps(dict(atlas=str(ATLAS),
                lookup=str(LOOKUP), kp=list(arm.motion_stiffness),
                kd=[2 * kp**.5 * ratio for kp, ratio in zip(arm.motion_stiffness, arm.motion_damping_ratio)],
                physics_dt=env.physics_dt, policy_dt=env.step_dt,
                robot_usd=owner.robot.cfg.spawn.usd_path)), flush=True)
        map_names = env._umi_atlas_seeds.cfg['joint_names']
        ids = list(range(owner.robot.num_joints))[owner.joint_ids] if isinstance(owner.joint_ids, slice) else owner.joint_ids
        self.seed_order = [map_names.index(owner.robot.joint_names[i]) for i in ids]

    def __getattr__(self, name):
        return getattr(self.solver, name)

    def process_actions(self, command):
        ids, robot = self.env_ids, self.owner.robot
        self.target = command.clone()
        p, q = math_utils.combine_frame_transforms(robot.data.root_link_pos_w[ids],
            robot.data.root_link_quat_w[ids], command[ids, :3], command[ids, 3:])
        seeds = self.env._umi_atlas_seeds.query(p - self.env.scene.env_origins[ids], q)
        seeds = seeds[:, self.seed_order]
        robot.write_joint_state_to_sim(seeds, torch.zeros_like(seeds),
                                      joint_ids=self.owner.joint_ids, env_ids=ids)
        self.solver.process_actions(command)

    def validate(self):
        ids, env = self.env_ids, self.env
        p, q = self.solver._compute_frame_pose()
        pe, re = math_utils.compute_pose_error(p[ids], q[ids], self.target[ids, :3], self.target[ids, 3:],
                                              rot_error_type='axis_angle')
        env._umi_map_position_error[ids] = pe.norm(dim=1)
        env._umi_map_rotation_error[ids] = re.norm(dim=1)
        env._umi_map_ik_valid[ids] = ((pe.norm(dim=1) <= .001) & (re.norm(dim=1) <= .01)
                                    & torch.isfinite(self.owner.robot.data.joint_pos[ids]).all(dim=1))
        env._umi_map_stats['proposed'] += len(ids)
        env._umi_map_stats['ik_converged'] += int(env._umi_map_ik_valid[ids].sum())


class MapEndEffectorAnywhere(reset_end_effector_round_fixed_asset):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.solver = _AtlasSolver(self.solver, self, env)

    def __call__(self, env, env_ids, fixed_asset_cfg, fixed_asset_offset, pose_range_b, robot_ik_cfg):
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        if len(env_ids) == 0:
            return
        hand_p, hand_q = env._umi_atlas_seeds.sample_free_hand_poses(len(env_ids), self.ranges[3:])
        # The map is anchored to the lab scene; account for environment origins
        # and the actual (possibly perturbed) robot root only when converting to IK.
        pos_b, quat_b = self.solver._compute_frame_pose()
        pos_b[env_ids], quat_b[env_ids] = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w[env_ids], self.robot.data.root_link_quat_w[env_ids],
            hand_p + env.scene.env_origins[env_ids], hand_q)
        self.solver.env_ids = env_ids
        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))
        # Retain the existing 25 partial updates; pose errors are diagnostic only.
        for _ in range(25):
            self.solver.apply_actions()
            position = self.robot.data.joint_pos[env_ids]
            position = position + .25 * (self.robot.data.joint_pos_target[env_ids] - position)
            self.robot.write_joint_state_to_sim(position[:, self.joint_ids],
                torch.zeros((len(env_ids), self.n_joints), device=env.device),
                joint_ids=self.joint_ids, env_ids=env_ids)
        self.solver.validate()


class MapEndEffectorGrasped(reset_end_effector_from_grasp_dataset):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.solver = _AtlasSolver(self.solver, self, env)

    def __call__(self, env, env_ids, dataset_dir, fixed_asset_cfg, robot_ik_cfg, gripper_cfg, pose_range_b={}):
        self.solver.env_ids = env_ids
        super().__call__(env, env_ids, dataset_dir, fixed_asset_cfg, robot_ik_cfg, gripper_cfg, pose_range_b)
        self.solver.validate()


class MapCubePlacement(reset_root_states_uniform):
    """Select covered cube locations while preserving sampled height/orientation."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        assert len(self.asset_cfgs) == 1
        if not hasattr(env, '_umi_atlas_seeds'):
            env._umi_atlas_seeds = AtlasSeeds(env)
        self.atlas = env._umi_atlas_seeds
        self.cube_name = self.asset_cfgs[0].name
        self.free_object = hasattr(env.cfg.events, 'reset_end_effector_pose')
        if self.cube_name == 'receptive_object':
            offsets = []
            for name in ('receptive_object', 'insertive_object'):
                meta = yaml.safe_load(Path(env.scene[name].cfg.spawn.usd_path).with_name('metadata.yaml').read_text())
                offsets.append(tuple(torch.tensor(meta['assembled_offset'][key], device=env.device, dtype=torch.float32).unsqueeze(0)
                                     for key in ('pos', 'quat')))
            (rp, rq), (ip, iq) = offsets
            inv_q = math_utils.quat_inv(iq)
            inv_p = -math_utils.quat_apply(inv_q, ip)
            self.goal_p, self.goal_q = math_utils.combine_frame_transforms(rp, rq, inv_p, inv_q)

    def __call__(self, env, env_ids, pose_range, velocity_range, asset_cfgs={},
                 offset_asset_cfg=None, use_bottom_offset=False):
        # Draw the complete upstream proposal once. Rejection below relocates
        # only XY, retaining its Z, quaternion and velocity samples.
        super().__call__(env, env_ids, pose_range, velocity_range, asset_cfgs,
                         offset_asset_cfg, use_bottom_offset)
        cube = env.scene[self.cube_name]
        remaining = env_ids
        for _ in range(128):
            if not len(remaining):
                return
            p = cube.data.root_pos_w[remaining] - env.scene.env_origins[remaining]
            q = cube.data.root_quat_w[remaining]
            good = self.atlas.cube_edge_clear(p, q)
            point = p.clone()
            if self.cube_name == 'receptive_object':
                point, _ = math_utils.combine_frame_transforms(p, q,
                    self.goal_p.expand(len(p), -1), self.goal_q.expand(len(p), -1))
            elif self.free_object:
                # A free cube is expected to settle onto the table. Use its
                # nominal resting-center height only for the XY workspace.
                point[:, 2] = self.atlas.cfg['tabletop_z_m'] + .03
            good &= self.atlas.covers_positions(point)
            remaining = remaining[~good]
            if len(remaining):
                pose = cube.data.root_pose_w[remaining].clone()
                xy = math_utils.sample_uniform(self.pose_range[:2, 0], self.pose_range[:2, 1],
                                                (len(remaining), 2), device=env.device)
                pose[:, :2] = cube.data.default_root_state[remaining, :2] + env.scene.env_origins[remaining, :2] + xy
                if self.use_bottom_offset:
                    pose[:, :2] -= self.bottom_offset_positions[self.cube_name][remaining, :2]
                cube.write_root_pose_to_sim(pose, env_ids=remaining)
        raise RuntimeError(f'Map cube sampler could not place {len(remaining)} '
                           f'{self.cube_name} environments after 128 batches.')


def _map_acceptance(env, accepted):
    before = int(accepted.sum())
    env._umi_map_stats['upstream_accepted'] += before
    ids = torch.where(accepted)[0]
    env._umi_map_stats['geometry_checked'] += len(ids)
    if len(ids):
        robot = env.scene['robot']
        if not hasattr(env, '_umi_sphere_geometry'):
            env._umi_sphere_geometry = UmiSphereGeometry(robot.body_names, env.device)
        self_clear, world_clear = env._umi_sphere_geometry.evaluate(
            robot.data.body_link_pos_w[ids], math_utils.matrix_from_quat(robot.data.body_link_quat_w[ids]),
            env.scene.env_origins[ids])
        accepted[ids] &= self_clear & world_clear
        env._umi_map_stats['self_rejected'] += int((~self_clear).sum())
        env._umi_map_stats['world_rejected'] += int((~world_clear).sum())
        placement_clear = torch.ones(len(ids), dtype=torch.bool, device=env.device)
        for name in ('insertive_object', 'receptive_object'):
            cube = env.scene[name]
            p = cube.data.root_pos_w[ids] - env.scene.env_origins[ids]
            placement_clear &= env._umi_atlas_seeds.cube_edge_clear(p, cube.data.root_quat_w[ids])
        accepted[ids] &= placement_clear
        env._umi_map_stats['placement_rejected'] += int((~placement_clear).sum())
    env._umi_map_stats['accepted'] += int(accepted.sum())
    if before:
        print('MAP_RESET_STATS ' + json.dumps(env._umi_map_stats), flush=True)
    return accepted


class MapResetSuccess(check_reset_state_success):
    def __call__(self, env, object_cfgs, robot_cfg, ee_body_name, collision_analyzer_cfgs,
                 max_robot_pos_deviation=.1, max_object_pos_deviation=.1, pos_z_threshold=-.01,
                 consecutive_stability_steps=5, insertive_asset_cfg=None, receptive_asset_cfg=None,
                 assembly_success_prob=None, assembly_threshold_scale=1.):
        result = super().__call__(env, object_cfgs, robot_cfg, ee_body_name, collision_analyzer_cfgs,
            max_robot_pos_deviation, max_object_pos_deviation, pos_z_threshold, consecutive_stability_steps,
            insertive_asset_cfg, receptive_asset_cfg, assembly_success_prob, assembly_threshold_scale)
        return _map_acceptance(env, result)
