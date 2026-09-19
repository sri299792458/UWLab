"""Screen start poses for the unchanged eight-second collection waveform.

This is a coarse geometric search, not a simulation or hardware safety proof.
Finalists require full-rate IK, native dynamics and source-hull validation.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'workstation'))
sys.path.insert(0, str(HERE.parent / 'curobo_umi'))
import collect_thunder as collector
from motion_geometry import UmiSphereGeometry, clearances
from reachability_geometry import batch_fk

collector.install_calibration()
fixed = []
for joint in collector.kin.CALIBRATED_JOINTS:
    transform = np.eye(4)
    transform[:3, :3] = collector.kin.rpy_to_matrix(joint['rpy'])
    transform[:3, 3] = joint['xyz']
    fixed.append(transform)


def fk_jacobian(q):
    """Batch the exact calibrated transform chain and geometric Jacobian."""
    transform = np.broadcast_to(collector.kin.T_180Z, (len(q), 4, 4)).copy()
    positions, axes = [], []
    for i in range(6):
        joint = transform @ fixed[i]
        positions.append(joint[:, :3, 3].copy())
        axes.append(joint[:, :3, 2].copy())
        turn = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        c, s = np.cos(q[:, i]), np.sin(q[:, i])
        turn[:, 0, 0] = c; turn[:, 1, 1] = c
        turn[:, 0, 1] = -s; turn[:, 1, 0] = s
        transform = joint @ turn
    axes = np.stack(axes, axis=1)
    displacement = transform[:, None, :3, 3] - np.stack(positions, axis=1)
    jacobian = np.concatenate((np.cross(axes, displacement), axes), axis=2).transpose(0, 2, 1)
    return transform, jacobian


def solve(q, target_position, target_rotation):
    q = q.copy()
    for _ in range(60):
        transform, jacobian = fk_jacobian(q)
        error = np.concatenate((target_position - transform[:, :3, 3],
            Rotation.from_matrix(target_rotation @ transform[:, :3, :3].transpose(0, 2, 1)).as_rotvec()), axis=1)
        good = (np.linalg.norm(error[:, :3], axis=1) < 1e-6) & (np.linalg.norm(error[:, 3:], axis=1) < 1e-5)
        if good.all():
            break
        transposed = jacobian.transpose(0, 2, 1)
        delta = np.linalg.solve(transposed @ jacobian + np.eye(6)[None] * 1e-8,
                               (transposed @ error[:, :, None]))[:, :, 0]
        delta *= np.minimum(1., .1 / np.maximum(np.abs(delta).max(axis=1), 1e-12))[:, None]
        q[~good] += delta[~good]
    return q, good


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas-candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--outward-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917'))
    cfg = json.loads((root / '49_clearance_and_placement/atlas/config.json').read_text())
    audit = json.loads((root / '15_table_reachability/model/export_audit.json').read_text())
    old = json.loads((HERE / 'workstation/collection.simulation_candidate.json').read_text())
    q0 = np.array(old['start_joint_positions_rad'])
    # Verify the batched implementation against the unmodified collector.
    test_q = q0 + np.random.default_rng(7119).uniform(-.2, .2, (16, 6))
    transforms, jacobians = fk_jacobian(test_q)
    fk_error = max(np.abs(transforms[i] - collector.kin.forward_kinematics_calibrated(q)[0]).max() for i, q in enumerate(test_q))
    jac_error = max(np.abs(jacobians[i] - collector.kin.compute_jacobian_calibrated(q)).max() for i, q in enumerate(test_q))
    assert max(fk_error, jac_error) < 1e-12, (fk_error, jac_error)
    base_rotation = Rotation.from_quat(np.array(cfg['robot_base_quaternion_world_wxyz'])[[1, 2, 3, 0]]).as_matrix()
    base_position = np.array(cfg['robot_base_position_world_m'])
    body_names = list(dict.fromkeys(['base_link'] + [j['child'] for j in audit['articulation_tree']]))
    current_tool = batch_fk(audit, q0[None])[audit['tool_frame']][0]
    tool_world = base_rotation @ current_tool[:3, 3] + base_position
    initial, _ = fk_jacobian(q0[None])
    # Keep the current wrist orientation and IK branch for the translated grid.
    grid = np.array([[x, y, cfg['tabletop_z_m'] + h] for h in (.36, .44, .52)
                     for y in (-.30, -.20, -.10, 0.) for x in (-.10, 0., .10, .20, .30)])
    if args.outward_only:
        grid = tool_world[None] + np.array([[0., -distance, 0.] for distance in (.10, .15, .20)])
    targets_position = initial[0, :3, 3] + (grid - tool_world) @ base_rotation
    grid_q, grid_good = solve(np.tile(q0, (len(grid), 1)), targets_position,
                            np.broadcast_to(initial[0, :3, :3], (len(grid), 3, 3)))
    candidates = [{'candidate': 0, 'source': 'existing_start', 'joint_positions_rad': q0.tolist()}]
    for world, q, good in zip(grid, grid_q, grid_good):
        if good:
            candidates.append({'candidate': len(candidates), 'source': 'translated_current_orientation',
                'tool_position_world_m': world.tolist(), 'joint_positions_rad': q.tolist()})
    for item in ([] if args.outward_only else json.loads(args.atlas_candidates.read_text())['candidates']):
        q = np.array(item['joint_positions_rad'])
        q = q0 + (q - q0 + np.pi) % (2 * np.pi) - np.pi
        candidates.append(dict(item, candidate=len(candidates), source='atlas_orientation', joint_positions_rad=q.tolist()))
    start_q = np.array([c['joint_positions_rad'] for c in candidates])
    starts, _ = fk_jacobian(start_q)
    offsets = collector.generate_offsets(old)
    indices = np.unique(np.r_[np.arange(0, len(offsets), 10), len(offsets) - 1])
    delta_rotation = Rotation.from_rotvec(offsets[indices, 3:]).as_matrix()
    q = start_q.copy()
    good = np.ones(len(q), bool)
    path = []
    for index, rotation in zip(indices, delta_rotation):
        q, solved = solve(q, starts[:, :3, 3] + offsets[index, :3], rotation[None] @ starts[:, :3, :3])
        good &= solved
        path.append(q.copy())
    path = np.array(path)
    joint_limits = np.array(cfg['joint_limits_rad'])
    if joint_limits.shape == (6, 2):
        lower, upper = joint_limits.T
    else:
        lower, upper = joint_limits
    good &= ((path >= lower) & (path <= upper)).all(axis=(0, 2))
    poses = batch_fk(audit, path.reshape(-1, 6))
    positions = np.stack([poses[name][:, :3, 3] @ base_rotation.T + base_position for name in body_names], axis=1)
    rotations = np.stack([base_rotation @ poses[name][:, :3, :3] for name in body_names], axis=1)
    adapter = UmiSphereGeometry(body_names, 'cuda')
    metrics = clearances(adapter, torch.tensor(positions, device='cuda', dtype=torch.float32),
        torch.tensor(rotations, device='cuda', dtype=torch.float32), torch.zeros(len(positions), 3, device='cuda'))
    metrics = {name: value.cpu().numpy().reshape(len(path), -1).min(axis=0) for name, value in metrics.items()}
    speed = np.abs(np.gradient(path, indices * .002, axis=0)).max(axis=0)
    for i, candidate in enumerate(candidates):
        candidate.update({name: float(value[i]) for name, value in metrics.items()})
        candidate.update(coarse_ik_pass=bool(good[i]),
            sphere_clear=bool(good[i] and metrics['self_margin_m'][i] >= 0 and metrics['world_margin_m'][i] >= 0),
            max_requested_joint_speed_rad_s=speed[i].tolist(),
            joint_positions_deg=np.degrees(start_q[i]).tolist())
        # Use both obstacle and arm separation; fixed sphere gaps can cap either metric.
        candidate['ranking_score_m'] = min(candidate['world_margin_m'], candidate['self_margin_m'])
    eligible = sorted([c for c in candidates if c['candidate'] != 0 and c['sphere_clear']],
                      key=lambda c: (c['ranking_score_m'], c['world_margin_m']), reverse=True)
    selected = []
    for candidate in eligible:
        if all(np.linalg.norm(np.array(candidate['joint_positions_rad']) - c['joint_positions_rad']) > .35 for c in selected):
            selected.append(candidate)
        if len(selected) == 6:
            break
    np.savez_compressed(args.output / 'coarse_paths.npz', joint_positions=path, sample_indices=indices)
    report = {'kind': 'coarse_geometric_screen', 'sample_count': len(indices), 'candidate_count': len(candidates),
        'collector_fk_max_error': float(fk_error), 'collector_jacobian_max_error': float(jac_error),
        'reference': candidates[0], 'candidates': candidates, 'selected': [c['candidate'] for c in selected],
        'scope': '20 ms ideal-tracking samples and conservative spheres for screening only; full 2 ms source-hull and dynamics checks are required.'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'finalists.json').write_text(json.dumps({'candidates': selected}, indent=2) + '\n')
    print(json.dumps({'reference': candidates[0], 'finalists': selected}, indent=2), flush=True)


if __name__ == '__main__':
    main()
