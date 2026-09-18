"""Solve the commanded path with calibrated IK; save it for source-hull checks.

This is geometric motion with ideal tracking, not a dynamics simulation.
Each solution starts at the previous joint angles to preserve the arm branch.
"""
import os
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "workstation"))
sys.path.insert(0, str(HERE.parent / "curobo_umi"))
import collect_thunder as collector
from reachability_geometry import batch_fk, pose_matrix

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--preview", type=Path, required=True)
parser.add_argument("--candidate", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
manifest = json.loads((args.preview / "preview.json").read_text())
native = np.load(args.preview / "trajectories.npz")
row_id = next(i for i, row in enumerate(manifest["rows"]) if row["candidate"] == args.candidate)
row = manifest["rows"][row_id]
collector.install_calibration()
q = np.array(row["q0"], dtype=float)
targets_p = native["targets_pos"][:, row_id]
targets_q = native["targets_quat"][:, row_id]
joint_rows, errors = [], []
failure = None
for t, (target_p, target_q) in enumerate(zip(targets_p, targets_q)):
    for iteration in range(50):
        p, quat = collector.kin.get_ee_pose(q)
        error = collector.kin.compute_pose_error(p, quat, target_p, target_q)
        if np.linalg.norm(error[:3]) < 1e-6 and np.linalg.norm(error[3:]) < 1e-5:
            break
        jacobian = collector.kin.compute_jacobian_calibrated(q)
        delta = np.linalg.solve(jacobian.T @ jacobian + 1e-8*np.eye(6), jacobian.T @ error)
        q += delta * min(1., .1 / max(np.abs(delta).max(), 1e-12))
    else:
        failure = {"sample": t, "time_s": t*manifest["dt"], "position_error_m": float(np.linalg.norm(error[:3])),
                   "rotation_error_rad": float(np.linalg.norm(error[3:]))}
        break
    joint_rows.append(q.copy()); errors.append(error.copy())
    if t % 1000 == 0:
        print(f"Solved {t}/{len(targets_p)} requested poses", flush=True)

summary = {"kind": "ideal_tracking_geometry", "source": str(args.preview), "candidate": args.candidate,
           "solved_samples": len(joint_rows), "requested_samples": len(targets_p), "failure": failure,
           "method": "Calibrated local inverse kinematics, previous solution as seed; no dynamics or controller tracking assumed"}
if failure is None:
    qs = np.array(joint_rows)
    audit_path = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917') + '/15_table_reachability/model/export_audit.json')
    audit = json.loads(audit_path.read_text())
    transforms = batch_fk(audit, qs)
    base_id = manifest["body_names"].index("base_link")
    base_pose = native["body_poses"][0, row_id, base_id].copy()
    base_pose[:3] -= native["env_origins"][row_id]
    base = pose_matrix(base_pose[:3], base_pose[3:])
    body = []
    for name in manifest["body_names"]:
        world = base @ transforms[name]
        xyzw = Rotation.from_matrix(world[:, :3, :3]).as_quat()
        body.append(np.c_[world[:, :3, 3], xyzw[:, [3, 0, 1, 2]]])
    body = np.stack(body, axis=1)[:, None]
    full = np.zeros((len(qs), 1, len(manifest["joint_names"])))
    full[:, 0, :6] = qs
    lower = np.array([next(j for j in audit["articulation_tree"] if j["name"] == name)["lower"] for name in collector.JOINT_NAMES])
    upper = np.array([next(j for j in audit["articulation_tree"] if j["name"] == name)["upper"] for name in collector.JOINT_NAMES])
    summary.update(max_position_error_m=float(np.linalg.norm(np.array(errors)[:, :3], axis=1).max()),
                   max_rotation_error_rad=float(np.linalg.norm(np.array(errors)[:, 3:], axis=1).max()),
                   max_joint_speed_rad_s=np.abs(np.gradient(qs, manifest["dt"], axis=0)).max(axis=0).tolist(),
                   max_joint_excursion_rad=np.abs(qs-row["q0"]).max(axis=0).tolist(),
                   within_joint_limits=bool(((qs >= lower) & (qs <= upper)).all()))
    manifest["rows"] = [{"candidate": args.candidate, "variant": "ideal_tracking_geometry", "q0": row["q0"],
                          "requires_direct_hull_check": True}]
    manifest["candidates"] = [c for c in manifest["candidates"] if c["candidate"] == args.candidate]
    manifest["scope"] = summary["method"]
    np.savez_compressed(args.output / "trajectories.npz", body_poses=body, full_joint_positions=full,
                        joint_velocities=np.gradient(qs, manifest["dt"], axis=0)[:, None], env_origins=np.zeros((1, 3)),
                        targets_pos=targets_p[:, None], targets_quat=targets_q[:, None])
    (args.output / "preview.json").write_text(json.dumps(manifest, indent=2)+"\n")
(args.output / "ik_report.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary, indent=2), flush=True)
