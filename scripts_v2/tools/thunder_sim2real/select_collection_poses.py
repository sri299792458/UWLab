"""Rank existing collision-clear cuRobo atlas seeds for a collection preview."""
import os
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from motion_geometry import REPO, UmiSphereGeometry, clearances

sys.path.insert(0, str(REPO / "scripts_v2/tools/curobo_umi"))
from reachability_geometry import batch_fk, SourceCollisionChecker

ROOT = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/umi_reset_from_defaults_20260911'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--count", type=int, default=12)
parser.add_argument("--height_min", type=float, default=.35)
parser.add_argument("--height_max", type=float, default=.55)
args = parser.parse_args()
cfg = json.loads((ROOT / "49_clearance_and_placement/atlas/config.json").read_text())
audit = json.loads((ROOT / "15_table_reachability/model/export_audit.json").read_text())
seeds = np.load(ROOT / "49_clearance_and_placement/lookup/atlas_joint_seeds.npy", mmap_mode="r")
rng = np.random.default_rng(73)
# Select the requested height band over the middle of the accessible table.
xs, ys, hs = (np.asarray(cfg[k]) for k in ("x_world_m", "y_world_m", "height_above_table_m"))
allowed_x = np.flatnonzero((xs >= -.15) & (xs <= .45))
allowed_y = np.flatnonzero((ys >= -.32) & (ys <= .10))
allowed_h = np.flatnonzero((hs >= args.height_min) & (hs <= args.height_max))
allowed_o = [o["index"] for o in cfg["orientations"] if o["tilt_deg"] <= 60]
indices = np.column_stack([rng.choice(a, 12000) for a in (allowed_h, allowed_y, allowed_x, allowed_o)])
q = np.array(seeds[tuple(indices.T)], dtype=np.float32)
valid = np.isfinite(q).all(axis=1) & (np.abs(np.sin(q[:, 4])) > .4) & (np.abs(np.sin(q[:, 2])) > .4)
q, indices = q[valid], indices[valid]
# Equivalent whole turns do not create a different physical starting pose.
q = ((q+np.pi) % (2*np.pi)-np.pi).astype(np.float32)
body_names = list(dict.fromkeys(["base_link"]+[j["child"] for j in audit["articulation_tree"]]))
adapter = UmiSphereGeometry(body_names, "cuda")
poses = batch_fk(audit, q)
base_r = Rotation.from_quat(np.array(cfg["robot_base_quaternion_world_wxyz"])[[1, 2, 3, 0]]).as_matrix()
base_p = np.array(cfg["robot_base_position_world_m"])
positions = np.stack([poses[name][:, :3, 3]@base_r.T+base_p for name in body_names], axis=1)
rotations = np.stack([base_r@poses[name][:, :3, :3] for name in body_names], axis=1)
metrics = clearances(adapter, torch.tensor(positions, device="cuda", dtype=torch.float32),
                     torch.tensor(rotations, device="cuda", dtype=torch.float32), torch.zeros(len(q), 3, device="cuda"))
metrics = {key: value.cpu().numpy() for key, value in metrics.items()}
eligible = (metrics["self_margin_m"] >= 0) & (metrics["world_margin_m"] >= .025)
rank = np.argsort(np.where(eligible, metrics["world_margin_m"], -np.inf))[::-1]
checker = SourceCollisionChecker(ROOT / "15_table_reachability/model", clearance=.001)
chosen = []
for row in rank:
    if not eligible[row]:
        break
    if any(np.linalg.norm(q[row]-entry["joint_positions_rad"]) < .8 for entry in chosen):
        continue
    if not checker.evaluate(q[row:row+1])["clear"][0]:
        continue
    hi, yi, xi, oi = indices[row]
    chosen.append({"candidate": len(chosen), "atlas_index_h_y_x_o": indices[row].tolist(),
                   "tool_position_world_m": [float(xs[xi]), float(ys[yi]), float(cfg["tabletop_z_m"]+hs[hi])],
                   "tool_orientation": cfg["orientations"][int(oi)],
                   "joint_positions_rad": q[row].tolist(),
                   "joint_positions_deg": np.degrees(q[row]).tolist(),
                   **{key: float(value[row]) for key, value in metrics.items()}})
    if len(chosen) >= args.count:
        break
if not chosen:
    raise RuntimeError("No sufficiently clear initial candidates found in the sampled atlas region")
report = {"sampled_entries": len(indices), "eligible_entries": int(eligible.sum()),
          "height_band_m": [args.height_min, args.height_max],
          "source_atlas": str(ROOT / "49_clearance_and_placement/atlas"),
          "selection": "Rank by remaining world clearance, with non-singular elbow/wrist and source-hull endpoint checks",
          "candidates": chosen}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
