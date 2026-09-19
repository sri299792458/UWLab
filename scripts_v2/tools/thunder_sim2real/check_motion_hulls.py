"""Check saved native body poses against the source collision hulls and lab boxes."""
import os
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts_v2/tools/curobo_umi"))
from reachability_geometry import SourceCollisionChecker, exact_pair_gap, set_transform

ROOT = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--preview", type=Path, required=True)
parser.add_argument("--candidate", type=int)
parser.add_argument("--include_sphere_failures", action="store_true",
                    help="Check exact source geometry even when conservative sphere screening fails")
args = parser.parse_args()
report = json.loads((args.preview / "preview.json").read_text())
trajectory = np.load(args.preview / "trajectories.npz")
bodies = trajectory["body_poses"]
checker = SourceCollisionChecker(ROOT / "15_table_reachability/model", clearance=.001)
map_cfg = json.loads((ROOT / "49_clearance_and_placement/atlas/config.json").read_text())
column_ids = [checker.world_names.index(name) for name in map_cfg["column_box_names"]]
required = np.full(len(checker.world_names), .001); required[column_ids] = .05
shape_ids = [report["body_names"].index(s["body"]) for s in checker.shapes]
base_id = report["body_names"].index("base_link")
# Native finger poses are used: reinstate pairs pruned only because the atlas
# hand was frozen open. Retain physical joint-adjacency and same-body exclusions.
adjacent = {frozenset((j["body0"], j["body1"])) for j in checker.audit["source_joints"]}
pairs = np.array([(i, j) for i, j in itertools.combinations(range(len(checker.shapes)), 2)
                  if checker.shapes[i]["body"] != checker.shapes[j]["body"]
                  and frozenset((checker.shapes[i]["body"], checker.shapes[j]["body"])) not in adjacent])
a, b = pairs.T
dynamic_pair = np.array([not (checker.shapes[i]["body"] in checker.fixed_hand and
                             checker.shapes[j]["body"] in checker.fixed_hand) for i, j in pairs])
passed_candidates = [c["candidate"] for c in report["candidates"] if all(
    args.include_sphere_failures or row.get("sphere_clear_all_samples", False) or row.get("requires_direct_hull_check", False)
    for row in report["rows"] if row["candidate"] == c["candidate"])]
if args.candidate is not None:
    passed_candidates = [candidate for candidate in passed_candidates if candidate == args.candidate]
output = {"source": str(args.preview), "sample_dt_s": report["dt"],
          "include_sphere_failures": args.include_sphere_failures, "candidates": []}

for candidate in passed_candidates:
    indices = [i for i, row in enumerate(report["rows"]) if row["candidate"] == candidate]
    candidate_report = {"candidate": candidate, "variants": []}
    for index in indices:
        poses = bodies[:, index]
        base_p = poses[:, base_id, :3]
        base_r = Rotation.from_quat(poses[:, base_id, 3:][:, [1, 2, 3, 0]]).as_matrix()
        shape_p = poses[:, shape_ids, :3]
        shape_r = Rotation.from_quat(poses[:, shape_ids, 3:][..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(len(poses), -1, 3, 3)
        rotations = np.einsum("tji,tsjk->tsik", base_r, shape_r)
        positions = np.einsum("tji,tsj->tsi", base_r, shape_p-base_p[:, None])
        transforms = np.zeros((len(poses), len(shape_ids), 4, 4))
        transforms[..., :3, :3] = rotations; transforms[..., :3, 3] = positions; transforms[..., 3, 3] = 1
        centers = np.einsum("tsij,sj->tsi", rotations, checker.local_centers)+positions
        extents = np.einsum("tsij,sj->tsi", np.abs(rotations), checker.local_extents)
        world_lower = np.linalg.norm(np.maximum(np.abs(centers[:, :, None]-checker.world_centers)-
                                                extents[:, :, None]-checker.world_extents, 0), axis=-1)
        world_lower[:, ~checker.world_mask] = np.inf
        self_lower = np.linalg.norm(np.maximum(np.abs(centers[:, a]-centers[:, b])-extents[:, a]-extents[:, b], 0), axis=-1)
        best = {"world_margin_m": np.inf, "world_gap_m": np.inf, "self_gap_m": np.inf, "moving_self_gap_m": np.inf}
        nearest = {}
        calls = 0
        for t in range(len(poses)):
            world_possible = ((world_lower[t]-required) < best["world_margin_m"]+1e-9) | (world_lower[t] < best["world_gap_m"]+1e-9)
            world_possible &= checker.world_mask
            self_possible = (self_lower[t] < best["self_gap_m"]+1e-9) | (dynamic_pair & (self_lower[t] < best["moving_self_gap_m"]+1e-9))
            wp = np.argwhere(world_possible); sp = np.flatnonzero(self_possible)
            needed = set(wp[:, 0].tolist()) | set(pairs[sp].ravel().tolist())
            for i in needed:
                set_transform(checker.objects[i], transforms[t, i])
            for k in sp:
                i, j = pairs[k]; gap = exact_pair_gap(checker.objects[i], checker.objects[j]); calls += 1
                for key, enabled in (("self_gap_m", True), ("moving_self_gap_m", dynamic_pair[k])):
                    if enabled and gap < best[key]:
                        best[key] = gap; nearest[key] = {"time_s": t*report["dt"], "a": checker.shapes[i]["name"], "b": checker.shapes[j]["name"]}
            for i, j in wp:
                gap = exact_pair_gap(checker.objects[i], checker.world_objects[j]); calls += 1
                for key, value in (("world_gap_m", gap), ("world_margin_m", gap-required[j])):
                    if value < best[key]:
                        best[key] = value; nearest[key] = {"time_s": t*report["dt"], "a": checker.shapes[i]["name"], "b": checker.world_names[j]}
        entry = {"variant": report["rows"][index]["variant"], "samples": len(poses), "exact_distance_calls": calls,
                 **best, "nearest_pairs": nearest, "clear": bool(best["self_gap_m"] >= .001 and best["world_margin_m"] >= 0)}
        candidate_report["variants"].append(entry)
        print(json.dumps({"candidate": candidate, **entry}), flush=True)
    candidate_report["clear"] = all(v["clear"] for v in candidate_report["variants"])
    output["candidates"].append(candidate_report)
    (args.preview / "source_hull_check.json").write_text(json.dumps(output, indent=2)+"\n")
    if candidate_report["clear"]:
        # One complete, independently checked candidate is enough to present.
        output["selected_candidate"] = candidate
        break
output["scope"] = "All 2 ms saved body samples; source convex hulls, 33 lab boxes, same-body/adjacent-link and fixed mounting exclusions. Source manifest distinguishes physics states from ideal-tracking geometry. No unknown real-world obstacles modeled."
(args.preview / "source_hull_check.json").write_text(json.dumps(output, indent=2)+"\n")
print(json.dumps({"selected_candidate": output.get("selected_candidate"), "tested_candidates": len(output["candidates"])}))
