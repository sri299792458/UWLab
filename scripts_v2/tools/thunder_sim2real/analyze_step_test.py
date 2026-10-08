"""Summarize a Thunder step-response record (workstation/step_test_thunder.py; sim counterpart R223, phase B).

For every held step: pose of the calibrated flange frame from the recorded joint positions (vendored UWLab kinematics,
Thunder calibration), measured along the commanded axis relative to the start (center) pose:
  reached     = displacement at the end of the hold (mm or deg); stop_short = commanded - reached
  moved_0p1s  = displacement 0.1 s after the step began, relative to the pose just before the step
  onset_s     = first time the displacement from the pre-step pose exceeds 0.5 mm / 0.25 deg (None if never)
  return_left = displacement still remaining 1.5 s after the target returned to zero
Prints a table and writes <record>.steps.json next to the record.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "workstation"))
import collect_thunder as ct  # noqa: E402
from vendor import ur5e_kinematics as kin  # noqa: E402

AXES = ("x", "y", "z", "rx", "ry", "rz")


def pose_offsets(q, center_pos, center_quat):
    """(n, 6) flange displacement from the center pose: translation (m) and rotation vector (rad), base_link frame."""
    out = np.zeros((len(q), 6))
    conj = np.array([center_quat[0], -center_quat[1], -center_quat[2], -center_quat[3]])
    for i, qi in enumerate(q):
        pos, quat = kin.get_ee_pose(qi)
        out[i, :3] = pos - center_pos
        out[i, 3:] = kin.quat_to_axis_angle(ct.quat_multiply(quat, conj))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    rec = torch.load(args.record, map_location="cpu", weights_only=False)
    cfg = rec["collection_config"]
    schedule = cfg["step_test"]["schedule"]
    ct.install_calibration()
    q = rec["joint_positions"].numpy()
    center_pos, center_quat = kin.get_ee_pose(rec["initial_joint_pos"].numpy())
    n = len(q)
    rows = []
    for step in schedule:
        ax = AXES.index(step["axis"]); unit = 1000.0 if ax < 3 else 180.0 / np.pi
        s0, hold, r0, rest = step["start_sample"], step["hold_samples"], step["return_start_sample"], step["return_samples"]
        if s0 + hold > n:
            break                                                     # incomplete (stopped) recording
        sign = np.sign(step["value"])
        seg = pose_offsets(q[max(s0 - 1, 0): r0 + rest][::1], center_pos, center_quat)
        pre = seg[0, ax]
        along = (seg[:, ax] - pre) * sign                              # displacement from the pre-step pose, signed
        reached = (seg[hold, ax]) * sign
        thresh = 0.5e-3 if ax < 3 else np.deg2rad(0.25)
        moving = np.nonzero(along[1:hold + 1] > thresh)[0]
        ret_end = min(hold + rest, len(seg) - 1)
        rows.append({"axis": step["axis"], "commanded": round(abs(step["value"]) * unit, 3), "sign": int(sign),
                     "reached": round(reached * unit, 3), "stop_short": round((abs(step["value"]) - reached) * unit, 3),
                     "moved_0p1s": round(along[min(50, hold)] * unit, 3),
                     "onset_s": round((moving[0] + 1) * ct.DT, 3) if len(moving) else None,
                     "return_left": round(seg[ret_end, ax] * sign * unit, 3) if ret_end > hold else None})
    unit_name = lambda axis: "mm" if axis in AXES[:3] else "deg"
    print(f"{'axis':>4} {'cmd':>7} {'sign':>4} {'reached':>8} {'short':>8} {'0.1s':>7} {'onset_s':>8} {'return_left':>11}")
    for r in rows:
        print(f"{r['axis']:>4} {r['commanded']:>6}{unit_name(r['axis'])[0]} {r['sign']:>4} {r['reached']:>8} {r['stop_short']:>8} "
              f"{r['moved_0p1s']:>7} {str(r['onset_s']):>8} {str(r['return_left']):>11}")
    summary = {"record": str(args.record), "friction_mode": cfg["step_test"]["friction_mode"],
               "direct_torque_params": rec.get("direct_torque_params"), "completed": rec.get("completed"),
               "failure": rec.get("failure"), "steps": rows, "doc": __doc__}
    out = args.record.with_suffix(".steps.json")
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
