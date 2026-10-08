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


def timing_review(times):
    """Keep gaps visible; no interpolation or replacement of missing samples."""
    intervals = np.diff(times)
    bad = np.flatnonzero(np.abs(intervals - ct.DT) > ct.STATE_INTERVAL_TOLERANCE_S)
    missing = np.maximum(np.rint(intervals / ct.DT).astype(int) - 1, 0)
    return {"clock": "robot_sample_times_s", "continuous_500hz": len(bad) == 0,
            "gap_intervals": [{"sample_index": int(i + 1), "interval_s": float(intervals[i]),
                               "missing_cycles": int(missing[i])} for i in bad],
            "missing_cycles": int(missing.sum()),
            "max_interval_s": float(intervals.max()) if len(intervals) else None,
            "observed_span_s": float(times[-1] - times[0]) if len(times) else 0.0}


def gaps_in(times, start, end):
    indices = np.arange(1, len(times))
    bad = np.abs(np.diff(times) - ct.DT) > ct.STATE_INTERVAL_TOLERANCE_S
    return int(np.sum(bad & (indices >= start) & (indices <= end)))


def analyze_policy(q, schedule, times):
    """--mode policy trials: displacement along the commanded axis from the pose just before the trial, at the end of every
    0.1 s policy step (mm or deg). after_first = end of step 1; after_commands = end of the last commanded step;
    after_trial = end of the zero-action tail; per_step_commanded = mean motion per commanded step."""
    rows = []
    for st in schedule:
        ax = AXES.index(st["axis"]); unit = 1000.0 if ax < 3 else 180.0 / np.pi
        s0, k, m, reps = st["start_sample"], st["policy_steps"], st["step_samples"], st["repeats"]
        r0 = st["return_start_sample"]
        if s0 + k * m >= len(q):
            break
        pre_pos, pre_quat = kin.get_ee_pose(q[s0 - 1])
        boundaries = s0 + np.arange(k + 1) * m
        # These are pre-command states at the next policy update: the preceding
        # command has had its full interval to act. Never invent missing states.
        ends = pose_offsets(q[boundaries[1:]], pre_pos, pre_quat)[:, ax] * np.sign(st["value"]) * unit
        end = min(r0 + st["return_samples"], len(q) - 1)
        return_left = pose_offsets(q[end:end + 1], pre_pos, pre_quat)[0, ax] * np.sign(st["value"]) * unit
        rows.append({"kind": st["kind"], "axis": st["axis"], "command_per_step": round(abs(st["value"]) * unit, 3),
                     "sign": int(np.sign(st["value"])),
                     "repeats": reps, "after_first": round(ends[0], 3), "after_commands": round(ends[reps - 1], 3),
                     "after_trial": round(ends[-1], 3), "per_step_commanded": round(ends[reps - 1] / reps, 3),
                     "step_ends": [round(v, 3) for v in ends],
                     "policy_step_durations_s": [round(float(v), 6) for v in np.diff(times[boundaries])],
                     "trial_elapsed_s": float(times[r0] - times[s0]),
                     "gaps_during_trial": gaps_in(times, s0, r0),
                     "gaps_during_return": gaps_in(times, r0 + 1, end),
                     "return_left": round(float(return_left), 3),
                     "return_elapsed_s": float(times[end] - times[r0]),
                     "return_complete": r0 + st["return_samples"] <= len(q)})
    return rows


def analyze_held(q, schedule, times, center_pos, center_quat):
    rows = []
    for step in schedule:
        ax = AXES.index(step["axis"]); unit = 1000.0 if ax < 3 else 180.0 / np.pi
        s0, r0, rest = step["start_sample"], step["return_start_sample"], step["return_samples"]
        if r0 >= len(q):
            break  # The hold endpoint state was not observed.
        sign = np.sign(step["value"])
        base = max(s0 - 1, 0)
        end = min(r0 + rest, len(q) - 1)
        seg = pose_offsets(q[base:end + 1], center_pos, center_quat)
        along = (seg[:, ax] - seg[0, ax]) * sign
        reached = seg[r0 - base, ax] * sign
        thresh = 0.5e-3 if ax < 3 else np.deg2rad(0.25)
        moving = np.flatnonzero(along[s0 - base:r0 - base + 1] > thresh) + s0
        # Last observed state at or before 0.1 seconds after this target starts.
        early = min(r0, int(np.searchsorted(times, times[s0] + 0.1 + 1e-9, side="right") - 1))
        rows.append({"axis": step["axis"], "commanded": round(abs(step["value"]) * unit, 3), "sign": int(sign),
                     "reached": round(reached * unit, 3), "stop_short": round((abs(step["value"]) - reached) * unit, 3),
                     "moved_0p1s": round(along[early - base] * unit, 3),
                     "moved_0p1s_observed_at_s": float(times[early] - times[s0]),
                     "onset_s": round(float(times[moving[0]] - times[s0]), 3) if len(moving) else None,
                     "hold_elapsed_s": float(times[r0] - times[s0]),
                     "return_left": round(seg[end - base, ax] * sign * unit, 3),
                     "return_elapsed_s": float(times[end] - times[r0]),
                     "return_complete": r0 + rest <= len(q),
                     "gaps_during_hold": gaps_in(times, s0, r0),
                     "gaps_during_return": gaps_in(times, r0 + 1, end)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    rec = torch.load(args.record, map_location="cpu", weights_only=False)
    cfg = rec["collection_config"]
    schedule = cfg["step_test"]["schedule"]
    ct.install_calibration()
    q = rec["joint_positions"].numpy()
    times = np.asarray(rec["robot_sample_times_s"], dtype=float)
    if times.shape != (len(q),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Analysis requires one finite, increasing robot timestamp per recorded state")
    timing = timing_review(times)
    print("Timing:", json.dumps(timing))
    if cfg["step_test"].get("mode", "held") == "policy":
        rows = analyze_policy(q, schedule, times)
        print(f"{'kind':>8} {'axis':>4} {'cmd/step':>9} {'reps':>4} {'1st step':>9} {'after cmds':>10} {'after 1 s':>9}")
        for r in rows:
            print(f"{r['kind']:>8} {r['axis']:>4} {r['command_per_step']:>9} {r['repeats']:>4} {r['after_first']:>9} "
                  f"{r['after_commands']:>10} {r['after_trial']:>9}")
        summary = {"record": str(args.record), "mode": "policy", "friction_mode": cfg["step_test"]["friction_mode"],
                   "direct_torque_params": rec.get("direct_torque_params"), "control_script": rec.get("control_script"),
                   "completed": rec.get("completed"), "failure": rec.get("failure"),
                   "timing": timing, "state_gap_policy": rec.get("state_gap_policy"),
                   "trials": rows, "doc": analyze_policy.__doc__}
        out = args.record.with_suffix(".steps.json")
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print("wrote", out)
        return
    center_pos, center_quat = kin.get_ee_pose(rec["initial_joint_pos"].numpy())
    rows = analyze_held(q, schedule, times, center_pos, center_quat)
    unit_name = lambda axis: "mm" if axis in AXES[:3] else "deg"
    print(f"{'axis':>4} {'cmd':>7} {'sign':>4} {'reached':>8} {'short':>8} {'0.1s':>7} {'onset_s':>8} {'return_left':>11}")
    for r in rows:
        print(f"{r['axis']:>4} {r['commanded']:>6}{unit_name(r['axis'])[0]} {r['sign']:>4} {r['reached']:>8} {r['stop_short']:>8} "
              f"{r['moved_0p1s']:>7} {str(r['onset_s']):>8} {str(r['return_left']):>11}")
    summary = {"record": str(args.record), "friction_mode": cfg["step_test"]["friction_mode"],
               "direct_torque_params": rec.get("direct_torque_params"), "completed": rec.get("completed"),
               "failure": rec.get("failure"), "timing": timing, "state_gap_policy": rec.get("state_gap_policy"),
               "steps": rows, "doc": __doc__}
    out = args.record.with_suffix(".steps.json")
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
