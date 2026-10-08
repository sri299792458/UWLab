# Controller step-response test (hardware counterpart of sim R223)

Measures how far Thunder's flange actually moves toward a **held** Cartesian target offset under the training controller
(OSC Kp 1000/1000/1000/50/50/50, damping ratio 1, torque limits 150/28): the static-friction dead band and "stop short" error.
`--mode policy` instead re-anchors the target to the measured pose every 0.1 s, as the trained policy's actions do.
Run with UR friction compensation **off** (the fitted setting) and **on** (URScript default per-joint scales, applied through
the fixed control script below).

Setup: same as the clean R187 sysid recording. Gripper open and empty, payload 1.04 kg / CoG (-0.002, 0.001, 0.047) as
configured on the controller, start pose c675 (S-W-E-) from `collection.remapped_full_amplitude_swe_candidate.json`. Move the arm
to the start pose yourself; the script refuses to run outside 0.02 rad of it or if the arm is moving.

After checking clearance, an already nearby arm can return to that pose with
`python move_to_start.py --config collection.step_test.json --execute`. This separate helper uses a standard asynchronous
`moveJ` at 0.05 rad/s and 0.1 rad/s². It refuses any initial joint error above 5 degrees, a moving arm or an unsafe/not-ready
robot state, checks the pose again after connecting control, and verifies the final pose and rest state. Ctrl-C or its
30-second timeout requests `stopJ` before cleanup. Without `--execute`, it prints the target without connecting. Do not
run another robot-control program alongside it. It does not plan collision avoidance.

Sequence (169 s, 48 steps): for each base-frame axis x, y (vertical), z, rx, ry, rz and sizes 5/10/20/40 mm or 2/4/8/16 deg,
both signs: hold the target offset 2 s, return to zero for 1.5 s. From an exactly centered pose, the largest offset gives
an initial spring component of 40 N / 14 N m; these are not bounds on the dynamic wrench, which also includes damping.
The joint guard stops the
run if any joint leaves the IK-predicted range + 5 deg. Keep the e-stop in hand.

Collection also stops if successive consistent robot states are not 2 ms apart (tolerance 0.4 ms), or if computation
leaves a state more than 20 ms old before sending its torque command. Partial records retain `completed=False` and
the failure reason. These checks detect timing failures; they do not prevent host or network stalls. A completed OFF
record must still pass `records.py validate` before fixed-step fitting. ON records are for comparison only.

The collector runs a full Python garbage collection before connecting, defers automatic cyclic collection during
torque control, and restores the caller's GC setting after robot/scheduler cleanup, before saving. This avoids the
generation-2 GC pause reproduced around 35,400 samples in the October 8 full runs. Reference counting remains active;
the recorder retains samples for this bounded sequence. This does not provide hard real-time kernel behavior.

Before the first torque command, 16 controller calculations warm the same computation path without sending commands.
The first sample then waits for a fresh robot frame, just like later samples. Captures include startup metadata and
host timestamps after each `directTorque` call so SDK-call latency can be distinguished from computation latency.

To check receive delivery and controller computation under current workstation load, use
`python audit_step_timing.py --config collection.step_test.json --output ~/thunder_step_test/timing_audit.json`.
This opens only the RTDE receive interface and computes targets in memory; it sends zero robot-control commands.
The JSON/NPZ outputs count missed cycles and are diagnostic evidence, not dynamics-fitting recordings. It does not
exercise torque submission. Torque collection retains the strict 2 ms continuity check.

The two completed October 8 full runs and their timing review are preserved in
[`../validation_results/step_response_hardware_20261008/`](../validation_results/step_response_hardware_20261008/README.md).

## ur_rtde 1.6.5 friction-compensation bug (October 8 ON run had compensation OFF)

In ur_rtde 1.6.5 (latest PyPI release), the PolyScope >= 5.26 branch of the `direct_torque` command in the compiled-in
`rtde_control.script` reads the requested scales into `viscous_scale` / `couloumb_scale`, but then passes
`viscous_scaling` / `coulomb_scaling`, which stay at their zero initialization. Every scale request is therefore silently
ignored. ur_rtde 1.6.4 passed the read values correctly. **The October 8 `--friction ur_default` record therefore ran
with zero compensation**, the same physical setting as the OFF run; both runs agree to about 0.1 mm.

`vendor/ur_rtde_1_6_5/` holds the verbatim 1.6.5 script, its MIT license, and `rtde_control_fixed.script`, which changes
only these two register reads (see `PROVENANCE.json`; `test_step_test.py` checks the diff and hashes). When nonzero
scales are requested, `collect_thunder.collect()` uploads the fixed script with `setCustomScriptFile` before setting the
payload, waits up to 5 s for it to run, and aborts before any torque command if it does not start. Zero-compensation runs
keep the stock script. Each record stores `control_script` (source, path, sha256). The script prints
`"control_script": "vendor fixed (ur_rtde 1.6.5 bug)"` in its plan for `ur_default`.

Hardware check of the fix: only a clear difference between the OFF and ON held-step records shows that compensation is
applied. If ON again matches OFF to about 0.1 mm, stop and report it; do not tune the scales.

## Modes and configs

* `--mode held` (default; sim R223 phase B): a fixed target offset from the start pose, held for 2 s, then 1.5 s at zero.
* `--mode policy` (sim R223 phases A/C): **the trained policy's action semantics**. Every 0.1 s, the target is set to the
  measured flange pose plus the offset (target = current pose + scale x action). Configured in
  `collection.step_test.json` -> `policy_test`, for every axis and both signs:
  * single: one 0.1 s command of 10/20/40 mm or 2/4/8/16 deg, then zero action (target = current pose);
  * constant: the same command repeated (10 mm x 4, 20 mm x 2; 4 deg x 4, 8 deg x 2), then zero action.
  Each trial lasts 10 policy steps (1 s). The target then returns to the initial center pose for 2 s. The cumulative
  command per trial is capped at the held test's 40 mm / 16 deg, so the joint guard is the same as in held mode.
  66 trials, 199 s.
* `collection.step_test_fine.json` (held mode): a finer grid around the October 8 dead band, 15/25/30 mm and
  0.5/1/1.5 deg, with all other settings identical. 36 steps, 127 s.

```bash
# 1) print a plan only (no robot connection); add --mode policy or use the fine config the same way
python step_test_thunder.py --config collection.step_test.json --friction off
# 2) policy semantics, compensation off
./run_realtime.sh python step_test_thunder.py --config collection.step_test.json --mode policy --friction off \
    --execute --output ~/thunder_step_test/policy_off.pt
# 3) fine held grid, compensation off
./run_realtime.sh python step_test_thunder.py --config collection.step_test_fine.json --friction off \
    --execute --output ~/thunder_step_test/fine_off.pt
# 4) compensation ON via the fixed script (UR default scales viscous 0.9/0.9/0.8/0.9/0.9/0.9, coulomb 0.8/0.8/0.7/0.8/0.8/0.8):
#    held first, which is directly comparable with the October 8 OFF run, then policy semantics
./run_realtime.sh python step_test_thunder.py --config collection.step_test.json --friction ur_default \
    --execute --output ~/thunder_step_test/step_ur_default_fixed.pt
./run_realtime.sh python step_test_thunder.py --config collection.step_test.json --mode policy --friction ur_default \
    --execute --output ~/thunder_step_test/policy_ur_default_fixed.pt
# 5) summary table (also writes <record>.steps.json; detects held/policy from the record)
python ../analyze_step_test.py ~/thunder_step_test/policy_off.pt
```

Requires PolyScope >= 5.25.1 for the per-joint scales (Thunder: 5.26.1) and ur-rtde 1.6.5. Records with compensation on are
for comparison only; `records.validate_record` rejects them for fitting.
