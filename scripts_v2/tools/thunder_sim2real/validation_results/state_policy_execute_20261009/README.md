# R214 state policy on Thunder: first execute — October 9, 2026

First `execute` of `workstation/run_state_policy.py` (commit `154a4db`) on the real cell. It ended after 0.26 s when the UR
safety system stopped the robot: **safety status VIOLATION, robot powered off**. Our runner did not stop it.

| File | What |
| --- | --- |
| `run1/summary.json`, `run1/log.npz` | the run: start checks, 500 Hz state and torques, the 3 policy steps |
| `first_test.state_policy.json` | config used: the committed `collection.state_policy.json` with `max_episode_s` 16 (training's episode length) |

Camera transform: identical (same SHA-256) to `../state_policy_dry_run_20261009/camera/T_base_l515_20261008.json`.

## Start

Arm at the configured start (largest joint error 0.21 deg), gripper closed and empty (227). Bottom cube resting on its -X
face (2.3 deg tilt), relabelled +Z up with a yaw error of 6.7 deg; carried cube +Z up. Same scene as the dry run.

## What happened

| | |
| --- | --- |
| Stop | `RuntimeError: directTorque returned failure` after 131 cycles (0.26 s), 3 policy steps |
| Robot afterwards (dashboard) | `Safetystatus: VIOLATION`, `Robotmode: POWER_OFF` |
| Peak joint speed (deg/s) | base 30.8, shoulder 8.5, elbow 45.6, wrist 1 124.9, wrist 2 39.9, **wrist 3 206.1** |
| Peak joint torque (Nm) | 52.4, 31.4, 36.8, **28.0**, 8.7, **28.0** (wrist 1 and 3 at the 28 Nm limit) |
| Joint motion (deg) | 5.4, 0.9, 7.8, -17.8, 7.3, **35.6** |
| Wrist | moved (-6, -67, 50) mm; peak 0.50 m/s, peak angular speed 3.98 rad/s (runner stop: 4 rad/s) |
| Policy actions | as in the dry run: large rotation about vertical (action[4] -5.8 to -6.8, scale 0.2, i.e. ~1.2 rad per step); gripper open |

## Likely cause

Training clamps joint speeds in simulation (`UR5E_VELOCITY_LIMITS`: wrists 3.14 rad/s = 180 deg/s; base, shoulder, elbow
1.57 rad/s), and the policy's actions push wrist rotation against that clamp. The real torque loop has no such clamp:
wrist 3 reached 206 deg/s (3.6 rad/s), faster than anything in training, and most likely past the UR safety joint speed
limit, giving the violation. The runner's stops check wrist (TCP) speed, not joint speeds. **To confirm:** the pendant's
error text and the joint speed limits in Safety > Joints.

Proposed (not yet implemented): reproduce the simulator's joint velocity clamp in the torque loop (damping beyond ~90% of
each joint's sim limit), plus a per-joint speed stop just above the sim limits, below the UR safety limit.

`SHA256SUMS` lists every file in this folder.
