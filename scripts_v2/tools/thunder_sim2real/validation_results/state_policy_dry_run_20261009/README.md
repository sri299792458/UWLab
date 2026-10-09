# R214 state policy on Thunder: camera check and dry run — October 8–9, 2026

First runs of `workstation/run_state_policy.py` (commit `23b343b`) on the real cell. Nothing was sent to the robot or the
gripper: `check` and `dry-run` only read them. The policy ran on real observations at 10 Hz; its targets were logged.

| Folder | What |
| --- | --- |
| `dry9/` | `dry-run`, 30 s from the configured start: `summary.json` and `log.npz` (500 Hz state, every policy step's observation, action, target, cube poses and ages) |
| `check4/` | `check`: carried cube held in the closed gripper, camera pose of the cube in the wrist frame vs training grasps |
| `camera/` | `T_base_l515_20261008.json` (the `--camera-transform` used, frame `ur_base`) and the Spark calibration it came from |

## Camera transform

From the Spark rig calibration of October 8: Thunder wrist D405 hand-eye (Tsai, 15 poses, UR `getActualTCPPose` with the
TCP at the flange), then the L515 from shared views of a ChArUco board (DICT_5X5_50, 6 x 9, 30/22 mm). Spread over the 15
poses: board in base 1.7/4.1/3.1 mm and 0.54 deg; L515 4.6/3.6/3.9 mm and 0.54 deg. Checks on the cell:

- The runner's kinematics match the controller's TCP pose (TCP at the flange) to 0.0 mm / 0.001 deg.
- The table, measured through the L515 from the ChArUco board (10 mm thick) lying on it, was raised 8.7 mm relative to
  training's `table_top_y_m`; the table was lowered to -0.2 mm.
- `check4`: held carried cube at (-0.007, -0.012, 0.192) m in the wrist frame; training grasps median (-0.001, 0.001,
  0.192) m; `consistent_with_training_grasps` true. A frame or direction error would be off by decimeters.

## dry9

Start: arm at the configured start (R214 Reaching start bank row 7738; largest joint error 0.21 deg), gripper closed and
empty (227). Bottom cube resting on its -X face (1.5 deg tilt, 0.3 mm above the table, one tag visible), relabelled to
+Z up with a yaw error of 7.0 deg from training's -90 deg. Carried cube +Z up (label unchanged), 1.0 mm above the table.

| | |
| --- | --- |
| Stop reason | `max_episode_s` (30 s), no failure |
| Policy steps / robot cycles | 300 / 15,000; no duplicate or late (> 2.4 ms) robot-state intervals |
| Policy step compute | max 1.53 ms |
| Cubes over the run | relabelled +Z tilt <= 2.6 deg; bottom cube yaw -83 to -80 deg; position spread bottom 6/12/3 mm, carried 6/9/4 mm (x/y/z base_link; y is up); ages <= 0.32 s |
| Gripper action | open in all 300 steps |

Every observation term of the first step is inside the range of the recorded training observations (z-scores against
the policy's normalizer; maxima over the contract fixture's 4 training episodes, `test_data/r214_contract_fixture.npz`,
in brackets): insertive_in_receptive 2.6 (2.5), joint_pos 3.8 (3.8, the
closed empty gripper as in Reaching starts), end_effector_pose 3.5 (3.6, wrist 0.47 m above the table), insertive_pose
1.7 (3.6), receptive_pose 1.7 (3.6).

The arm does not move in a dry run, so after the first step the policy keeps responding to an unchanged state; the
actions are not a prediction of executed motion. The rotation about vertical (action[4], scale 0.2) stays near the top of
the training fixture's range (median |a| 6.0 vs fixture max 6.1).

## Earlier attempts (not archived)

- dry3–dry5: stopped at the start, cubes not yet detected; the L515's first frames come before its exposure settles. The
  runner now waits up to 5 s.
- dry7: completed, but the start reading of the bottom cube came from one tag and was tilted 35.5 deg; the face relabel
  chosen from it showed the policy the bottom cube on its side for the whole run. The runner now waits for a reading with
  both cubes resting flat (<= 10 deg).
- dry8: ended before the start checks without a recorded reason.

`SHA256SUMS` lists every file in this folder.
