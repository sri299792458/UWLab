# Running the R214 state policy on Thunder

`run_state_policy.py` runs R214 (broad 60 mm cube stacking, official Robotiq + UMI fingertips, hardware arm profile; final
checkpoint 1999) with cube poses from the external L515 and AprilCube. Work through the four modes in order; each later mode
does everything the earlier ones do. Nothing moves before `execute`, and `execute` asks you to type `GO`.

## What is matched to training (and how it was checked, R226)

| Piece | Deployment | Check |
|---|---|---|
| Policy | `state_policy/r214/policy_r214_1999_weights.npz`, NumPy forward pass (0.08 ms) | equals the actions R214 produced in its training sim (checked at every load) and its TorchScript export (tests) |
| Observation (215 numbers) | `state_policy/obs.py` | rebuilds all 40,960 recorded sim observations to 1e-6 |
| Wrist pose | calibrated kinematics, base_link | matches the sim wrist pose to 0.003 mm / 0.06 deg |
| Gripper joints (6 in the observation) | Robotiq position -> finger_joint (anchors 3 / 92 / 226 -> 0 / 0.288 / 0.819 rad) -> 5 linkage joints | linkage p99 error <= 0.013 rad; the position->angle map between anchors is an approximation |
| Arm action | every 0.1 s: target = wrist pose + (0.02, 0.02, 0.02, 0.02, 0.2, 0.02) x action, Kp 1000/50, ratio 1 | same rule as training (unit test); same 500 Hz loop and OSC as the step tests |
| Gripper action | action[6] < 0 closes, else opens; speed 128, force 0 | as decided |
| Cubes | bottom cube = tags 10-15, carried cube = tags 20-25 (G1 detector configs) | cube frame = sim cube frame (same aprilcube source) |

Three things are ours, not training's: the safety stops in `collection.state_policy.json` "limits", latency compensation
of the held cube (while the gripper reports holding it, its last camera pose is moved with the wrist from the later of
capture and grasp to now, using the logged joint angles, with no age limit), and cube face relabelling (`state_policy/cube_symmetry.py`:
the cubes are physically symmetric, so at the start of a run each cube's frame is relabelled once to +Z up, the bottom
cube at the +Z-up yaw nearest training's -90 deg; a placement like training's is left unchanged).

## Setup

- Install in the workstation environment: `pyrealsense2` and AprilCube (`pip install -e <your aprilcube clone>`, commit
  `80ed7c7`). The runner needs no PyTorch.
- **Camera transform**: a 4x4 matrix in meters with `p_base = T @ p_camera`, camera = L515 **color** optical frame.
  Save it as `.npy`, or as `.json` with key `T_base_camera`. Pass `--camera-transform-frame ur_base` if it was computed
  from `getActualTCPPose` / pendant poses (UR controller Base), or `base_link` if computed in the sim / our kinematics
  frame. The two differ by 180 deg about the base z axis, which on Thunder flips up/down; this is what flipped the
  October 6 pull-out direction.
- **Cubes**: bottom cube = tags 10-15, carried cube = tags 20-25, each resting flat on any face (faces are relabelled).
  The bottom cube's yaw must be within 15 deg of training's after relabelling: `track` reports `bottom_cube_yaw_error_deg`,
  and dry-run / execute refuse to start beyond +/-15. Keep both inside the trained region (`inside_trained_region`).
- **Gripper**: activated on the pendant (the runner never sends activation motion), then **closed and empty**: training
  Reaching / Near-Object starts have the gripper closed. `execute` refuses to start otherwise.
- **Arm start**: one fixed pose, `start_joint_positions_rad` in the config = (15.6, -145.8, -70.3, -144.6, -160.0, 96.0) deg.
  It is a training start (R214 Reaching start bank, row 7738): S-W-E-, gripper 4 deg from straight down, wrist 0.47 m above
  the table, wrist 3 at 96 deg, and the arm stays >= 0.12 m from the L515's lines of sight to the whole trained cube region.
  Every joint must be within 0.02 rad of it. Bring the arm within 10 deg with the pendant, then
  `python move_to_start.py --config collection.state_policy.json --execute`.

## Steps

```bash
# 1) camera only: both cubes detected, bottom_above_table_mm ~ 0, inside_trained_region true, |bottom_cube_yaw_error_deg| <= 15
python run_state_policy.py --mode track --camera-transform <T.npy> --camera-transform-frame ur_base --output ~/thunder_policy/track1
```
```bash
# 2) close the gripper on the carried cube (pendant), hold the arm still: held_cube_in_wrist_m must match training
#    (median [-0.001, 0.001, 0.192] m; consistent_with_training_grasps true). A frame or direction error is off by decimeters.
python run_state_policy.py --mode check --camera-transform <T.npy> --camera-transform-frame ur_base --output ~/thunder_policy/check1
```
```bash
# 3) arm at the start, gripper closed and empty, cubes placed: the policy runs on real observations, nothing is sent.
./run_realtime.sh python run_state_policy.py --mode dry-run --camera-transform <T.npy> --camera-transform-frame ur_base --output ~/thunder_policy/dry1
```
```bash
# 4) execute (types GO to start; Ctrl-C or any safety stop hands over to a position hold, as in the collectors)
./run_realtime.sh python run_state_policy.py --mode execute --camera-transform <T.npy> --camera-transform-frame ur_base --output ~/thunder_policy/run1
```

Each run writes `summary.json` (start-pose and cube reports, stop reason, timing) and, for dry-run / execute, `log.npz`
(500 Hz joints, torques, targets; every policy step's observation, action, cube poses and ages, gripper position).

## Safety stops (`limits`)

Stops (then torque-to-hold) on: wrist outside the R214 rollout wrist box + 5 cm, wrist lower than 0.11 m above the table
(training minimum 0.133 m), a joint outside the training range + 0.2 rad, wrist speed > 1.0 m/s or > 4 rad/s, a policy target
more than 0.3 m outside the box, a stale cube pose (bottom cube 10 s; free carried cube 2 s; a held cube has no limit), a stale
gripper position (0.2 s), a robot-state gap over 6 ms, or 30 s elapsed. R214 moves up to about 1 m/s in sim (R220): lower
the speed limits for cautious first runs, knowing the policy may then hit them during normal motion.

Tests: `python -m unittest test_state_policy` (no robot, camera or gripper needed).
