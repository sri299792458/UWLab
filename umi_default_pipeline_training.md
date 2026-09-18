# UMI training with the regenerated reset bank

## Current mount and training status

The active configuration now uses the physically corrected mounting quaternion
`[0.5, 0.5, 0.5, 0.5]` and a freshly computed **20 mm** map in
`/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917`.
See [MOUNTING_CORRECTION.md](scripts_v2/tools/thunder_sim2real/MOUNTING_CORRECTION.md)
and the [portable handoff](scripts_v2/tools/thunder_sim2real/portable/README.md).
The old 10 mm data and live training checkout remain separately preserved.
No new long training run is launched by the mount/data update.

The existing Stage-1 run did launch on September 17 using the previous mount and banks.
At checkpoint 1000, a 256-trial-per-family deterministic replay finds zero stack success
from the three harder families. Partly assembled starts are already successful in
108/256 trials; holding gives 113/256 final successes, while the learned policy gives
100/256. It commands the gripper open on 96–100% of actions. The always-zero
`progress_context` metric is intentional bookkeeping; the real issue is the learned
behavior. Correcting the mount alone does not demonstrate that this learning problem is fixed.
The [measured diagnosis and reproduction scripts](/data/kanth042/datasets/umi_reset_from_defaults_20260911/52_training/diagnosis_20260917/README.md)
retain the exact checkpoint and source identities.

Geometry/physics/integration checks do not establish learned stacking. The preparation
helper requires a fresh PPO smoke record matching the corrected banks before a full
launch; the earlier run's smoke report is not evidence for the new mounting.

## Historical notes through the earlier launch preparation


As of September 17, the lower (receptive) cube again inherits UWLab's
**kinematic** setting: resets can place it, but gravity and contact cannot move
it during an episode. This applies to training, evaluation, and all four
full-arm reset generators. Partial-assembly discovery already fixed the lower
cube. This decision removes only our override that made it dynamic; existing
upper-cube motion mode, controller and rewards were unchanged by that decision.
The subsequent approved mass and sampling updates are described below. See the [change and validation record](/data/kanth042/datasets/umi_reset_from_defaults_20260911/46_fixed_receptive_cube/README.md).

As of September 14, the source restores May's original alignment reward and
success definition. It inherits the original `ProgressContext`, `TaskCommand`,
reward terms, weights, and metadata-based alignment thresholds. The added hand
release, withdrawal, table-support, and settling requirements and release metrics
have been removed. On September 17, the user requested reverting the separate
per-environment counter-reset change as well. The complete reward module now
matches upstream, including its original counter-reset behavior. The dependent
subset-isolation validation and training-preparation requirement were removed.

As of September 17, standalone UMI grasp generation uses the cube asset's
authored 100 g mass and matching inertia, as approved after the controlled
[stock-cube comparison](/data/kanth042/datasets/umi_reset_from_defaults_20260911/44_stock_cube_grasp_comparison/README.md).
The original UWLab cube actually simulated at 64 g in that test: its configured
1 g override was ineffective because its USD lacked the mass schema. With the
same UMI hand, it accepted 691/4,096 grasps at 64 g and 0/6,144 when explicitly
made 1 g. The September 14 change to our cube's generation mass therefore did
not reproduce the stock cube's runtime behavior. Hand inertials remain unchanged.

The later September 17 decision uses the selected assets' authored **100 g mass and inertia in every offline generation stage**, including partial assemblies and all four full-arm reset families. Generation disables cube mass randomization. Training and evaluation retain their existing 20–200 g range, with only the upper cube dynamic.

The completed map uses 10 mm XYZ spacing, heights 0–600 mm, 504 orientations and 50 mm robot-to-column clearance. Cube placements use position-only map coverage and 50 mm clearance from their rotated bounding footprint to the table edges. Cube Z/orientation samples are retained while finding XY locations; hand-angle samples are independent of position coverage. The added grasp-file and nearest-hand-angle vetoes were removed after reviewing the original recipe intent. Initial IK pose-error rejection and the extra airborne 10 mm support rule are removed; final simulated arm collision/column-clearance and cube-edge checks remain. Pose errors are diagnostic only. See the [exact change record](umi_map_reset_changes.md).

The 591 fresh 100 g grasps are reused. A new partial-assembly run recorded 5,536 records (4,521 distinct relative poses); all values are finite. The unmodified upstream recorder's within-batch repeats are retained. New map/reset outputs are in `/data/kanth042/datasets/umi_reset_from_defaults_20260911/49_clearance_and_placement`. The map and replacement full-arm bank are complete. All 40,732 retained states passed native geometry and structural checks, and all ten original physics/controller fixtures passed. The separate task-1 measurement lifted 0/256 cubes although all 256 hands rose; 255 starts had an almost fully closed gripper. Retained grip is not an upstream requirement or an added readiness criterion. The bank is staged while training remains paused. See the [completed generation and validation record](/data/kanth042/datasets/umi_reset_from_defaults_20260911/49_clearance_and_placement/README.md).

The policy commands Cartesian hand changes and the gripper rather than joint angles. Different initial arm configurations can realize the same hand pose. Their causal effect on the earlier learning outcome has not been isolated; the map's current configuration selection is retained.

The September 12 training process used its launch-time code with
the former release objective. It was stopped at the user's request on September 15
and has not been restarted. A fresh ten-iteration readiness test has now passed
with the restored baseline and final bank, using four GPUs and 16,384 environments
per rank. All logged metrics and checkpoint tensors were finite. The historical
launch below remains separate from the [new readiness and launch package](/data/kanth042/datasets/umi_reset_from_defaults_20260911/52_training/README.md).

The replacement bank contains **40,732** states: 10,644 general, 10,010 resting-grasp, 10,049 airborne-grasp and 10,029 partial-assembly states. Generation uses this new map and its input datasets. Training/evaluation now default to `49_clearance_and_placement/OmniReset`. The old bank is archived. The readiness smoke test completed ten iterations from fresh weights with local logging; the full training run has not started.

The native controller fixture lifted and released 52/64 airborne samples. The separate 256-environment grasp measurement lifted 232 airborne cubes, with 225 moving with the hand and dropping after opening, versus 0 resting-grasp cubes while all 256 hands rose. Those are controller measurements under training randomization, not policy success rates. May also had mostly closed table starts: its scripted lift raised 10/256 cubes, yet its learned policy completed 249/256 task-1 episodes. The previous UMI bank's scripted lift raised 2/256 cubes. These measurements do not isolate a cause of stalled training and do not create a retained-grasp requirement. Geometry validity does not establish retained grasps.

The launch helper `scripts_v2/tools/map_reset/prepare_training.py` now checks the final September 17 bank, restored-baseline physics tests, and fresh four-GPU smoke result. Its default operation prepares a source/data/validation snapshot and launch script under `52_training`; `--launch` is a separate explicit operation. The prepared full run retains four GPUs (3, 4, 5, 7), 16,384 environments per rank, 32 rollout steps, 40,000 iterations and original PPO/network parameters. It starts from fresh weights, not the smoke checkpoint.

The explicit Cartesian controller uses translation Kp/Kd **500/160** and
rotation Kp/Kd **60/0.1**. The original torque formula is retained:
`tau = J.T @ (Kp * pose_error - Kd * hand_velocity)`.
Physics runs at 120 Hz, with decimation 12 and 10 Hz policy actions.

## Historical September 12 bank and launch

[The exact reset-pipeline changes](umi_map_reset_changes.md) describe the atlas
seed and accepted sphere collision model. Pose-error rejection has since been removed. The
recorded generation run used the then-configured cube/grasp proposal distributions,
task bounds, masses, contacts, gripper settings, and stability criteria. The map and sphere
checks are used for bank generation and validation; training loads saved states
through the existing reset manager.

Every retained state passed structural and native reload collision checks. Two
resting rows were excluded by the same collision criterion after reload, with
the original bank and both row indices preserved. All nine physics fixtures
passed on the final files. The scripted loaded probe lifted and released 61 of
64 sampled cubes. A separate one-second hold probe retained support clearance
for 245 of 256 airborne samples under fresh task masses. These are controller
and data checks, not trained-policy success rates.

The fresh PPO launch uses physical GPUs **3, 4, 5, 7**, **16,384 environments per
rank** (65,536 total), 32 rollout steps, 40,000 iterations, and global seed 42.
Each rank uses seed 42 plus its rank. The original PPO and policy-network
settings are retained, and `resume=False` starts from fresh weights. The
completed-bank PPO smoke test must pass before launch.

The launch package is recorded in
[27_training/prepared_run.json](/data/kanth042/datasets/umi_reset_from_defaults_20260911/27_training/prepared_run.json).
It points to the exact command, source archive, input and validation hashes,
local log, W&B run URL, and live-run verification report. The preparation and
verification scripts are `scripts_v2/tools/map_reset/prepare_training.py` and
`scripts_v2/tools/map_reset/verify_training.py`.

[The regeneration report](/data/kanth042/datasets/umi_reset_from_defaults_20260911/26_map_reset_regeneration/README.md)
contains data provenance and a native sample gallery. The selected gains come
from [the handling experiment](/data/kanth042/datasets/umi_reset_from_defaults_20260911/25_lift_move_release/README.md).

The rejected implicit-controller run `pps7vks3` remains stopped and preserved.
Its checkpoint is not used by this launch. The earlier guide and diagnostic
history are preserved in
[training_guide_before_regeneration.md](/data/kanth042/datasets/umi_reset_from_defaults_20260911/26_map_reset_regeneration/training_guide_before_regeneration.md).

## September 17: camera contact reward candidate — deferred

During task-3 WebRTC playback of checkpoint 7,500, the user observed the newer
policy adjusting the upper cube with the wrist camera mount. In the subsequently
viewed May checkpoint 19,600 episodes, the user did not observe this behavior.
These observations do not measure how frequently either policy uses the mount.

The OmniReset paper, Appendix A.3.1, describes enlarged invisible geometry around
the wrist camera as a deployment safety buffer. The stock D415 USD contains an
enabled `d415_and_cable` collider. The newer D405 R2 asset has a mount/camera hull
without an equivalent separate enlarged camera/cable buffer. Enlargement affects
physics and reset acceptance; it does not automatically penalize contact, and a
policy could still exploit the virtual buffer to push objects.

Consider a small reward penalty for unintended camera/mount contact with task
objects. This is a candidate for later review, not an implemented or approved
reward change. Contact detection must distinguish the camera/mount from intended
finger-pad contact; coefficient and force threshold are not selected. Inspect
contacts already present in reset states and measure both contact frequency and
task success before judging a penalty's effect. The buffer difference has not
been isolated as the cause of the observed policy behavior.

Evidence: [paper](https://arxiv.org/html/2603.15789v3),
[asset inspection](/data/kanth042/datasets/umi_reset_from_defaults_20260911/43_may_task3_webrtc/d415_safety_buffer_verification.json),
and [checkpoint-7,500 contact audit](/data/kanth042/datasets/umi_reset_from_defaults_20260911/42_task2_task3_webrtc/policy/mount_contact_audit.json).

## September 17: hovering reward audit — no reward change

The saved checkpoint-6100 task-0/task-1 episodes earned positive proximity/alignment reward during unsuccessful hovering. Over their final five seconds, median hand displacement was about 0.1 mm and median cube-goal progress was zero. The measured net reward rates in never-successful episodes were +0.11875/s and +0.15595/s. These are observations from saved rollouts; they do not identify a unique cause of the training plateau. The original state-based rewards and motion penalties are retained. Full equations, reconstruction checks and limitations are in the [reward audit](/data/kanth042/datasets/umi_reset_from_defaults_20260911/50_stall_reward_audit/README.md).
