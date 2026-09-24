# Upstream robot, fixed-goal cube pilot

This configuration uses the released UWLab robot calibration, stock Robotiq hand,
D415, mount/table, 40 mm cubes, controller, rewards and PPO settings. The first
local code change corrects critic velocity vectors. The easy-task configuration
is a separate change. It is an explicit local adaptation of the paper's narrow
start/fixed-goal idea; upstream does not publish a named easy-cube preset or bank.
Thunder calibration, mounting, D405, 60 mm cubes and custom fingers are not part
of this reference experiment.

The lower-cube XY is fixed at `(0.45, 0.15)` m, with identity orientation. Height
uses the unchanged upstream support and cube bottom offsets. Free-hand and
carrying reset proposals use `x in [0.35, 0.37]`, `y in [-0.01, 0.01]` m.
Upstream height/orientation sampling, robot-root jitter and grasp diversity remain.
Near-goal poses come from the released object-relative partial assemblies at the
fixed goal. The lower cube remains kinematic, as in upstream training.

Use the published stock-cube grasp and partial-assembly files from asset revision
`f9486a69b30a38355c1c2415fc15f6d6be8cb5b3`. Generate new whole robot/object states
with `scripts_v2/tools/record_reset_states.py`; changing cube coordinates in an
old bank would not preserve a valid arm/grasp configuration.

Set `UWLAB_EASY_CUBE_DATASET_DIR` to the dataset root for both generation and
training. The generator's `--dataset_dir` must point to that same root. For each
family, use task `OmniReset-UR5eRobotiq2f85-CubeEasy-<family>-v0` and supply the
original family name to `--reset_type`. Generate these families:

1. `ObjectAnywhereEEAnywhere`
2. `ObjectRestingEEGrasped` (after the free-hand bank is finalized)
3. `ObjectAnywhereEEGrasped`
4. `ObjectPartiallyAssembledEEGrasped`

Settling can move a cube outside its proposal patch. The experiment's data audit
therefore selects complete free-hand/table-grasp rows whose saved cube XY remains
within the patch; it does not edit coordinates or joints. Carrying and near-goal
states retain their accepted elevated configurations, with measured ranges saved
in the manifest. All families must retain the fixed lower-cube pose. Reserve
separate held-out rows for subsequent evaluation. The four training families are
sampled equally, regardless of their retained row counts.

Train with task `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-CubeEasy-v0`.
The first pilot is one fresh seed-42 run, four GPUs with 16,384 environments each,
32 rollout steps and **1,000 PPO iterations**. This is 2,097,152 environment
transitions per iteration. The native runner numbers iterations 0–999; its final
checkpoint is `model_999.pt`. No subsequent training experiment is queued until
the pilot is reviewed. A flat early curve does not establish permanent failure.

Each run records the source commit/archive, resolved configuration, bank and
asset hashes, dependencies, seeds and exact launcher. Use separate worktrees or
frozen checkouts when developing another code change during training.

The September 24 instance and executable generation/audit/launch records are at
`/data/kanth042/datasets/umi_reset_from_defaults_20260911/104_upstream_easy_cube_20260924`.
The installed RSL-RL revision `959ccbc4712400a75efaa63cb827fd8939776825` has identical
runtime source to upstream's pin `e7cd3c77bdb3c94753612f208c725e1add38a655`; the latter
only adds review-tool metadata. The launcher records this equivalence and checks
the installed PPO/runner/policy/storage file hashes without modifying the shared
environment.
