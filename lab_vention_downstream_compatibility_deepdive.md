# Lab Vention Migration: Downstream Compatibility Deep Dive

This note explains what changed when we introduced the lab Vention setup, what
was intentionally kept compatible with UWLab, and what still needs to be done
before claiming we can reproduce the UWLab OmniReset pipeline on the lab setup.

The core idea is:

```text
Keep downstream software interfaces stable.
Regenerate geometry-dependent data.
Do not pretend the old reset datasets/camera data are still physically valid.
```

## Compatibility Goal

For downstream code to be "indifferent" to the new Vention setup, the code
should still see the same scene entity names, observation names, action shape,
recorder schema, reset-state file schema, and task concepts.

That does not mean the physical distribution is identical. The lab Vention frame
changes where the robot is, where the tabletop is, what the cameras see, and
which object placements are reachable. Those values are not supposed to be
hidden from the policy/data. They should be reflected through regenerated reset
states, camera alignment, and eventually lab-specific training/data-collection
env variants.

## What Stayed The Same

### Axis And Unit Convention

The lab setup still uses the same Isaac/USD convention as UWLab:

```text
scene units: meters
up axis: +Z
horizontal workspace plane: X/Y
quaternion order in configs: w, x, y, z
per-env placement: env origin + configured local/root pose
```

Here "horizontal workspace plane: X/Y" means:

```text
X = horizontal axis along the long tabletop dimension after LAB_VENTION_* is applied
Y = horizontal axis along the short tabletop dimension after LAB_VENTION_* is applied
Z = vertical axis, positive upward, normal to the tabletop
```

For the current lab right-arm scene, the transformed tabletop bounds are:

```text
tabletop X extent: -0.834840 to 0.650160 m  # long direction, about 1.485 m
tabletop Y extent: -0.432305 to 0.287695 m  # short direction, about 0.720 m
tabletop top Z:     0.842350 m
```

The first conservative object reset rectangle is a subset of that tabletop:

```text
sampled object X: -0.100 to 0.625 m
sampled object Y: -0.385 to 0.265 m
sampled surface Z: 0.842350 m
```

The selected simulated right arm root is:

```text
robot root position: (0.177660, 0.377695, 1.466000)
robot root rotation: (0.7071068, 0.0, 0.7071068, 0.0)  # w, x, y, z
```

Because the tabletop positive-Y edge is at about `Y = 0.287695 m`, the robot
root is on the positive-Y side of the tabletop, slightly past that edge. The
object reset rectangle stays on the tabletop itself.

### Origin And Per-Env Placement

The robot base is not the world origin.

The Vention CAD origin is not the world origin.

The tabletop center is not the world origin.

The Isaac world origin is the simulation coordinate frame origin:

```text
world origin = (0, 0, 0)
```

In the single-env viewer, `env_0` is effectively placed at that world origin.
In multi-env training/recording, Isaac gives each environment its own
`env.scene.env_origins[i]` so many copies can be tiled in one simulation.

So the actual world placement rule is:

```text
world pose in env_i = env.scene.env_origins[i] + configured local/root pose
```

For example:

```text
table root in env_i = env_i origin + LAB_VENTION_POS
robot root in env_i = env_i origin + LAB_RIGHT_ARM_ROBOT_POS
object reset in env_i = env_i origin + sampled lab-tabletop X/Y/Z
```

The CAD-to-scene rotation is just the transform needed to place the imported
Vention CAD into the same Isaac convention. It does not create a new coordinate
system for the task code.

### Scene Entity Names

We intentionally kept the important Isaac scene entity names:

```text
robot
table
ur5_metal_support
insertive_object
receptive_object
front_camera
side_camera
wrist_camera
```

This is the biggest compatibility choice. UWLab events, observations,
recorders, rewards, and randomizers refer to these names through
`SceneEntityCfg`. Keeping the names means downstream code can continue to look
up `scene["table"]`, `scene["robot"]`, `scene["insertive_object"]`, etc.

### Robot And Object Asset Semantics

We did not change the simulated UR5e/Robotiq asset globally. We also did not
change the peg and peg-hole object assets.

The action space remains the same for the reset-state task:

```text
6 arm OSC dimensions + 1 gripper dimension
```

The reset-state recorder still writes the same type of `.pt` file:

```text
initial_state/articulation/robot/...
initial_state/rigid_object/insertive_object/...
initial_state/rigid_object/receptive_object/...
initial_state/rigid_object/table/...
initial_state/rigid_object/ur5_metal_support/...
```

This was validated by a smoke recorder run that exported:

```text
/data/kanth042/converted_assets/lab_vention_asset_v60/reset_state_smoke_dataset/Resets/Peg__PegHole/resets_ObjectAnywhereEEAnywhere.pt
```

### Original UWLab Environments

The original UWLab env IDs were not modified. We added lab-specific env IDs
instead.

Existing original envs still point at the original flat UWLab setup unless a
lab-specific env ID/config is chosen.

## What Changed

### 1. Lab Vention Asset Pipeline

We started from the lab Vention STEP export:

```text
/data/kanth042/downloads/VentionAssembly_511882_v60.STEP
```

The reproducible conversion path is:

```text
STEP
  -> lab_vention_noinst_raw.usd
  -> lab_vention.usd
```

The processed asset is:

```text
/data/kanth042/converted_assets/lab_vention_asset_v60/lab_vention.usd
```

The processed asset follows the relevant UWLab support-asset pattern:

```text
/lab_vention
  /visuals
  /collisions
```

Important choices:

- CAD UR arms were removed from the Vention asset because the simulated robot
  articulation is spawned separately.
- The tabletop visual mesh was promoted to a stable path:

  ```text
  /lab_vention/visuals/vention_mat
  ```

- Collision is represented by simple invisible Cube proxies with
  `PhysicsCollisionAPI`, matching the relevant UWLab support-asset style.
- We deliberately did not use raw CAD mesh collisions as the main collision
  representation.

Why this matters downstream:

- Code that randomizes the table visual can still target `visuals/vention_mat`.
- The scene still has one entity called `table`.
- Physics sees a simple support asset rather than a fragile CAD assembly.

### 2. Shared Lab Layout Constants

We introduced:

```text
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/lab_layout_cfg.py
```

This file centralizes the lab-specific geometry:

```text
LAB_VENTION_USD_PATH
LAB_VENTION_POS
LAB_VENTION_ROT
LAB_VENTION_SCALE

LAB_RIGHT_ARM_ROBOT_POS
LAB_RIGHT_ARM_ROBOT_ROT

LAB_TABLETOP_TOP_Z
LAB_TABLETOP_PANEL_REL_PATH

LAB_RIGHT_ARM_REACHABLE_RECT_X
LAB_RIGHT_ARM_REACHABLE_RECT_Y
LAB_RIGHT_ARM_RESET_X_RANGE
LAB_RIGHT_ARM_RESET_Y_RANGE
```

This prevents the viewer, reset-state env, camera-align env, and future
data-collection envs from each carrying their own private copy of the lab
numbers.

### 3. Lab Reset-State Scene Variant

We added a lab-specific reset-state scene:

```text
LabRightArmResetStatesSceneCfg
```

It changes the physical layout:

```text
robot -> selected lab right-arm station
table -> processed lab Vention USD
ground -> z = 0
ur5_metal_support -> hidden compatibility entity
```

The old `ur5_metal_support` visual plate is not drawn in the lab scene. The lab
Vention CAD already contains the physical support geometry. However, the entity
name still exists as a tiny hidden kinematic cuboid so older code paths that
expect the entity name do not immediately break.

### 4. Object Reset Reference Changed

This is the most important behavioral change.

Old UWLab `ObjectAnywhereEEAnywhere` reset:

```text
object pose = env origin + ur5_metal_support default pose + sampled offset
```

That was implemented through:

```text
offset_asset_cfg = SceneEntityCfg("ur5_metal_support")
```

Lab right-arm reset:

```text
object pose = env origin + sampled lab-tabletop X/Y/Z pose
```

The lab object reset terms no longer pass `offset_asset_cfg`. Their X/Y ranges
are lab tabletop coordinates in the local env frame, and their Z range is based
on:

```text
LAB_TABLETOP_TOP_Z = 0.84235
```

We still keep:

```text
use_bottom_offset = True
```

That means the configured Z is a placement surface height, while the object root
pose is shifted according to the object's metadata so the visible/collision
bottom sits on that surface.

### 5. Lab Reset Workspace

The current conservative lab reset rectangle is:

```text
X: -0.100 to 0.625   # long tabletop direction
Y: -0.385 to 0.265   # short tabletop direction
Z tabletop top: 0.84235
```

These values came from a tabletop reachability probe over:

```text
/lab_vention/visuals/vention_mat
```

The larger reachability result suggested a bigger top-down-gripper reachable
rectangle:

```text
x: -0.1171 to 0.6502
y: -0.4083 to 0.2877
```

We shrank it slightly for the first reset-state variant.

### 6. Lab Reset Success Threshold

The original reset-state success check only required objects to remain above a
near-ground threshold.

That is not enough for the lab frame. An object can fall off the tabletop and
still be above the ground for part of the episode, or land on the ground while
the receptive object stayed on the table.

The lab-only reset-state config therefore sets:

```text
pos_z_threshold = LAB_TABLETOP_TOP_Z - 0.03
```

This affects only:

```text
LabRightArmObjectAnywhereEEAnywhereResetStatesCfg
```

It does not modify the original UWLab reset-state config.

### 7. New Env Registrations

We added lab env IDs instead of changing the original ones:

```text
OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-LabRightArm-v0
OmniReset-Ur5eRobotiq2f85-CameraAlign-LabRightArm-v0
```

This keeps the old paper configs available for comparison while giving us a lab
path that can be evolved safely.

### 8. Lab Camera-Align Scene Variant

We added:

```text
LabRightArmCameraAlignSceneCfg
LabRightArmCameraAlignEnvCfg
```

This scene places the robot and lab Vention frame in the lab right-arm layout,
but the camera model/poses are still inherited from the existing D415-era
camera-align setup.

This is useful for visual inspection and transitional alignment, but it is not
the final D405 migration.

## Why Downstream Code Is Mostly Indifferent

### Entity Lookup Is Stable

Most downstream code does not care whether `scene.table` is `pat_vention.usd`
or the lab Vention asset. It cares that `scene["table"]` exists and is a rigid
object.

We preserved that.

The same is true for:

```text
scene["robot"]
scene["insertive_object"]
scene["receptive_object"]
scene["ur5_metal_support"]
```

### Recorder Schema Is Stable

`record_reset_states.py` worked unchanged with the lab task:

```text
successful reset states exported: 20
```

The output file has the same high-level structure as UWLab reset-state files.
That is essential because `MultiResetManager` expects that schema.

### Observation Shapes Are Stable

The state-policy observations are largely relative:

```text
end_effector_pose in robot frame
insertive object pose in wrist_3_link frame
receptive object pose in wrist_3_link frame
insertive object pose in receptive object frame
joint positions
previous actions
```

These shapes do not change just because the robot root moved in the world.

That is the right design for sim2real: absolute world placement should matter
much less than robot-relative and object-relative state.

### Action Interface Is Stable

The OSC action still controls the same robot bodies/joints. We did not change
the action term shape or names.

### Table Visual Randomization Can Stay Compatible

UWLab RGB randomization targets:

```text
mesh_names = ["visuals/vention_mat"]
```

The processed lab Vention asset intentionally exposes:

```text
/lab_vention/visuals/vention_mat
```

So a future lab RGB/data-collection scene can keep the same table mesh target
instead of rewriting the randomizer logic.

## What Downstream Is Not Yet Indifferent To

### Old Reset-State Datasets Are Not Physically Valid For Lab

Old reset-state `.pt` files were generated for the old flat Vention/UR5-plate
layout.

Even if the schema loads, the physical meaning is wrong for the lab frame. The
robot root, table height, object workspace, and camera views do not match.

For paper replication on the lab Vention setup, we should regenerate lab reset
states rather than use the cloud reset states directly.

### Only One Reset Type Has A Lab Variant So Far

Currently the lab-specific reset-state variant covers:

```text
ObjectAnywhereEEAnywhere
```

The UWLab full pipeline uses multiple reset types:

```text
ObjectAnywhereEEAnywhere
ObjectRestingEEGrasped
ObjectAnywhereEEGrasped
ObjectPartiallyAssembledEEGrasped
```

To replicate the full paper pipeline, all geometry-dependent reset-state
generation paths need lab variants or lab-compatible datasets.

### RL State Train/Eval Configs Still Point At Old Layout

The main RL state configs still inherit `RlStateSceneCfg`, which uses:

```text
Props/Mounts/UWPatVention/pat_vention.usd
Props/Mounts/UWPatVention2/Ur5MetalSupport/ur5plate.usd
```

They also use `MultiResetManager` with:

```text
dataset_dir = UWLAB_CLOUD_ASSETS_DIR/Datasets/OmniReset
```

So the training/eval envs have not yet been migrated to the lab Vention layout.

That is good for isolation, but it means the lab reset-state work is not yet
feeding the full RL train/eval path.

### RGB Data Collection Still Uses Old Layout And D415 Assumptions

The RGB data-collection scene still inherits the original scene and camera
poses.

It still uses the existing D415-era wrist camera pose/model and old external
camera positions.

The lab camera-align scene exists, but we have not yet created the lab RGB
data-collection/eval envs that:

- spawn the lab Vention scene;
- use lab reset datasets;
- use final D405 wrist camera intrinsics/extrinsics;
- place front/side cameras according to the lab setup;
- keep image observation names/shapes compatible.

### Absolute Visual Distribution Changes

Even if state observations are relative, RGB policies see the world. The lab
frame, tabletop, arm mount, camera perspective, and D405 mount will change the
images.

For RGB replication, downstream is not indifferent until we regenerate data and
train/evaluate under the lab visual distribution.

## Current Validation

### Lab Reset Validation

We added:

```text
scripts_v2/tools/validate_lab_right_arm_resets.py
```

Large validation run:

```text
samples: 800
support failures: 33 / 800 = 4.125%
receptive drift events: 276 / 800 = 34.5%
min receptive end z: 0.857513
min insertive end z: 0.015000
```

Interpretation:

- The receptive object stayed on the tabletop in that sample.
- Some insertive outcomes fell below the tabletop during settling.
- The lab-specific `pos_z_threshold` should reject those off-table outcomes
  during reset-state recording.

### Recorder Smoke Test

The actual UWLab recorder successfully exported lab reset states:

```text
successful reset states exported: 20
insertive z range: 0.857350 to 0.872350
receptive z range: 0.857513 to 0.857513
insertive xy range:
  x: -0.094318 to 0.598245
  y: -0.381519 to 0.256029
receptive xy range:
  x: -0.065055 to 0.592777
  y: -0.371538 to 0.251117
unique robot root pose count: 1
unique table root pose count: 1
```

This proves the lab reset variant is wired into the existing recorder flow.

## What This Means For Reproducing The UWLab Paper

The migration so far is a foundation, not the full paper replication yet.

To reproduce the UWLab paper pipeline on the lab Vention setup, the safe path is:

1. Keep original UWLab env IDs untouched as baselines.
2. Add lab-specific env IDs/configs for each stage.
3. Regenerate lab reset-state datasets for every reset type used by the paper.
4. Point lab train/eval/RGB configs at the lab datasets.
5. Keep observation/action schemas identical unless there is a deliberate paper
   deviation.
6. Validate that observation keys/shapes match the paper configs.
7. Only then train/evaluate with the same pipeline/hyperparameters.

## Minimal Next Config Work

The next meaningful changes should be:

### A. Lab RL State Config

Create lab train/eval state configs that:

- inherit the same observation/action/reward/termination structure;
- use a lab scene equivalent to `RlStateSceneCfg`;
- spawn the processed lab Vention asset as `table`;
- keep `ur5_metal_support` as a hidden compatibility entity;
- load reset states from a lab dataset directory.

The goal is not to change the policy interface. The goal is to change only the
scene geometry and reset dataset source.

### B. Lab Reset Datasets For All Reset Types

The paper pipeline uses more than `ObjectAnywhereEEAnywhere`.

We should generate lab versions for:

```text
ObjectAnywhereEEAnywhere
ObjectRestingEEGrasped
ObjectAnywhereEEGrasped
ObjectPartiallyAssembledEEGrasped
```

The first one is proven in smoke form. The others need lab-aware handling.

### C. Lab RGB Config

Create lab RGB data-collection/eval configs that preserve:

```text
front_rgb
side_rgb
wrist_rgb
image sizes
policy/data_collection observation groups
action interface
termination logic
```

but replace:

```text
scene geometry
camera extrinsics/intrinsics
reset dataset directory
wrist camera mount/camera from D415 to D405
```

### D. D405 Migration

The D405 work should be treated as a separate sensor/mount migration layered on
top of the lab scene migration.

The lab Vention migration changes the world/layout. The D405 migration changes
the wrist sensor geometry and camera model.

## Current Bottom Line

Downstream code is not automatically indifferent to the lab setup in the sense
of "old data and old cameras still work." That would be physically wrong.

But the migration has been structured so downstream code can become indifferent
at the software-interface level:

```text
same entity names
same action shape
same observation names/shapes
same reset-state file schema
same recorder and MultiResetManager path
same table visual mesh target
separate lab env IDs to avoid breaking original UWLab baselines
```

The remaining work is to create the lab versions of the downstream train/eval
and RGB configs, and to regenerate the geometry-dependent datasets under the lab
layout.
