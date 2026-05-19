# Lab Scene Migration From First Principles

## Purpose

This document resets the reasoning for migrating the current UWLab OmniReset
scene to the lab setup.

The current repo scene is a horizontal/tabletop Vention setup. The lab setup is
a UR5e mounted on a vertical Vention frame, with a separate physical workspace
surface where task objects sit.

The goal is not to rename everything. The goal is to understand the existing
names, preserve them where possible, and change the geometry in a principled
way.

## Existing Code Names And What They Mean

Keep these names in code for now:

```text
robot
ur5_metal_support
table
insertive_object
receptive_object
```

Their physical meaning in this scene is:

```text
robot
  The UR5e + Robotiq articulation at {ENV_REGEX_NS}/Robot.
  This is the actual arm.

ur5_metal_support
  The UR5 mounting plate asset at {ENV_REGEX_NS}/UR5MetalSupport.
  This is the plate/support immediately under the robot base.

table
  Misleading name in this repo.
  It loads Props/Mounts/UWPatVention/pat_vention.usd.
  Treat it as the Vention/support/frame asset, not the workspace tabletop.

receptive_object
  The task fixture or receiving object.
  In some variants this can be a tabletop asset, drawer box, peg hole, wall, etc.

insertive_object
  The object being manipulated or inserted.
```

Current key files:

```text
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/rl_state_cfg.py
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/reset_states_cfg.py
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/camera_align_cfg.py
```

## Current Scene, In Plain Terms

The current base robot pose is identity:

```python
robot.init_state.pos = (0, 0, 0)
robot.init_state.rot = (1, 0, 0, 0)
```

So the current scene effectively uses the robot base as the old task anchor.

The UR5 plate is just below the robot:

```python
ur5_metal_support.init_state.pos = (0.0, 0.0, -0.013)
ur5_metal_support.init_state.rot = (1.0, 0.0, 0.0, 0.0)
```

The `table` asset is offset from that origin:

```python
table.init_state.pos = (0.4, 0.0, -0.881)
table.init_state.rot = (0.707, 0.0, 0.0, -0.707)
```

This means the current scene origin is not a measured physical point on the
Vention frame or the real workspace table. It is an old task convention: place
the robot near the origin, then arrange support/object assets around it.

## The Correct Physical Model

For the lab setup, separate the robot mounting stack from the task workspace.

### Mounting Stack

This is the hardware that holds the robot:

```text
scene world
  -> table                  # existing code name, physically the Vention frame/support
      -> ur5_metal_support  # UR5 mounting plate
          -> robot          # UR5e base/articulation
              -> robotiq_base_link
                  -> rgb_wrist_camera
```

### Workspace Stack

This is where the task objects sit:

```text
scene world
  -> workspace surface / physical tabletop
      -> receptive_object
      -> insertive_object
```

The current code partially mixes these two stacks because object resets are
often offset from `ur5_metal_support`. That was convenient in the old
horizontal setup, but it is not a good assumption once the robot is mounted on
a vertical frame and the object workspace is a separate table/surface.

## The Main Design Rule

Do not ask "where should the robot go?" in isolation.

Ask for the whole transform graph:

```text
scene world -> table
table -> ur5_metal_support
ur5_metal_support -> robot
scene world -> workspace surface
workspace surface -> task objects
robot/robotiq -> wrist camera
```

If these transforms are correct, the scene makes sense.

If only `robot` is moved, the robot may float or appear bolted to nothing.

If only `table` is moved, the Vention visual may look correct but the robot and
objects may still be wrong.

If objects remain offset from `ur5_metal_support`, they may no longer land on
the real workspace table.

## What The Scene World Should Mean

The current scene world does not have deep physical meaning. For the lab
migration we should choose a convention and document it.

Recommended convention for the first lab migration:

```text
Use the Vention frame/support as the primary measured anchor.
Keep the code name `table` for that asset.
Place the workspace surface relative to the same scene world.
```

This does not require renaming the code field `table`. It only means:

```text
table = Vention frame/support asset
```

If the Vention USD asset origin is useful, we can set `table` close to identity
and measure everything relative to it. If the asset origin is awkward, we can
still place `table` at a chosen world pose, but we must record that pose
clearly.

## Values We Need To Know

These are the real unknowns.

### Vention Frame Pose

```text
scene world -> table
```

This places the existing `table` asset, which is physically the Vention
support/frame.

For a clean first pass, this can be chosen for convenience. For example:

```text
table.pos = (0, 0, 0)
table.rot = upright orientation
```

The exact upright orientation depends on how `pat_vention.usd` was authored.

### UR5 Plate Pose On Vention

```text
table -> ur5_metal_support
```

This is where the UR5 mounting plate sits on the Vention frame.

This should come from:

```text
Vention CAD
hole/slot counts on the physical frame
manual measurements from a known frame corner/hole
```

It should not be guessed independently for every scene.

### Robot Base Pose On Plate

```text
ur5_metal_support -> robot
```

This is mostly mechanical. Once the UR5 plate and UR base are modeled
correctly, this transform should be fixed.

### Workspace Surface Pose

```text
scene world -> workspace surface
```

This is the physical table/mat/surface where objects sit.

This is separate from the existing code name `table`, because `table` currently
loads the Vention/support asset.

The workspace surface height matters because it determines:

```text
object resting height
camera appearance
reachable end-effector poses
contact/insertion geometry
```

### Object Poses On Workspace

```text
workspace surface -> receptive_object
workspace surface -> insertive_object
```

Object reset ranges should ultimately be relative to the workspace surface, not
to the robot mounting plate, unless the workspace surface really is attached to
that plate.

### Wrist Camera Pose

```text
robotiq_base_link -> rgb_wrist_camera
```

This is the D405 migration. It depends on the robot/gripper frame, but it is a
separate transform from the Vention and workspace table geometry.

## What The Current Reset Code Does

Several reset terms use:

```python
"offset_asset_cfg": SceneEntityCfg("ur5_metal_support")
"use_bottom_offset": True
```

The reset implementation effectively places objects by taking:

```text
object default pose
+ sampled xyz/rpy
+ ur5_metal_support default position
- object bottom offset
```

So today, object ranges are implicitly tied to `ur5_metal_support`.

That means `z = 0` means "object bottom on the surface implied by the
ur5_metal_support offset", not "object bottom on a separately modeled lab
workspace table."

This is the key thing that must change or be reinterpreted for the vertical
Vention lab setup.

## What The Paper Says About Height

The paper randomizes object pose, including object `z`, but the code and paper
do not show a broad randomization of the workspace table height itself.

The paper's Drawer Assembly discussion is about local relative geometry:

```text
drawer lip causes a height mismatch
layered mats with a cutout align the drawer bottom with the workspace surface
```

The lesson is:

```text
Match the relative height geometry at the contact/insertion site.
Do not assume table/workspace height can be arbitrary.
```

For our lab setup, the workspace surface height should be measured and matched
in sim.

## Recommended Implementation Strategy

### Phase 1: Geometry Sandbox In Camera Alignment

Start in:

```text
camera_align_cfg.py
```

Do not touch training first.

Set a single fixed geometry:

```text
table              -> upright Vention frame/support
ur5_metal_support  -> fixed pose on table/Vention
robot              -> fixed pose on ur5_metal_support
workspace surface  -> measured physical task table/mat
objects            -> placed on workspace surface
```

Keep existing scene field names:

```python
self.scene.table.init_state...
self.scene.ur5_metal_support.init_state...
self.scene.robot.init_state...
```

But reason about them using the physical hierarchy:

```text
table -> ur5_metal_support -> robot
```

### Phase 2: Measure The Lab

Measure or derive:

```text
table -> ur5_metal_support
ur5_metal_support -> robot
scene world/table -> workspace surface
workspace surface -> common object spawn region
```

Use CAD where possible. Use tape/hole counts for a first pass. Then validate
with camera images and a known robot joint pose.

### Phase 3: Fix Object Anchoring

Do not blindly keep:

```python
offset_asset_cfg = SceneEntityCfg("ur5_metal_support")
```

for all object resets.

For the lab vertical-frame setup, object placement should be tied to the
workspace surface. Options:

1. Use world-frame object ranges with no offset asset.
2. Add a small explicit workspace anchor asset if needed.
3. Use a task fixture/receptive object as the local anchor for task-specific
   resets.

The safest initial evaluation path is world-frame object ranges measured from
the chosen scene world.

### Phase 4: Update Main RL/Eval Configs

After the camera-align scene is visually correct, copy the measured geometry to:

```text
rl_state_cfg.py
reset_states_cfg.py
data_collection_rgb_cfg.py
```

Then regenerate reset states if the old reset-state files assume the horizontal
tabletop layout.

### Phase 5: D405 Wrist Camera

Once the robot and workspace geometry are sane, apply the D405 wrist camera
calibration:

```text
robotiq_base_link -> rgb_wrist_camera
D405 intrinsics
```

Do this after the mount/workspace geometry is stable so we can interpret camera
images correctly.

## What Not To Do

Do not rename existing code fields just to make them semantically nicer.

Do not treat the existing `table` code name as the physical workspace table.

Do not leave objects offset from `ur5_metal_support` if the workspace table is
separate from the robot mounting plate.

Do not tune `robot`, `table`, and `ur5_metal_support` as unrelated world poses
if we can express them as a physical transform chain.

Do not assume the current scene origin is a Vention or lab datum.

Do not assume the policy is robust to a changed workspace height unless that
variation was represented in training or fine-tuning.

## Minimal Data Sheet To Fill In

Use meters and quaternion order `(w, x, y, z)`.

```text
table.pos = (?, ?, ?)
table.rot = (?, ?, ?, ?)

ur5_metal_support relative to table:
  pos = (?, ?, ?)
  rot = (?, ?, ?, ?)

robot relative to ur5_metal_support:
  pos = (?, ?, ?)
  rot = (?, ?, ?, ?)

workspace surface relative to scene world or table:
  pos = (?, ?, ?)
  rot = (?, ?, ?, ?)

object spawn region relative to workspace surface:
  x = (?, ?)
  y = (?, ?)
  z = (?, ?)

D405 wrist camera relative to robotiq_base_link:
  pos = (?, ?, ?)
  rot = (?, ?, ?, ?)
  intrinsics = ?
```

## Validation Checklist

Geometry is ready when:

- The existing `table` asset appears as the upright Vention/support frame.
- `ur5_metal_support` sits on the Vention frame where the real UR5 plate is.
- `robot` appears bolted to `ur5_metal_support`.
- The workspace surface is at the measured real height and location.
- Objects spawn on the workspace surface, not on the robot mounting plate.
- A known real UR5e joint configuration qualitatively matches the sim pose.
- Front, side, and wrist camera views show the same task-relevant geometry as
  the real setup.
- The policy observation images show the object, workspace, and gripper in the
  expected relationship.

## Bottom Line

The migration is not just "rotate the Vention" and not just "move the robot
root."

The migration is:

```text
define the physical transform graph
keep existing code names where possible
place Vention, plate, robot, workspace surface, objects, and cameras according
to that graph
then validate the policy observations
```
