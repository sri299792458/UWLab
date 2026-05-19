# Concept Notes

Explanations and study notes while reading the code. Implementation decisions and TODOs stay in `running_notes.md`.

## 2026-05-18 - What a USD File Means Here

USD means Universal Scene Description. In Isaac Sim / Isaac Lab, a `.usd` file is the main scene-asset format used to describe objects, robots, cameras, lights, materials, meshes, physics properties, and hierarchical transforms.

It is similar to URDF in the narrow sense that both can describe a robot as links connected by joints, with visual geometry, collision geometry, and inertial properties.

It is broader than URDF because USD can describe the whole simulated scene, not just a robot model. A USD can contain:

- A tree of prims, which are scene nodes like links, meshes, cameras, lights, and joints.
- Transforms between prims.
- Mesh geometry and materials.
- Visual and collision geometry.
- Physics schemas for rigid bodies, joints, articulations, mass, friction, and collision.
- Cameras and rendering properties.
- References to other USD files, so a large asset can be composed from smaller assets.
- Variants and layers, so assets can be overridden or specialized without copying everything.

URDF is mainly a robot-description XML format. It is good for saying:

- This robot has these links.
- These links are connected by these joints.
- Each link has visual, collision, and inertial data.

USD is a general scene graph format. In Isaac Sim, the simulator ultimately wants USD. A URDF is often imported or converted into USD, and then the USD becomes the asset Isaac actually spawns.

For the D415 to D405 migration, the important distinction is:

- The URDF-like part answers: what are the robot links and joints?
- The USD scene part answers: what exact mount mesh is attached to the gripper, what is its prim name, where is the camera prim, what visual/collision geometry is present, and how does Isaac spawn the asset?

So the D405 change is not just a URDF robot-model change. It is also a USD scene-asset change.

## 2026-05-18 - What Are Prims?

In USD, a prim is a primitive scene object. Practically, it is a node in the USD scene tree.

The word "primitive" does not mean "simple shape" here. A prim can represent many kinds of things:

- A transform frame.
- A robot link.
- A mesh.
- A camera.
- A light.
- A material.
- A physics body.
- A joint.
- A reference to another USD asset.

So when Isaac code says `prim_path`, it means "the path to a particular node in the USD scene graph."

Example:

```text
/World
  /envs
    /env_0
      /Robot
        /base_link
        /shoulder_link
        /forearm_link
        /wrist_3_link
        /robotiq_base_link
          /visuals
            /D415_to_Robotiq_Mount
          /rgb_wrist_camera
```

Each line in that tree is a prim. The full path to the wrist camera prim would be:

```text
/World/envs/env_0/Robot/robotiq_base_link/rgb_wrist_camera
```

That path is analogous to a filesystem path, but it points into the USD scene hierarchy instead of the disk.

### Why Prims Matter For This Repo

The wrist camera config uses a prim path:

```python
prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera"
```

This means Isaac expects to find or create a camera sensor at the `rgb_wrist_camera` prim under `robotiq_base_link`.

The mount randomization code also uses a prim-like mesh path:

```python
mesh_names=["robotiq_base_link/visuals/D415_to_Robotiq_Mount"]
```

This means the code expects the robot USD to contain a visual mesh prim with that exact name and relative path.

For the D405 migration, if the new USD uses:

```text
robotiq_base_link/visuals/D405_to_Robotiq_Mount
```

but the code still searches for:

```text
robotiq_base_link/visuals/D415_to_Robotiq_Mount
```

then the visual randomization code will not find the D405 mount mesh.

### Prim Versus Mesh

A prim is the scene node. A mesh is one possible kind of geometry stored on or referenced by a prim.

So:

- `D405_to_Robotiq_Mount` could be a prim.
- That prim may contain mesh geometry loaded from an `.stl`, `.obj`, or another USD file.
- The prim has a path in the scene.
- The mesh is the actual shape rendered or used for collision.

This distinction matters because code usually refers to the prim path, not directly to the original `.stl` file.

### Prim Versus Link

In URDF, we usually think in terms of links and joints.

In USD, a robot link is represented by one or more prims. For example, `robotiq_base_link` may be a prim that has child prims for visuals, collisions, sensors, and attached meshes.

So a URDF link name may appear as a USD prim name, but USD can contain more hierarchy under that link than URDF usually makes obvious.

### Important Mental Model

Think of USD as a scene tree. A prim is one addressable object in that tree.

For our D405 work, the critical prim questions are:

- Where is the gripper base link prim?
- What is the exact D405 mount visual prim path?
- Does the wrist camera prim still live at `robotiq_base_link/rgb_wrist_camera`?
- If we change any prim name, which config files still point to the old name?

## 2026-05-18 - What To Explore In The Isaac Sim USD Viewer

The viewer is currently loading the existing D415-mounted UR5e + Robotiq USD:

```text
https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated/ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd
```

The purpose of exploring it is not just to "see the robot." The goal is to understand what the USD actually contains, which prim names the code depends on, and what must change when replacing the D415/mount with the D405/mount.

### 1. First Understand The Stage Tree

Open the Stage panel and expand:

```text
/World
  /Robot
```

This is the live USD scene hierarchy. Every row is a prim.

You should inspect the tree like a robot model plus filesystem:

- `/World` is the root scene.
- `/World/Robot` is where our script spawned the robot articulation.
- Child prims under `/World/Robot` are links, joints, visuals, collisions, sensors, and attached objects.

The key skill is to connect a visible object in the viewport to its prim in the Stage tree.

### 2. Find The Gripper Base Link

Search or expand until you find:

```text
/World/Robot/robotiq_base_link
```

This is important because the wrist camera config attaches the simulated camera under this link:

```python
prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera"
```

In the viewer, select `robotiq_base_link` and look at:

- Its children in the Stage tree.
- Its local transform.
- Its visual children.
- Its collision children, if any.
- Any attached camera or mount prims.

Mental model: `robotiq_base_link` is the parent coordinate frame for the gripper base. If the wrist camera is a child of this prim, its transform is interpreted relative to the gripper base frame.

### 3. Find The D415 Mount Prim

Search the Stage tree for:

```text
D415_to_Robotiq_Mount
```

This exact name matters because the code currently hardcodes:

```python
mesh_names=["robotiq_base_link/visuals/D415_to_Robotiq_Mount"]
```

When you select that prim, inspect:

- Where it is in the tree.
- Whether it is under `robotiq_base_link/visuals`.
- Its transform relative to the gripper.
- Whether it has a mesh child or directly contains mesh geometry.
- Its material.
- Whether it has collision geometry or is visual-only.

This is one of the main D405 migration targets. The new D405 mount USD must either:

- Use a new prim name and update the code to point to it, or
- Keep a backward-compatible prim alias with the old path.

The cleaner option is to rename the code/config to the D405 prim path after we know the exact D405 USD hierarchy.

### 4. Find The Wrist Camera Prim

Search for:

```text
rgb_wrist_camera
```

Expected path:

```text
/World/Robot/robotiq_base_link/rgb_wrist_camera
```

This prim is important because the Isaac Lab sensor config refers to it by path. Select it and inspect:

- Its parent.
- Its translation.
- Its rotation/orientation.
- Whether it is a real USD camera prim or just a transform/location prim.
- Whether the camera axes look reasonable in the viewport.

Then compare what you see in the viewer to the config values:

```python
pos=(0.0182505, -0.00408447, -0.0689107)
rot=(0.34254336, -0.61819255, -0.6160212, 0.347879)
```

Important: the viewer shows the transform stored in the live USD scene. The Isaac Lab `TiledCameraCfg` may create or override camera sensor properties at runtime. So do not assume the USD alone contains the final calibrated camera model. The USD hierarchy and the Python sensor config together define the simulated camera.

### 5. Inspect Coordinate Frames

Turn on frame/axis visualization if available. Then inspect:

- `wrist_3_link`
- `robotiq_base_link`
- `rgb_wrist_camera`
- The D415 mount prim

The core question is:

```text
How is the camera frame placed relative to the gripper frame?
```

For the D405 migration, this relationship changes because:

- The D405 physical body is different.
- The D405 optical center is different.
- The D405 mount geometry is different.
- The camera may sit closer/farther from the gripper.
- The viewing direction may be tilted differently.

This is why we need new camera extrinsics, not just a mesh swap.

### 6. Inspect The Robot Link And Joint Structure

In the Stage tree, look for the arm links:

```text
base_link
shoulder_link
upper_arm_link
forearm_link
wrist_1_link
wrist_2_link
wrist_3_link
robotiq_base_link
```

Also look for joints:

```text
shoulder_pan_joint
shoulder_lift_joint
elbow_joint
wrist_1_joint
wrist_2_joint
wrist_3_joint
finger_joint
```

The viewer launch printed these body names:

```text
base_link
shoulder_link
upper_arm_link
forearm_link
wrist_1_link
wrist_2_link
wrist_3_link
robotiq_base_link
left_outer_knuckle
right_outer_knuckle
right_inner_knuckle
left_inner_knuckle
left_outer_finger
right_outer_finger
right_inner_finger
left_inner_finger
```

This helps separate the robot structure from the camera/mount additions.

For the D405 task, the arm and gripper joint structure should mostly stay the same. The change should be concentrated near the wrist/gripper visual/camera area.

### 7. Inspect Visual Geometry Versus Collision Geometry

Many robot USDs have separate prims for:

- Visual meshes: what you see.
- Collision meshes: simplified geometry used by physics.

When you select the mount or camera body, check whether it has collision geometry.

This matters because a camera mount can affect physics in two different ways:

- Visual-only: it appears in rendering but does not collide.
- Collision-enabled: it can hit objects, the table, or the robot itself.

For policy training, this difference matters. If the D405 mount physically sticks out more than the D415 mount, and we ignore its collision geometry, the policy may learn motions that would collide on the real robot.

### 8. Inspect Materials And Appearance

Select the mount and camera body visuals and inspect the material assignments.

For RGB policies, appearance matters because the wrist camera can see:

- The gripper fingers.
- The mount.
- The camera body edge, depending on field of view.
- Reflections or dark plastic regions.

The current RGB randomization includes the mount mesh:

```python
mesh_names=["robotiq_base_link/visuals/D415_to_Robotiq_Mount"]
```

That means the authors considered the wrist mount appearance important enough to randomize. For D405, we should make sure the new D405 mount visual is included in the same kind of randomization.

### 9. Inspect The Camera View Itself

If the viewer exposes the camera viewport or lets you switch to a camera prim, switch to the wrist camera:

```text
rgb_wrist_camera
```

Look for:

- Does the gripper block the image?
- Is the camera looking toward the workspace?
- Is the table/workpiece area visible?
- Is the optical axis tilted downward enough?
- Are objects at the expected distance?
- Is the mount or gripper visible in the image?

For D405, this is central because the D405 is designed for closer-range sensing than the D415. The D405 wrist setup may intentionally place the camera close to the gripper and pointed at the manipulation site.

### 10. Inspect Camera Intrinsics Separately From Pose

The camera pose is the extrinsic relationship:

```text
robotiq_base_link -> rgb_wrist_camera
```

The camera intrinsics describe the pinhole model:

- focal length
- aperture/sensor size
- image width/height
- clipping range

In this repo, the wrist camera Python config currently uses:

```python
height=240
width=320
focal_length=24.55
```

and in the camera alignment config:

```python
height=480
width=640
focal_length=24.55
```

In the viewer, do not assume the camera body mesh tells you the intrinsics. The visible camera model is geometry. The rendered image model comes from sensor configuration.

For D405, we need to update both:

- extrinsics: where the camera is mounted
- intrinsics: how the camera projects 3D points into pixels

### 11. Inspect References And Asset Composition

Select `/World/Robot` and look for references or payloads in the property panel.

USD files are often composed from references to other USD files. The live scene may say:

```text
/World/Robot references ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd
```

Inside that USD, some meshes may reference other mesh files.

For the D405 migration, this tells us whether we should:

- Edit the robot USD directly.
- Create a new D405 robot USD that references most of the old robot and overrides only the wrist mount/camera.
- Export a fully flattened USD.

The second option is often cleaner if supported by the existing asset workflow.

### 12. Inspect What Is D415-Specific

Make a mental list of every D415-specific thing visible in the USD:

- Mount prim name.
- Mount mesh geometry.
- Camera body geometry, if present.
- Camera prim placement.
- Any material named after D415.
- Any collision geometry attached to the old mount.
- Any hardcoded visual path under `robotiq_base_link/visuals`.

Then separate these into two categories:

1. Things that must be changed for physical accuracy.
2. Things that only matter because code refers to their names.

Example:

```text
D415_to_Robotiq_Mount
```

This is both physical and code-relevant:

- Physical: wrong mount geometry for a D405.
- Code-relevant: visual randomization searches for this exact prim path.

### 13. Inspect What Should Stay Stable

Not everything should change.

Things that should ideally stay stable:

- UR5e arm link names.
- UR5e joint names.
- Robotiq gripper joint names.
- `robotiq_base_link` name.
- `rgb_wrist_camera` name, if possible.
- The robot root path convention under `/World/envs/env_*/Robot` in task environments.

Keeping these stable reduces code churn.

For example, if the new D405 USD keeps:

```text
robotiq_base_link/rgb_wrist_camera
```

then these configs do not need path changes:

```python
prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera"
camera_path_template="/World/envs/env_{}/Robot/robotiq_base_link/rgb_wrist_camera"
```

Only the pose and intrinsics need to change.

### 14. What To Screenshot Or Record

While exploring, capture screenshots of:

- Full robot view.
- Close-up of the gripper, old D415 mount, and camera.
- Stage tree expanded at `robotiq_base_link`.
- Properties panel for `D415_to_Robotiq_Mount`.
- Properties panel for `rgb_wrist_camera`.
- Camera view from `rgb_wrist_camera`, if available.

These screenshots become useful when comparing the old D415 setup against the new D405 setup.

### 15. The Main Questions To Answer In The Viewer

By the end of the viewer pass, we want concrete answers to these:

1. What is the exact old D415 mount prim path?
2. Is `rgb_wrist_camera` present in the USD, or only created by Python config?
3. What is the parent of `rgb_wrist_camera`?
4. Does the mount have collision geometry?
5. Does the camera body have collision geometry?
6. Are there D415-specific material names or mesh references?
7. Can we keep `robotiq_base_link/rgb_wrist_camera` unchanged for D405?
8. What should the new D405 mount prim be named?
9. Does the D405 mount STL need to become a USD mesh asset before it can be cleanly referenced?
10. Should the D405 robot be a separate USD variant instead of replacing the D415 USD?

The most important conclusion: the viewer should help us decide the exact USD surgery. We are trying to avoid guessing from filenames alone.
