# D405 Wrist Camera Migration Guide

## Goal

Replace the current UWLab D415 wrist-camera and D415 wrist-mount assumption
with the lab Intel RealSense D405 wrist camera and D405 Robotiq 2F-85 mount.

This is a local wrist/gripper migration:

```text
Robot/robotiq_base_link/rgb_wrist_camera
```

It should not change the table origin, the lab Vention origin, the robot root
pose, object reset X/Y ranges, or the reset-state file schema.

The migration has four pieces:

```text
1. D405 mount/camera geometry in the robot USD
2. D405 color-camera intrinsics
3. D405 wrist-camera extrinsics
4. UWLab config updates and validation
```

Replacing only the STL is not enough. The policy camera is created from Python
config, and the D405 has different intrinsics, optical center, body geometry,
and mount geometry from the old D415 setup.

## Values To Get From Hardware Today

Fill this in directly from the D405 hardware and Spark calibration output:

```text
Spark camera key = ?
  example: /spark/cameras/lightning/wrist_1
  use the actual key for the selected lab right arm

D405 serial number = ?

D405 color profile used for calibration = ? x ? @ ? fps
D405 color profile used for policy sim = ? x ? @ ? fps

D405 color intrinsics at calibration profile:
  fx = ?
  fy = ?
  cx = ?
  cy = ?
  image_size = [?, ?]
  distortion_model = ?
  distortion_coeffs = [?, ?, ?, ?, ?]

D405 color intrinsics at policy profile:
  fx = ?
  fy = ?
  cx = ?
  cy = ?
  image_size = [?, ?]
  distortion_model = ?
  distortion_coeffs = [?, ?, ?, ?, ?]

Spark hand-eye result:
  flange_from_camera.rotation_matrix = ?
  flange_from_camera.translation_vector_m = ?

Calibration quality:
  reprojection_error_mean_px = ?
  reprojection_error_max_px = ?
  target_pose_translation_std_m = ?
  target_pose_rotation_error_mean_deg = ?

UWLab final wrist camera offset:
  parent prim = Robot/robotiq_base_link
  child prim = rgb_wrist_camera
  robotiq_base_link_from_rgb_wrist_camera.pos = ?
  robotiq_base_link_from_rgb_wrist_camera.rot = ?  # w, x, y, z
  Isaac camera convention = opengl

D405 robot asset:
  D405 robot USD path = ?
  D405 mount visual prim path = ?
  D405 mount collision prim path = ?
  wrist mount randomization mesh path = ?
```

If Spark does not write the distortion model enum today, still save the
coefficients and image size. We can add the model enum to the capture script
later if needed.

## Current UWLab State

### Robot Asset Still Names The D415

The current robot asset config points to a D415-mounted robot USD:

```text
source/uwlab_assets/uwlab_assets/robots/ur5e_robotiq_gripper/ur5e_robotiq_2f85_gripper.py
```

Current USD name:

```text
ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd
```

For the migration, create a D405 robot USD variant first. Do not overwrite the
D415 robot until the D405 version is validated.

### Wrist Camera Is Runtime-Created

The wrist camera is created by config at runtime under:

```text
{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera
```

The important config files are:

```text
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/camera_align_cfg.py
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/data_collection_rgb_cfg.py
```

Current D415-style wrist camera values:

```python
prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera"
height=240, width=320                 # data_collection_rgb_cfg.py
height=480, width=640                 # camera_align_cfg.py
pos=(0.0182505, -0.00408447, -0.0689107)
rot=(0.34254336, -0.61819255, -0.6160212, 0.347879)
convention="opengl"
spawn=sim_utils.PinholeCameraCfg(focal_length=24.55)
```

### D415 Mount Mesh Is Hardcoded For Randomization

RGB randomization currently targets:

```text
robotiq_base_link/visuals/D415_to_Robotiq_Mount
```

The D405 migration must update that path to the D405 mount mesh prim after the
D405 robot USD is created.

## D405 Mount Assets Already Present

UWLab already has the printable D405 mount mesh:

```text
source/uwlab_assets/data/Props/Mounts/D405WristMount/d405_wrist_mount_30deg_37mm.stl
source/uwlab_assets/data/Props/Mounts/D405WristMount/README.md
```

It came from Spark:

```text
/data/kanth042/repos/spark-data-collection/Hardware/CAD/camera_mounts/d405_wrist_mount/printable/d405_wrist_mount_30deg_37mm.stl
```

Editable source in Spark:

```text
/data/kanth042/repos/spark-data-collection/Hardware/CAD/camera_mounts/d405_wrist_mount/source/d405_wrist_mount_30deg_37mm.step
/data/kanth042/repos/spark-data-collection/Hardware/CAD/camera_mounts/d405_wrist_mount/source/d405_wrist_mount_parametric.py
```

Nominal geometry documented by Spark:

```text
D405 orientation: landscape
yaw / roll: 0 deg / 0 deg
plate pitch: 30 deg
D405 M3 hole station: 37 mm
acceptable station range: 36-38 mm
flat lead-in before tilted plate: 0 mm
D405 M3 hole X centers: +/-10 mm
```

## Frame Rules

### Keep These Names Stable

Keep these paths stable unless we find a strong reason not to:

```text
Robot/base_link
Robot/wrist_3_link
Robot/robotiq_base_link
Robot/robotiq_base_link/rgb_wrist_camera
```

Keeping `rgb_wrist_camera` under `robotiq_base_link` lets existing observation
configs continue to use:

```python
SceneEntityCfg("wrist_camera")
```

and:

```python
camera_path_template="/World/envs/env_{}/Robot/robotiq_base_link/rgb_wrist_camera"
```

### Do Not Mix World Layout With Wrist Camera Offset

The D405 offset is local to `robotiq_base_link`.

It is not measured from the Isaac world origin, the Vention CAD origin, the
tabletop center, or the robot root pose. The lab Vention migration places the
robot/table in each env. The D405 migration places a camera under the gripper.

### Spark Result Is Not Directly The UWLab Offset

Spark wrist calibration solves:

```text
flange_from_camera
```

UWLab needs:

```text
robotiq_base_link_from_rgb_wrist_camera
```

because the simulated camera is parented to:

```text
Robot/robotiq_base_link
```

So the transform conversion is:

```text
robotiq_base_link_from_camera
  = inverse(flange_from_robotiq_base_link) @ flange_from_camera
```

The value of `flange_from_robotiq_base_link` must come from the robot/gripper
USD or from an agreed mechanical transform. Do not assume it is identity.

### Camera Convention Must Be Converted

Spark/RealSense/OpenCV calibration is optical-frame oriented. UWLab currently
uses:

```python
TiledCameraCfg.OffsetCfg(..., convention="opengl")
```

That means the final `offset.rot` must be expressed for Isaac's OpenGL camera
convention, not pasted directly from a RealSense/OpenCV optical-frame rotation.

Practical rule:

```text
calibration result -> mechanical parent transform -> camera convention conversion -> UWLab offset
```

## Spark Calibration Workflow

Work in the Spark repo:

```bash
cd /data/kanth042/repos/spark-data-collection
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
```

Set the camera key once. Replace this with the actual selected right-arm key if
the lab naming uses `thunder` or another arm name:

```bash
export D405_CAMERA=/spark/cameras/lightning/wrist_1
```

### 1. Create `sensors.local.yaml`

Create:

```text
/data/kanth042/repos/spark-data-collection/data_pipeline/configs/sensors.local.yaml
```

from:

```text
/data/kanth042/repos/spark-data-collection/data_pipeline/configs/sensors.example.yaml
```

Minimal wrist-camera entry:

```yaml
sensors:
  /spark/cameras/lightning/wrist_1:
    serial_number: "<D405_SERIAL>"
```

### 2. Verify ChArUco Detection

```bash
python data_pipeline/debug_charuco_detection.py \
  --camera "$D405_CAMERA" \
  --sensors-file data_pipeline/configs/sensors.local.yaml \
  --width 640 \
  --height 480 \
  --fps 30 \
  --save-annotated-dir /tmp/d405_charuco_debug
```

Default board assumptions in Spark:

```text
dictionary: DICT_4X4_50
squares: 6 x 9
square length: 0.03 m
marker length: 0.022 m
```

On the server, inspect the saved annotated PNGs instead of relying on an OpenCV
window.

### 3. Record Calibration Poses

```bash
python data_pipeline/record_calibration_poses.py \
  --active-arms lightning \
  --output-file data_pipeline/configs/calibration_poses.local.json \
  --min-poses 10
```

Use the selected real arm name. Record 10-15 good wrist poses where the D405
sees the ChArUco board clearly. Prefer fewer clean views over many weak views.

Important Spark assumption:

```text
UR TCP must be set to the tool flange.
```

### 4. Run Hand-Eye Calibration

```bash
python data_pipeline/calibrate_rig.py \
  --sensors-file data_pipeline/configs/sensors.local.yaml \
  --poses-file data_pipeline/configs/calibration_poses.local.json \
  --output-file data_pipeline/configs/calibration.local.json \
  --camera "$D405_CAMERA" \
  --width 640 \
  --height 480 \
  --fps 30
```

Spark writes:

```text
cameras["<D405_CAMERA>"].intrinsics.camera_matrix
cameras["<D405_CAMERA>"].intrinsics.distortion_coeffs
cameras["<D405_CAMERA>"].intrinsics.image_size
cameras["<D405_CAMERA>"].hand_eye_calibration.rotation_matrix
cameras["<D405_CAMERA>"].hand_eye_calibration.translation_vector
```

The RealSense code that writes intrinsics is:

```text
/data/kanth042/repos/spark-data-collection/data_pipeline/calibration/realsense.py
```

It opens the color stream and saves the RealSense factory color intrinsics for
the requested width/height/fps profile.

### 5. Validate Calibration

```bash
python data_pipeline/validate_calibration_click.py \
  --camera "$D405_CAMERA" \
  --sensors-file data_pipeline/configs/sensors.local.yaml \
  --calibration-file data_pipeline/configs/calibration.local.json
```

This tool uses RGB/depth and the solved hand-eye transform to project clicked
image points into the robot base frame. It may need X forwarding or a local GUI.
If that is awkward today, at minimum save the calibration output and annotated
ChArUco frames so we can inspect them.

## Converting Hardware Values To UWLab

### Intrinsics

Spark gives:

```python
camera_matrix = [
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0],
]
image_size = [width, height]
distortion_coeffs = [...]
```

UWLab currently uses only:

```python
sim_utils.PinholeCameraCfg(focal_length=24.55)
```

For D405, use the full intrinsic matrix if the installed IsaacLab camera API
supports it. If that API is not available, derive the closest pinhole focal/FOV
values from `fx`, `fy`, `width`, and `height`, and document the approximation.

For a pinhole camera:

```text
horizontal_fov = 2 * atan(width  / (2 * fx))
vertical_fov   = 2 * atan(height / (2 * fy))
```

If policy sim runs at 320 x 240 but calibration was captured at 640 x 480, use
direct D405 intrinsics for 320 x 240 whenever possible. If the exact profile
cannot be queried today, use the half-resolution scaling as a temporary value:

```text
fx_320 = fx_640 * 0.5
fy_240 = fy_480 * 0.5
cx_320 = cx_640 * 0.5
cy_240 = cy_480 * 0.5
```

Distortion note:

- Isaac's normal pinhole path does not apply RealSense distortion coefficients.
- For policy comparison, either undistort real D405 images before comparing to
  sim, or add a distortion-aware render/postprocess path later.

### Extrinsics

Convert the Spark `flange_from_camera` result into the local UWLab camera offset
under `robotiq_base_link`.

The final values go here:

```python
wrist_camera = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(...),          # robotiq_base_link_from_rgb_wrist_camera translation
        rot=(...),          # w, x, y, z
        convention="opengl",
    ),
    spawn=...,
)
```

Do not tune this by looking at the viewer alone. The viewer can confirm gross
orientation, but the final value should come from the hardware calibration plus
the known `flange_from_robotiq_base_link` transform.

## UWLab Asset Work

### 1. Create A D405 Robot USD Variant

Create a new robot asset variant rather than replacing the D415 USD in place.
The new USD should:

- keep the same UR5e and Robotiq link/joint names;
- keep `robotiq_base_link`;
- remove or hide the D415 mount visual;
- add the D405 mount visual mesh;
- add an approximate D405 mount collision shape;
- optionally add D405 camera body visual/collision geometry;
- preserve compatible `metadata.yaml` beside the robot USD.

Recommended prim names:

```text
robotiq_base_link/visuals/D405_to_Robotiq_Mount
robotiq_base_link/collisions/D405_to_Robotiq_Mount
```

### 2. Add A D405 Robot Config

File:

```text
source/uwlab_assets/uwlab_assets/robots/ur5e_robotiq_gripper/ur5e_robotiq_2f85_gripper.py
```

Add a D405 articulation config beside the existing D415 config, then validate
before deciding whether the D405 becomes the default.

### 3. Patch Camera Align First

File:

```text
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/camera_align_cfg.py
```

Patch:

```text
wrist_camera.offset.pos
wrist_camera.offset.rot
wrist_camera.spawn
```

Use the higher-resolution profile here, likely 640 x 480, because this config
is for visual alignment and debugging.

### 4. Patch RGB Data Collection

File:

```text
source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/data_collection_rgb_cfg.py
```

Patch:

```text
wrist_camera.offset.pos
wrist_camera.offset.rot
wrist_camera.spawn
randomize_wrist_camera.base_position
randomize_wrist_camera.base_rotation
randomize_wrist_camera_focal_length.focal_length_range
```

Keep the observation key stable:

```text
wrist_rgb
```

The policy observation should still be the processed wrist camera tensor unless
we intentionally change the policy interface.

### 5. Patch Mount Appearance Randomization

In both normal RGB and OOD RGB event configs, replace:

```text
robotiq_base_link/visuals/D415_to_Robotiq_Mount
```

with the final D405 mount visual prim path, for example:

```text
robotiq_base_link/visuals/D405_to_Robotiq_Mount
```

## Validation

### Asset Validation

Launch a scene and check:

- D405 mount mesh appears under `robotiq_base_link`;
- old D415 mount is gone, hidden, or not used;
- D405 mount collision roughly covers the printed mount envelope;
- camera path exists at
  `/World/envs/env_0/Robot/robotiq_base_link/rgb_wrist_camera`;
- selecting `rgb_wrist_camera` shows the gripper/workspace region.

### Camera Observation Validation

The viewport is not the policy tensor. Also validate:

```text
obs["policy"]["wrist_rgb"]
```

Expected processing in the current pipeline:

```text
RGB tensor from wrist_camera
float conversion / 255
CHW layout
resize to 224 x 224 for policy observation
```

### Calibration Validation

Use Spark quality values and visual checks:

- enough valid pose pairs;
- reasonable reprojection error;
- stable base-to-target estimates across poses;
- no obvious sign/axis mistake in the converted UWLab offset;
- real D405 frame and sim wrist frame have similar gripper silhouette,
  object/workspace placement, and field of view for the same robot pose.

## Recommended Implementation Order

1. Choose the exact Spark camera key for the lab right-arm D405.
2. Fill `sensors.local.yaml` with the D405 serial number.
3. Verify ChArUco detection and save annotated images.
4. Record 10-15 clean calibration poses with TCP set to tool flange.
5. Run `calibrate_rig.py` and inspect `calibration.local.json`.
6. Copy the hardware values into the fill-in worksheet at the top of this doc.
7. Derive `robotiq_base_link_from_rgb_wrist_camera`.
8. Convert camera convention to UWLab `convention="opengl"`.
9. Build the D405 robot USD variant with visual and collision mount geometry.
10. Add a D405 robot articulation config.
11. Patch `camera_align_cfg.py` and validate the wrist view.
12. Patch `data_collection_rgb_cfg.py`.
13. Patch wrist camera randomization base values.
14. Patch D405 mount visual-randomization mesh names.
15. Save real/sim wrist image comparisons from fixed robot poses.
16. Decide whether old D415 RGB policies are usable for evaluation only, or
    whether D405 RGB data collection/fine-tuning is required.

## Acceptance Criteria

The D405 migration is ready for policy evaluation when:

- D405 robot config can be selected without changing old D415 baselines;
- D405 mount visual and collision are present;
- `rgb_wrist_camera` remains under `robotiq_base_link`;
- D405 intrinsics replace the D415 focal-only config;
- D405 extrinsics replace the D415 wrist-camera pose;
- wrist camera randomization is centered on the D405 pose;
- wrist mount visual randomization targets the D405 mesh path;
- policy `wrist_rgb` tensor shows the expected real lab workspace;
- fixed real/sim robot poses produce comparable wrist views.

## Relationship To The Lab Vention Migration

The lab Vention migration changes scene-level placement:

```text
table entity pose
robot root pose
object reset X/Y/Z ranges
camera-align scene layout
```

The D405 migration changes wrist-level sensor geometry:

```text
robotiq_base_link -> rgb_wrist_camera
D405 mount visual/collision geometry
D405 camera intrinsics
D405 wrist-camera randomization base values
```

They must agree during validation because the camera view depends on both the
robot's scene placement and the local wrist-camera offset. In code and notes,
keep them separate: scene layout values stay in the lab layout/config files,
and D405 values stay with the robot/camera configs.
