# Running Notes

Implementation notes only: concrete change surface, decisions, TODOs, validation steps, and repo edits. Conceptual explanations go in `concept_notes.md`.

## 2026-05-18 - D415 to D405 Wrist Camera and Mount Change Surface

Goal: replace the original OmniReset D415 wrist camera and D415 wrist mount assumption with the lab D405 wrist camera and D405 wrist mount.

### Current Findings

- A D405 wrist mount CAD asset already exists locally:
  - `source/uwlab_assets/data/Props/Mounts/D405WristMount/d405_wrist_mount_30deg_37mm.stl`
  - `source/uwlab_assets/data/Props/Mounts/D405WristMount/README.md`
- The robot articulation still points to the D415-mounted calibrated robot USD:
  - `source/uwlab_assets/uwlab_assets/robots/ur5e_robotiq_gripper/ur5e_robotiq_2f85_gripper.py`
  - Current USD name: `ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd`
- The kinematics helper loads calibration metadata from a `metadata.yaml` file next to the robot USD path:
  - `source/uwlab_assets/uwlab_assets/robots/ur5e_robotiq_gripper/kinematics.py`
  - Therefore, a new D405 robot USD path also needs a compatible `metadata.yaml` beside it.
- The simulated wrist camera is configured separately from the robot USD:
  - `source/uwlab_tasks/uwlab_tasks/manager_based/ur5e/stacking/camera_align_cfg.py`
  - `source/uwlab_tasks/uwlab_tasks/manager_based/ur5e/stacking/data_collection_rgb_cfg.py`
- The current wrist camera prim path is:
  - `{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera`
  - If the new D405 USD keeps this prim name and parent link, fewer code changes are needed.
- The existing wrist camera pose and focal length are still D415-style calibrated values:
  - Position: `(0.0182505, -0.00408447, -0.0689107)`
  - Rotation quaternion: `(0.34254336, -0.61819255, -0.6160212, 0.347879)`
  - Focal length: `24.55`
- Wrist camera randomization uses the same old base position, base rotation, and focal-length range in `data_collection_rgb_cfg.py`.
- Mount visual randomization is hardcoded to the D415 mount mesh name:
  - `robotiq_base_link/visuals/D415_to_Robotiq_Mount`
  - This appears in both regular RGB randomization and OOD RGB randomization.
- The sim-to-real documentation still describes the D415 wrist camera, D415 mount, and D415-mounted USD:
  - `docs/source/publications/omnireset/sim2real.rst`

### Change Layers

1. Asset/USD geometry

   We need a D405-mounted UR5e + Robotiq USD, or a separate D405 robot asset variant. This USD should include the D405 mount, the relevant camera visual/collision geometry if desired, and the simulated wrist camera prim.

   The least disruptive prim path is:

   `$Robot/robotiq_base_link/rgb_wrist_camera$`

   Keeping this path means the existing camera configs can keep their `prim_path` and only update the pose/intrinsics.

2. Robot articulation config

   The global robot config currently points to the D415 USD. Changing this directly affects camera alignment, RGB data collection, RL state training/evaluation, sysid, and reset-state generation.

   Safer path: add a D405 robot asset/config first, validate it, then decide whether to make it the default.

3. Kinematics metadata

   `kinematics.py` looks for `metadata.yaml` next to the robot USD. If we point to a new D405 USD directory, that directory needs compatible metadata.

   If only the camera and wrist mount change, the arm calibration metadata may stay the same. If the robot calibration or URDF changes, the metadata must be regenerated or updated.

4. Camera extrinsics and intrinsics

   The D405 is a different physical camera from the D415, and the mount changes the camera pose relative to `robotiq_base_link`.

   These fields need new D405-calibrated values:

   - `wrist_camera.offset.pos`
   - `wrist_camera.offset.rot`
   - `wrist_camera.spawn.focal_length`

   They appear in:

   - `camera_align_cfg.py`
   - `data_collection_rgb_cfg.py`

5. Camera randomization

   The randomization base values should follow the new D405 calibration:

   - `randomize_wrist_camera.base_position`
   - `randomize_wrist_camera.base_rotation`
   - `randomize_wrist_camera_focal_length.focal_length_range`

   The amount of randomization may need to change because the D405 has different optics, working distance, and depth/noise behavior than the D415.

6. Mount visual randomization

   The D415 mesh path must either be replaced or preserved as a backward-compatible alias.

   Current hardcoded mesh name:

   `$robotiq_base_link/visuals/D415_to_Robotiq_Mount$`

   For a clean D405 asset, update this to the actual D405 mount mesh prim path in the new USD.

7. Documentation and workflow

   The sim-to-real docs need to stop describing the wrist camera as D415-specific once the D405 path is validated.

   Documentation should include:

   - D405 mount STL path
   - D405 robot USD path
   - D405 camera alignment workflow
   - Where to paste calibrated `pos`, `rot`, and `focal_length`
   - Whether old D415 policies/datasets are still compatible

8. Validation

   Before training or collecting new data, validate:

   - The new robot USD loads.
   - The `rgb_wrist_camera` prim exists under `robotiq_base_link`.
   - `CameraAlignEnvCfg` renders a nonblank wrist camera image.
   - The camera alignment script can tune the D405 pose.
   - The visual randomization code finds the D405 mount mesh path.
   - The kinematics metadata still loads.
   - RGB data collection produces sensible wrist views.

### Important Design Decision

The wrist camera change has two separate layers:

1. The physical robot asset layer: the USD geometry, mount, camera body, prim names, and metadata location.
2. The task/config layer: Isaac camera sensor pose, focal length, randomization, and camera alignment values.

For D405 migration, both layers need to agree. Updating only the STL or only the camera config is not enough.

### Open Questions

- Should the D405 robot become the default `UR5E_ARTICULATION`, or should we first add a separate D405 variant?
- What are the final calibrated D405 extrinsics relative to `robotiq_base_link`?
- What focal length or full camera intrinsics should the simulated D405 use?
- What exact prim name will the D405 mount mesh have inside the USD?
- Do old D415 wrist-camera policies remain useful after viewpoint adaptation, or should D405 RGB data be recollected before serious policy evaluation?

## 2026-05-18 - USD GUI Inspection Setup

- Added `scripts_v2/tools/view_ur5e_robot_usd.py` to spawn the current UWLab UR5e + Robotiq robot USD in a minimal Isaac Lab scene.
- The script prints the robot USD path, joint names, and body names, then keeps the simulation alive for GUI/WebRTC inspection.
- Local Isaac Sim is available through the `uwlab-isaac51` conda environment:
  - `/data/kanth042/envs/uwlab-isaac51/bin/python`
  - `/data/kanth042/envs/uwlab-isaac51/bin/isaacsim`
- The repo does not currently have an `_isaac_sim` symlink, so `uwlab.sh` needs `CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51` when launched from this shell.
- Isaac Sim/Kit currently prompts for NVIDIA Omniverse EULA acceptance on first use. Do not set `OMNI_KIT_ACCEPT_EULA=YES` until the user explicitly confirms.
- Intended WebRTC/private-network launch command after EULA acceptance:

  ```bash
  OMNI_KIT_ACCEPT_EULA=YES LIVESTREAM=2 ENABLE_CAMERAS=1 CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 \
      ./uwlab.sh -p scripts_v2/tools/view_ur5e_robot_usd.py --livestream 2
  ```

- `LIVESTREAM=2` / `--livestream 2` is the private/local network WebRTC mode in Isaac Lab. `LIVESTREAM=1` is for public-network WebRTC and may require `PUBLIC_IP`.
- User accepted the NVIDIA Omniverse/Kit EULA on 2026-05-18. Persisted acceptance here:
  - `/data/kanth042/envs/uwlab-isaac51/lib/python3.11/site-packages/isaacsim/kit/EULA_ACCEPTED`
- Successfully launched the current D415-mounted UR5e + Robotiq USD viewer with:

  ```bash
  CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 LIVESTREAM=2 ENABLE_CAMERAS=1 \
      ./uwlab.sh -p scripts_v2/tools/view_ur5e_robot_usd.py --livestream 2
  ```

- NVIDIA StreamSDK ETLI trace generation is disabled by default via `/app/livestream/webrtcEtli=false` in the Isaac Lab/Isaac Sim Kit experience files.
- Later on 2026-05-18, the viewer entry points were consolidated into `scripts_v2/tools/view_ur5e_camera_scene.py`:
  - `--view scene` for the full camera-alignment scene.
  - `--view robot` for the robot-only USD view.
  - `--view table` for the OmniReset Vention/support asset.
- Running process details from this launch:
  - wrapper PID: `222796`
  - Python/Isaac PID: `222812`
  - server IP: `134.84.150.140`
  - HTTP/service port: `8011`
  - WebRTC signaling TCP port: `49100`
  - WebRTC media UDP port: `47998`
  - readiness check: `http://127.0.0.1:8011/v1/streaming/ready` returned `Status: Streaming session active (keep alive)`
- The spawned USD path was:
  - `https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated/ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd`

## 2026-05-18 - Lab Vention / Right-Arm Migration Session

This session was about understanding the existing UWLab scene well enough to replace the old flat Vention/table setup with our lab Vention frame, then eventually replace the wrist D415 with a D405.

### Process Cleanup / Livestream Notes

- The Isaac/Kit process does not always shut down quickly on `Ctrl+C`, especially with livestream/WebRTC active.
- We added hard-exit signal handling to the viewer scripts so `Ctrl+C` exits faster from our inspection tools.
- We stopped patching many Kit experience files directly and switched to a more centralized override:
  - `uwlab.sh` exports `ISAACLAB_KIT_ARGS`.
  - `/data/kanth042/repos/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py` appends `ISAACLAB_KIT_ARGS` into Isaac Lab's Kit launch args.
- Current default Kit args added through that path:

  ```bash
  --/app/livestream/webrtcEtli=false --/app/livestream/outDirectory=/tmp/uwlab_livestream
  ```

- Purpose:
  - prevent new `NvStreamer*.etli` trace files from accumulating by default;
  - push any livestream output that still appears into `/tmp/uwlab_livestream`;
  - avoid repeated one-off patches to individual `.kit` files.
- Existing Kit logs were cleaned. At the time of cleanup, logs were not the dominant disk issue compared with accumulated livestream trace files.

### Viewer Script Consolidation

- We moved toward one reusable viewer instead of separate throwaway scripts.
- Main viewer now used for this work:

  ```bash
  scripts_v2/tools/view_ur5e_camera_scene.py
  ```

- Useful modes:
  - `--view scene`: full camera-alignment scene.
  - `--view robot`: current UR5e + Robotiq + D415 wrist-mount robot only.
  - `--view table`: old/current OmniReset table/support asset only.
  - `--view lab-right-arm`: new lab Vention preview with the simulated robot placed at the selected right-arm station.
- The viewer has a `--show-ground` option for isolated inspection modes.
- The lab Vention mode has a `--show-cad-arms` option:
  - default hides the CAD-exported UR arms so we see the simulated UWLab robot;
  - `--show-cad-arms` overlays the CAD arms for alignment debugging.

### Important Camera Viewer Clarification

- Selecting `rgb_wrist_camera` in the Isaac viewport camera menu changes the viewport camera.
- That does not automatically show "what the policy sees" as a live sensor debug pane.
- The policy sees the rendered image from the configured Isaac camera sensor attached to the robot.
- For true policy/sensor inspection, we need a script path that explicitly reads the camera tensor/image output and displays or saves it.
- A white viewport from the USD camera can happen because the viewport camera/framing/exposure is not the same thing as the policy camera observation pipeline.

### Existing D415 / Future D405 Takeaway

- The camera values in the D415 config should not be guessed for the D405.
- D405 migration needs values from the real hardware/calibration workflow:
  - camera intrinsics, such as resolution, focal lengths, optical center, distortion model/coefficients;
  - wrist camera extrinsics relative to the gripper/robot frame;
  - an updated simulated camera config that matches the calibrated D405 as closely as needed.
- The D405 replacement is therefore two layers:
  - physical asset layer: new camera body/mount/USD/prim names;
  - task/config layer: sensor pose, focal length or intrinsics, randomization bases, and validation scripts.

### STEP-to-USD Conversion Work

- Added a conversion utility:

  ```bash
  scripts_v2/tools/conversions/convert_step_to_usd.py
  ```

- Added a general USD inspection utility:

  ```bash
  scripts_v2/tools/view_usd_asset.py
  ```

- The first lab Vention STEP preview showed suspicious geometry/alignment:
  - the frame orientation did not match how we wanted to inspect it;
  - after rotation, the asset appeared to float or sit below the floor depending on how the transform was applied;
  - one view suggested the caster/wheel contact plane was not consistent.
- We corrected the `fit-to-ground` logic so it computes the bounding box after the chosen rotation, not just from the raw unrotated CAD bounds.
- We then checked a newer Vention export:

  ```bash
  /data/kanth042/downloads/VentionAssembly_511882_v60.STEP
  ```

- Converted output:

  ```bash
  /data/kanth042/converted_assets/lab_vention_preview_v60/lab_vention_raw.usd
  ```

- Conversion command:

  ```bash
  cd /data/kanth042/repos/UWLab

  CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 \
  ./uwlab.sh -p scripts_v2/tools/conversions/convert_step_to_usd.py \
    --input /data/kanth042/downloads/VentionAssembly_511882_v60.STEP \
    --output /data/kanth042/converted_assets/lab_vention_preview_v60/lab_vention_raw.usd
  ```

- The v60 file looked geometrically correct in the viewer.
- The CAD/USD uses millimeter-scale source values, so the USD is spawned at scale `0.001`.

### Validated Lab Vention Preview Transform

For the converted v60 USD, the preview transform that made the frame orientation and ground contact look right was:

```text
scale = (0.001, 0.001, 0.001)
rot   = (0.5, 0.5, 0.5, 0.5)       # Isaac/USD quaternion order: w, x, y, z
pos   = (1.793445, 0.340075, -0.030851)
```

The command used to inspect the raw v60 asset:

```bash
cd /data/kanth042/repos/UWLab

CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 LIVESTREAM=2 ENABLE_CAMERAS=1 \
./uwlab.sh -p scripts_v2/tools/view_usd_asset.py \
  --usd-path /data/kanth042/converted_assets/lab_vention_preview_v60/lab_vention_raw.usd \
  --fit-to-ground --show-ground \
  --rot 0.5 0.5 0.5 0.5 \
  --eye 2.4 -3.0 1.7 \
  --target 0.0 0.0 0.85 \
  --livestream 2
```

Measured tabletop top surface in this preview:

```text
LAB_TABLETOP_TOP_Z = 0.84235 m
```

This value matters because object spawn/reset positions that are meant to lie on the workspace tabletop must be interpreted relative to this surface, not blindly reused from the old flat-frame scene.

### Lab Setup Interpretation

- The old repo setup is a single-arm UR5e on a flat Vention/table-like support.
- Our lab setup is a larger Vention frame with two vertical posts and two UR5e arms, closer to a humanoid/two-arm station.
- The user chose the right arm visible in the lab preview image as the first target arm.
- We are not renaming the conceptual scene pieces right now. The important thing is to know what the existing names mean:
  - `table` in the current configs is the old Vention/support/workspace asset, not necessarily a generic table in the everyday sense.
  - `ur5_metal_support` is used by the task code as an anchor/reference for object resets and offsets in the old setup.
  - The new lab Vention CAD already includes the physical support geometry for the selected arm station, so we should not visually spawn the old flat `ur5_metal_support` plate on top of it.
- The old `ur5_metal_support` name may still need to survive temporarily as a logical/hidden reference until the reset code is refactored.

### Right Arm Alignment

- In the v60 CAD hierarchy, the selected right-arm CAD group is:

  ```text
  /lab_vention_raw/lab_vention_raw/tn__0_
  ```

- The other arm group is:

  ```text
  /lab_vention_raw/lab_vention_raw/tn__0_1_
  ```

- The CAD UR5 base assembly for the selected right station is:

  ```text
  /lab_vention_raw/lab_vention_raw/tn__0_/tn__ur5e_base_assySTEP_pR
  ```

- Candidate simulated UWLab robot root pose for the right arm:

  ```text
  LAB_RIGHT_ARM_ROBOT_POS = (0.177660, 0.377695, 1.466000)
  LAB_RIGHT_ARM_ROBOT_ROT = (0.7071068, 0.0, 0.7071068, 0.0)
  ```

- We compared this against the local current robot USD:

  ```text
  /data/kanth042/tmp/uwlab-assets-inspect/ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd
  ```

- The simulated robot base geometry is centered in x/y and extends upward from the root, which is consistent with using the CAD base assembly pose as the robot root/base mounting pose.
- The preview with the simulated right arm looked correct to the user.

Command for the current lab-right-arm preview:

```bash
cd /data/kanth042/repos/UWLab

CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 LIVESTREAM=2 ENABLE_CAMERAS=1 \
./uwlab.sh -p scripts_v2/tools/view_ur5e_camera_scene.py \
  --view lab-right-arm \
  --show-ground \
  --livestream 2
```

Optional CAD overlay for checking alignment:

```bash
cd /data/kanth042/repos/UWLab

CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 LIVESTREAM=2 ENABLE_CAMERAS=1 \
./uwlab.sh -p scripts_v2/tools/view_ur5e_camera_scene.py \
  --view lab-right-arm \
  --show-ground \
  --show-cad-arms \
  --livestream 2
```

### World / Origin / Table Reasoning

- The robot base is not assumed to be the world origin.
- The world origin is just the scene coordinate frame chosen by the simulation setup.
- The old scene effectively encoded a particular relationship between:
  - world origin;
  - old Vention/table asset;
  - `ur5_metal_support`;
  - robot root;
  - object spawn/reset area.
- In the lab migration, simply moving the robot root is not enough if objects and reset distributions still use the old support/table anchor.
- The object reset values only make sense if their reference surface and anchor match the actual workspace table.
- If the physical workspace tabletop height changes outside the training/randomization envelope, a learned policy can fail because the camera images, robot reach geometry, and object poses no longer match what it learned.
- The paper/config discussion about varying table height appears to be about the workspace table/object surface, not the Vention frame itself.

### Current Migration Direction

Minimal-change direction for the lab-right-arm migration:

1. Keep existing names where practical so the code remains understandable against the original repo.
2. Spawn the new lab Vention CAD as the scene `table`/support visual for lab previews.
3. Spawn the simulated UWLab UR5e robot at the selected right-arm CAD base pose.
4. Hide the CAD-exported UR arms by default, because the simulated robot should be the active articulation.
5. Keep or replace `ur5_metal_support` carefully:
   - visually, the old support should not be duplicated;
   - logically, existing reset code may still need a named anchor until object spawning is refactored.
6. Re-anchor object reset/spawn distributions to the real workspace tabletop surface instead of assuming the old scene tabletop.
7. Only after the lab geometry is coherent, do the D405 wrist camera asset/calibration migration.

### Immediate Next Engineering Step

Move the hardcoded lab constants out of the viewer script into a shared lab-layout config module so later scene/task configs can reuse the same numbers:

```text
LAB_VENTION_USD_PATH
LAB_VENTION_POS
LAB_VENTION_ROT
LAB_VENTION_SCALE
LAB_RIGHT_ARM_ROBOT_POS
LAB_RIGHT_ARM_ROBOT_ROT
LAB_TABLETOP_TOP_Z
LAB_CAD_ARM_REL_PATHS
```

After that, add a lab-right-arm task scene variant and validate:

- robot loads at the right station;
- lab Vention frame is visual/static;
- old CAD arms are hidden;
- object reset positions land on the real tabletop;
- wrist camera observation is nonblank;
- policy/eval scripts still know which camera observation they are using.

### Tabletop Surface Decision

After inspecting how UWLab models the old workspace table, we decided to follow the same pattern for the lab migration instead of adding a separate HDPE tabletop asset immediately.

Current UWLab pattern:

- `scene.table` is the whole old Vention/workspace asset:

  ```text
  Props/Mounts/UWPatVention/pat_vention.usd
  ```

- The top/contact-looking visual surface inside that asset is:

  ```text
  /vention_pat/visuals/vention_mat
  ```

- RGB appearance randomization targets that inner mesh by relative path:

  ```python
  mesh_names=["visuals/vention_mat"]
  ```

- The table friction randomization is applied to the `table` scene entity, with:

  ```text
  static friction: 0.3 to 0.6
  dynamic friction: 0.2 to 0.5
  restitution: 0.0
  ```

Lab Vention v60 equivalent:

- Keep the lab Vention assembly as the `table` scene entity.
- Do not create a separate tabletop scene object yet.
- Use the tabletop panel prim inside the lab Vention USD as the equivalent of UWLab's `visuals/vention_mat`.
- The identified lab tabletop panel prim is:

  ```text
  /lab_vention_raw/lab_vention_raw/tn__importedPANEL72E663E8E8F2_1147116_om0E
  ```

- Relative to the spawned `/World/Table` parent, the useful path is:

  ```text
  lab_vention_raw/tn__importedPANEL72E663E8E8F2_1147116_om0E
  ```

- Its transformed bounds in the validated lab preview are approximately:

  ```text
  size: 1.485 m x 0.720 m x 0.00635 m
  top z: 0.84235 m
  bottom z: 0.83600 m
  ```

Implication:

- For minimal-change migration, keep `table` as the full support/workspace asset.
- For visual randomization, target the inner tabletop panel prim.
- For object reset/spawn height, use `LAB_TABLETOP_TOP_Z`.
- If collision stability becomes a problem later, then add a clean invisible collision slab, but that should be a validation-driven change, not the default first migration step.

### Lab Right-Arm Camera-Align Config Added

Added a separate lab-right-arm camera alignment environment instead of changing the original camera-align env:

```text
OmniReset-Ur5eRobotiq2f85-CameraAlign-LabRightArm-v0
```

This variant:

- keeps the existing default camera-align env unchanged;
- spawns the lab Vention v60 CAD at the validated transform;
- hides the CAD-exported UR arms by default;
- spawns the simulated UWLab UR5e at the selected right-arm station;
- keeps a hidden `UR5MetalSupport` placeholder so the old scene entity name remains available;
- sets the ground plane at world `z = 0`.

Important implementation detail:

- The lab Vention CAD is spawned as an `AssetBaseCfg` visual in the camera-align variant, not as a physics `RigidObjectCfg`.
- I tried treating the raw STEP-converted CAD as a `RigidObjectCfg`, but Isaac warned that rigid-body properties could not be applied cleanly because the converted CAD contains many instanced prims.
- The camera-align scene does not need the Vention frame to be the contact surface for object physics, so visual-only is the safer minimal change here.
- For future RL/data-collection task migration, we still need to decide whether to:
  - preprocess the lab Vention USD into a cleaner physics asset, or
  - add a clean collision/contact slab while keeping the CAD panel for visuals.

Smoke checks performed:

- `py_compile` passed for the changed camera-align config and viewer script.
- A headless original camera-align env reset/step reached:

  ```text
  env
  reset
  step 7 ['front_camera', 'side_camera', 'wrist_camera']
  ```

- A headless lab-right-arm camera-align env reset/step reached the same marker:

  ```text
  env
  reset
  step 7 ['front_camera', 'side_camera', 'wrist_camera']
  ```

- The earlier USD regex warning from hiding CAD arms was fixed by resolving concrete matching table prims before hiding child CAD arm groups.
- No leftover viewer/Isaac processes were found after the smoke runs.

Command to launch the lab camera-align scene in the viewer:

```bash
cd /data/kanth042/repos/UWLab

CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 LIVESTREAM=2 ENABLE_CAMERAS=1 \
./uwlab.sh -p scripts_v2/tools/view_ur5e_camera_scene.py \
  --view scene \
  --scene-layout lab-right-arm \
  --livestream 2
```

### Viewer Kill Issue Follow-Up

The lab camera-align viewer again resisted `Ctrl+C` during a WebRTC launch.

Observed stuck processes:

```text
bash ./uwlab.sh -p scripts_v2/tools/view_ur5e_camera_scene.py --view scene --scene-layout lab-right-arm --livestream 2
/data/kanth042/envs/uwlab-isaac51/bin/python scripts_v2/tools/view_ur5e_camera_scene.py --view scene --scene-layout lab-right-arm --livestream 2
```

The wrapper was in process group `297760`, with Python/Isaac child `297767`.

Cleanup used:

```bash
kill -TERM -- -297760
kill -KILL -- -297760
kill -KILL 297767
```

After `SIGKILL`, the process was gone; any short-lived `[python] <defunct>` entry is a zombie placeholder and not an active CPU/GPU process.

Follow-up code change:

- Moved the hard-exit signal handler earlier in both viewer scripts:
  - `scripts_v2/tools/view_ur5e_camera_scene.py`
  - `scripts_v2/tools/view_usd_asset.py`
- Previously the handler was installed after `AppLauncher(args_cli)`, leaving a bad startup window where Kit could swallow or delay `Ctrl+C`.
- Now the handler is installed immediately after parsing CLI args and before Isaac Kit starts.
- `py_compile` passed for both scripts.

Emergency kill pattern for future stuck viewer runs:

```bash
ps -eo pid,ppid,pgid,sid,stat,etime,cmd | rg 'view_ur5e_camera_scene|view_usd_asset|uwlab.sh'
kill -TERM -- -<PGID>
sleep 3
kill -KILL -- -<PGID>
```

### Lab Camera-Align Peg/Hole Falling Through Table

Observed in the lab-right-arm camera-align scene: the peg and peg-hole objects
appeared on the ground instead of on the tabletop.

Cause:

- `LabRightArmCameraAlignSceneCfg` inherits `insertive_object` and
  `receptive_object` from the original `RlStateSceneCfg`.
- In the original scene those objects initialize at `(0, 0, 0)` and are later
  reset by task events in the normal RL/data-collection flows.
- The camera-align scene has no reset/randomization events, so the inherited
  `(0, 0, 0)` object roots were left unchanged.
- The lab Vention CAD is currently spawned as a visual `AssetBaseCfg` in this
  viewer because the raw STEP-converted USD has many instanced CAD prims and did
  not accept rigid-body properties cleanly.
- Therefore the visual tabletop was not providing physics contact for the peg.

Temporary fix attempted for the camera-align viewer only:

- Added lab object placement constants in `lab_layout_cfg.py`.
- Measured object bottom offsets from the USD bounding boxes:
  - peg bottom offset: `0.030000 m`
  - peg-hole bottom offset: `0.017703 m`
- Placed the object roots so their visible bottoms sit on the measured lab
  tabletop top plane `LAB_TABLETOP_TOP_Z = 0.84235 m`:
  - peg: `(0.34, 0.20, 0.87235)`
  - peg-hole: `(0.24, 0.20, 0.860053)`
- Overrode both objects in `LabRightArmCameraAlignSceneCfg` as kinematic with
  gravity disabled. This is intentional for this visual/camera-alignment viewer:
  it keeps the objects available for camera inspection without claiming that the
  raw Vention CAD is a finished physics collision asset.
- This was later backed out. It treated a symptom of the bad asset pipeline
  instead of correcting the STEP-to-USD asset processing.

Validation:

- `python -m py_compile` passed for:
  - `camera_align_cfg.py`
  - `lab_layout_cfg.py`

Important follow-up:

- For actual RL/data collection on the lab setup, we should not rely on this
  visual-only workaround. We need either a cleaned lab Vention/table collision
  asset or a simple collision tabletop that matches the real HDPE surface while
  the detailed CAD remains visual.

### Correction: We Jumped the Gun on the STEP Asset Pipeline

After discussing the collision setup again, the bigger mistake was not the peg
and peg-hole symptom itself. The mistake was treating the raw STEP-to-USD output
as if it were already a UWLab-ready scene asset.

What went wrong:

- We started from a Vention STEP file and converted it to a raw USD so it could
  be visualized.
- That was only a CAD preview step.
- We then began using that raw converted USD inside the lab scene/viewer as if it
  were the replacement for the UWLab `pat_vention.usd` table asset.
- That skipped the actual asset-production step: building a USD with meaningful
  `/visuals` and `/collisions` structure.
- Because that step was skipped, downstream issues appeared immediately:
  - the Vention frame was visual-only in the lab scene;
  - object contact with the tabletop was not trustworthy;
  - hiding/removing CAD robot arms was handled at spawn time instead of being
    part of a clean processed asset;
  - the peg/hole falling tempted us into patching object positions rather than
    fixing the asset.

Why that was premature:

- UWLab's task config does not build the environment support geometry directly
  in Python.
- It spawns an asset as a `RigidObjectCfg`, with kinematic rigid-body properties,
  and relies on the asset to carry the useful visual/collision structure.
- Therefore the correct next step is to process the whole STEP assembly into a
  UWLab-style asset, not to keep patching the raw CAD preview or the objects
  around it.

Correction applied:

- Backed out the lab camera-align object pinning.
- Removed the temporary `LAB_PEG_*` object placement constants.
- The peg/hole falling is now treated as evidence that the raw STEP-converted
  USD is not yet the right scene asset.

Updated asset direction:

- Build a processed USD for the whole lab Vention STEP assembly before treating
  it as the `table` asset in the task scene.
- Follow the UWLab-style structure:

  ```text
  /lab_vention
    /visuals
    /collisions
  ```

- The full Vention frame should be represented, not only the tabletop:
  - tabletop / HDPE panel
  - horizontal rails
  - vertical frame members
  - mounting plates / brackets
  - wheels / casters, if useful for scene contact or visual grounding

- The CAD robot arms exported inside the Vention STEP can be removed from the
  processed lab Vention asset. We will spawn the real simulated UWLab UR5e
  separately at the measured right-arm mount pose.

- Collision should be authored for the non-robot Vention assembly. The CAD arms
  should not become colliders because they would duplicate and interfere with the
  simulated robot articulation.

Practical next step:

- Reconvert or de-instance the STEP so the assembly has explicit mesh prims.
- Then create a processed `lab_vention.usd` with whole-assembly visuals and
  collision geometry, and update the lab scene to spawn that asset as the
  kinematic `table` entity.

### Processed Lab Vention Asset - First Pass

We proceeded with the corrected STEP asset pipeline carefully:

1. Reconverted the Vention v60 STEP with instancing disabled.

   Source STEP:

   ```text
   /data/kanth042/downloads/VentionAssembly_511882_v60.STEP
   ```

   Explicit raw USD output:

   ```text
   /data/kanth042/converted_assets/lab_vention_asset_v60/lab_vention_noinst_raw.usd
   ```

   Command:

   ```bash
   CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 \
   ./uwlab.sh -p scripts_v2/tools/conversions/convert_step_to_usd.py \
     --headless \
     --input /data/kanth042/downloads/VentionAssembly_511882_v60.STEP \
     --output /data/kanth042/converted_assets/lab_vention_asset_v60/lab_vention_noinst_raw.usd \
     --no-instancing \
     --tess-lod 1
   ```

   HOOPS reported:

   ```text
   Total Meshes in USD = 95
   Total Triangles in USD = 327784
   ```

   Static USD inspection confirmed:

   ```text
   Mesh: 95
   Xform: 139
   instance: 0
   ```

2. Added a repeatable processor script:

   ```text
   scripts_v2/tools/conversions/build_lab_vention_asset.py
   ```

   Purpose:

   - consume the de-instanced raw USD;
   - create a UWLab-style processed asset:

     ```text
     /lab_vention
       /visuals
       /collisions
     ```

   - remove the CAD-exported UR5e arms from both visuals and collisions;
   - make `/collisions` invisible;
   - apply `PhysicsRigidBodyAPI` and `PhysicsMassAPI` to `/lab_vention`;
   - apply `PhysicsCollisionAPI` and `PhysicsMeshCollisionAPI` to every collision
     mesh;
   - use `convexDecomposition` as the first-pass collision approximation,
     matching the general UWLab mesh-conversion pattern.

3. Built the processed asset.

   Output:

   ```text
   /data/kanth042/converted_assets/lab_vention_asset_v60/lab_vention.usd
   ```

   Command:

   ```bash
   /data/kanth042/envs/uwlab-isaac51/bin/python \
     scripts_v2/tools/conversions/build_lab_vention_asset.py \
     --input /data/kanth042/converted_assets/lab_vention_asset_v60/lab_vention_noinst_raw.usd \
     --output /data/kanth042/converted_assets/lab_vention_asset_v60/lab_vention.usd
   ```

   Processor output:

   ```text
   Root prim: /lab_vention
   Removed CAD arm groups: 4
     - /lab_vention/visuals/lab_vention_noinst_raw/tn__0_
     - /lab_vention/visuals/lab_vention_noinst_raw/tn__0_1_
     - /lab_vention/collisions/lab_vention_noinst_raw/tn__0_
     - /lab_vention/collisions/lab_vention_noinst_raw/tn__0_1_
   Visual meshes: 61
   Collision meshes: 61
   Collision approximation: convexDecomposition
   ```

4. Static verification of the processed USD:

   ```text
   default prim: /lab_vention
   /lab_vention schemas: PhysicsRigidBodyAPI, PhysicsMassAPI
   /lab_vention/collisions visibility: invisible
   visual meshes: 61
   collision meshes: 61
   collision meshes with PhysicsCollisionAPI: 61
   collision meshes with PhysicsMeshCollisionAPI: 61
   CAD arm prims: removed
   ```

5. Updated lab scene/viewer wiring to use the processed asset:

   - `LAB_VENTION_USD_PATH` now points to:

     ```text
     /data/kanth042/converted_assets/lab_vention_asset_v60/lab_vention.usd
     ```

   - `LabRightArmCameraAlignSceneCfg.table` now uses `RigidObjectCfg`, matching
     the UWLab pattern, rather than `AssetBaseCfg`.
   - Removed the spawn-time CAD-arm hiding hook from `camera_align_cfg.py`
     because the processed asset no longer contains the CAD arms.
   - The lightweight viewer now also uses the processed asset; `--show-cad-arms`
     is left as a deprecated no-op because the arms are removed from the asset.

Validation so far:

- `py_compile` passed for:
  - `scripts_v2/tools/conversions/build_lab_vention_asset.py`
  - `scripts_v2/tools/view_ur5e_camera_scene.py`
  - `camera_align_cfg.py`
  - `lab_layout_cfg.py`
- A headless AppLauncher-backed config import succeeded for
  `LabRightArmCameraAlignEnvCfg`.
- A first headless env reset without `--enable_cameras` reached scene creation
  but failed at camera initialization, which was expected because camera sensors
  require the AppLauncher camera flag.
- Rerunning the same env reset with `--enable_cameras` completed with process
  exit code `0`.
- Warnings observed during reset were the existing robot/gripper/Isaac warnings;
  no Vention table collision-schema failure was observed.

Collision mesh status:

- The `/collisions` meshes are not manually simplified proxy geometry yet.
- They are copied from the de-instanced non-robot CAD visual meshes and marked
  with `PhysicsMeshCollisionAPI`.
- The current approximation is `convexDecomposition`, so PhysX will cook a
  collision approximation at runtime, but the USD still contains one collision
  mesh per retained visual mesh.
- If spawn time, contact quality, or GPU memory becomes a problem, the next
  refinement should be to replace some collision copies with hand-authored simple
  proxy geometry for the tabletop, rails, posts, plates, and wheel contact areas.

### Correction: Match UWLab Collision Simplification Exactly

The first processed lab Vention asset still did not follow UWLab closely enough.

What I checked:

- The Python task config only spawns `pat_vention.usd` as a kinematic
  `RigidObjectCfg`.
- The real collision simplification is in the referenced asset file:

  ```text
  /data/kanth042/tmp/Props/instanceable_meshes.usd
  ```

- `pat_vention.usd` references this file from:

  ```text
  /vention_pat/visuals
  /vention_pat/collisions
  ```

UWLab `pat_vention` collision structure:

```text
visual meshes: 2
collision Cube prims: 7
PhysicsCollisionAPI prims: 7
PhysicsMeshCollisionAPI prims: 0
```

The collision prims are invisible `Cube` prims, not copied CAD meshes:

```text
/vention_pat/collisions/mesh_0 Cube scale=(1.364823, 1.086514, 0.066000)
/vention_pat/collisions/mesh_1 Cube scale=(0.040000, 0.040000, 2.000000)
/vention_pat/collisions/mesh_2 Cube scale=(0.040000, 0.040000, 2.000000)
/vention_pat/collisions/mesh_3 Cube scale=(0.040000, 0.040000, 2.000000)
/vention_pat/collisions/mesh_4 Cube scale=(0.040000, 0.040000, 2.000000)
/vention_pat/collisions/mesh_5 Cube scale=(1.364823, 1.086514, 0.038081)
/vention_pat/collisions/mesh_6 Cube scale=(1.364823, 1.086514, 0.046382)
```

So the earlier `convexDecomposition` / `PhysicsMeshCollisionAPI` version was
still wrong for this Vention table migration.

Correction applied:

- Updated `scripts_v2/tools/conversions/build_lab_vention_asset.py`.
- The processor now:
  - copies the de-instanced CAD only into `/lab_vention/visuals`;
  - removes the CAD arm groups from visuals;
  - creates `/lab_vention/collisions` from invisible `Cube` prims;
  - applies `PhysicsCollisionAPI` to the cube prims;
  - does not author copied mesh collisions;
  - does not author `PhysicsMeshCollisionAPI`;
  - does not use `convexDecomposition`.

Rebuilt lab asset:

```text
/data/kanth042/converted_assets/lab_vention_asset_v60/lab_vention.usd
```

Processor output after correction:

```text
Root prim: /lab_vention
Removed CAD arm groups: 2
  - /lab_vention/visuals/lab_vention_noinst_raw/tn__0_
  - /lab_vention/visuals/lab_vention_noinst_raw/tn__0_1_
Visual meshes: 61
Collision cube proxies: 33
```

Static verification after correction:

```text
default prim: /lab_vention
Mesh prims: 61
Cube prims: 33
PhysicsCollisionAPI prims: 33
PhysicsMeshCollisionAPI prims: 0
PhysicsRigidBodyAPI prims: 1
PhysicsMassAPI prims: 1
```

The number of collision cubes is not the same as `pat_vention` because the lab
assembly contains more retained top-level Vention components after removing the
CAD arms. But the simplification method now matches UWLab: invisible cube
colliders, not CAD mesh colliders.

### UWLab Collision Asset Audit Scope

Clarification after the question "did we only check `pat_vention.usd` or
everything?":

- I have not audited every asset in the whole UWLab repository.
- I did check the scene-relevant OmniReset assets involved in this migration:
  - workspace support table: `Props/Mounts/UWPatVention/pat_vention.usd`
  - its referenced mesh/collision file: `Props/Mounts/UWPatVention/Props/instanceable_meshes.usd`
  - robot support plate: `Props/Mounts/UWPatVention2/Ur5MetalSupport/ur5plate.usd`
  - its referenced mesh/collision file:
    `Props/Mounts/UWPatVention2/Ur5MetalSupport/Props/instanceable_meshes.usd`
  - insertive object: `Props/Custom/Peg/peg.usd`
  - receptive object: `Props/Custom/PegHole/peg_hole.usd`

Findings:

```text
pat_vention referenced asset:
  visual Mesh prims: 2
  collision Cube prims: 7
  PhysicsCollisionAPI prims: 7
  PhysicsMeshCollisionAPI prims: 0

ur5plate referenced asset:
  visual Mesh prims: 1
  collision Cube prims: 1
  PhysicsCollisionAPI prims: 1
  PhysicsMeshCollisionAPI prims: 0

peg:
  visual Mesh prims: 1
  collision Mesh prims: 1
  PhysicsCollisionAPI prims: 1
  PhysicsMeshCollisionAPI prims: 1

peg_hole:
  visual Mesh prims: 1
  collision Mesh prims: 1
  PhysicsCollisionAPI prims: 1
  PhysicsMeshCollisionAPI prims: 1
```

Interpretation:

- UWLab support/environment geometry uses simplified invisible `Cube` collision
  proxies.
- UWLab small manipulated objects use mesh collision.
- Therefore, for our lab Vention assembly, following UWLab means the support
  frame/table asset should use invisible cube proxies, while peg/peg-hole should
  remain as their existing mesh-collision object assets.

Next validation step:

- Launch the lightweight lab-right-arm viewer using the processed asset and
  inspect:
  - Vention frame visuals look correct;
  - CAD arms are gone;
  - simulated UR5e still aligns with the right-arm station;
  - wheels remain on the ground;
  - no obvious collision-schema warnings appear on spawn.

### Relevant Tabletop Scope Clarification

For this migration, ignore the FurnitureBench `SquareTableTop` asset. It belongs
to a separate object/receptive-object workflow and is not the pattern we are
trying to reproduce for the lab support frame.

The relevant UWLab pattern is only:

```text
scene.table -> Props/Mounts/UWPatVention/pat_vention.usd
tabletop/contact-looking mesh -> /vention_pat/visuals/vention_mat
```

Therefore the lab migration should compare against that structure:

```text
scene.table -> /data/kanth042/converted_assets/lab_vention_asset_v60/lab_vention.usd
lab tabletop/contact-looking mesh -> tabletop panel mesh inside lab_vention.usd
```

Current state: we have not split the HDPE/top panel into a separate USD asset.
It remains a prim inside the processed lab Vention USD, matching the relevant
`pat_vention.usd` pattern.

### Lab Tabletop Mesh Naming Cleanup

Next cleanup: the processed lab Vention asset originally exposed the HDPE/top
panel only through the raw CAD-generated name:

```text
/lab_vention/visuals/lab_vention_noinst_raw/tn__importedPANEL72E663E8E8F2_1147116_om0E/Mesh
```

That was technically correct but brittle. UWLab's old table asset exposes the
surface through a stable mesh path:

```text
/vention_pat/visuals/vention_mat
```

So the lab asset builder now promotes the panel mesh to the same style of stable
path, without creating a separate tabletop USD:

```text
/lab_vention/visuals/vention_mat
```

The duplicate raw CAD panel visual is removed from the processed asset so the
panel is not drawn twice.

Regenerated asset:

```text
/data/kanth042/converted_assets/lab_vention_asset_v60/lab_vention.usd
```

Validation after regeneration:

```text
/lab_vention/visuals/vention_mat
  valid: true
  type: Mesh
  APIs: MaterialBindingAPI

raw CAD panel path:
  /lab_vention/visuals/lab_vention_noinst_raw/tn__importedPANEL72E663E8E8F2_1147116_om0E
  valid: false

visual Mesh prims: 61
collision Cube prims: 33
PhysicsCollisionAPI prims: 33
PhysicsMeshCollisionAPI prims: 0
```

The lab layout constant now points to:

```text
LAB_TABLETOP_PANEL_REL_PATH = "visuals/vention_mat"
```

Headless environment reset was rerun after this rebuild and completed with:

```text
SMOKE_RESET_OK
```

The expected camera observations were present:

```text
front_rgb: (240, 320, 3)
side_rgb:  (240, 320, 3)
wrist_rgb: (240, 320, 3)
```

### Workspace Reachability Question

The lab tabletop/workspace panel size is:

```text
1.485 m x 0.720 m x 0.00635 m
```

From the processed lab Vention USD plus the current lab scene transform, the
tabletop world AABB is:

```text
x: -0.834840 to  0.650160
y: -0.432305 to  0.287695
z:  0.836000 to  0.842350
```

The right-arm robot root pose is:

```text
pos: (0.177660, 0.377695, 1.466000)
rot: (0.7071068, 0.0, 0.7071068, 0.0)
```

The robot root is about `0.62365 m` above the tabletop top surface. Distances
from the robot root to tabletop top corners range from about `0.788 m` at the
nearest corner to about `1.439 m` at the farthest corner. Therefore the full
tabletop must not be used as an object spawn region for the right arm.

Very rough first-pass radial estimate, using only straight-line distance from
the robot root to points `5 cm` above the tabletop:

```text
total tabletop area: 1.069 m^2

within 0.75 m radius: ~26% / 0.280 m^2
within 0.80 m radius: ~35% / 0.378 m^2
within 0.85 m radius: ~44% / 0.474 m^2
within 0.90 m radius: ~54% / 0.572 m^2
within 0.95 m radius: ~63% / 0.671 m^2
within 1.00 m radius: ~72% / 0.772 m^2
```

This is only a geometry sanity check. It does not include IK feasibility, joint
limits, gripper orientation, self-collision, table/frame collision, or the
object-specific grasp offsets. It should not become the final reset logic.

UWLab's current handling:

- The old reset-state configs do not compute a full tabletop reachability mask.
- They hard-code object pose ranges relative to `ur5_metal_support`.
- Example old receptive-object reset range:

  ```text
  x: 0.3 to 0.55
  y: -0.1 to 0.3
  z: 0.0
  offset_asset_cfg: ur5_metal_support
  ```

- Example old insertive-object reset range:

  ```text
  x: 0.3 to 0.55
  y: -0.1 to 0.5
  z: 0.0 to 0.3
  offset_asset_cfg: ur5_metal_support
  ```

- For training/eval, UWLab mostly uses reset-state datasets through
  `MultiResetManager`, so the policy sees poses from pre-generated reset states
  rather than sampling the whole table online.

For our lab migration, we should not use the full HDPE panel as the reset
domain. The correct next step is to build a lab-specific tabletop reachability
probe:

1. Sample a grid over `/lab_vention/visuals/vention_mat`.
2. For each point, test one or more wrist/gripper approach poses using the same
   UR5e/Robotiq kinematics and IK conventions as UWLab.
3. Reject points that fail IK, violate joint margins, collide with the Vention
   frame/table, or require unusable gripper orientation.
4. Save the accepted region as a mask/polygon.
5. Make reset sampling draw from that region, not from the old
   `ur5_metal_support` offset box.

### Lab Tabletop Reachability Probe - First Tool

Added a first offline reachability analysis tool:

```text
scripts_v2/tools/analyze_lab_tabletop_reachability.py
```

What it does:

- reads the processed lab Vention asset;
- targets the stable tabletop mesh:

  ```text
  /lab_vention/visuals/vention_mat
  ```

- samples a grid over the tabletop;
- transforms sampled points into the selected right-arm UR5e base frame;
- loads the calibrated UR5e kinematics from the same robot `metadata.yaml`
  used by UWLab;
- solves batched IK from multiple seed configurations;
- writes a JSON summary and CSV grid.

This is still a kinematic probe, not final reset logic. It does not yet run
PhysX collision checks against the Vention frame/table, self-collision, or
object-specific grasp/assembly validity.

Command used for the dense first pass:

```bash
CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 ./uwlab.sh -p \
  scripts_v2/tools/analyze_lab_tabletop_reachability.py \
  --nx 61 --ny 31 --probe both --yaw-count 7 --iterations 90
```

Output files:

```text
/data/kanth042/converted_assets/lab_vention_asset_v60/reachability/right_arm_tabletop_reachability.json
/data/kanth042/converted_assets/lab_vention_asset_v60/reachability/right_arm_tabletop_reachability.csv
```

Dense grid result:

```text
grid: 61 x 31 = 1891 tabletop points

wrist_position probe:
  accepted: 1152 / 1891 = 0.609
  approximate accepted area: 0.651 m^2
  accepted bbox:
    x: -0.4636 to 0.6502
    y: -0.4323 to 0.2877
  largest all-reachable rectangle:
    x: -0.1171 to 0.6502
    y: -0.2883 to 0.2877
    area: ~0.452 m^2

top_down_gripper probe:
  accepted: 1338 / 1891 = 0.708
  approximate accepted area: 0.757 m^2
  accepted bbox:
    x: -0.5626 to 0.6502
    y: -0.4323 to 0.2877
  largest all-reachable rectangle:
    x: -0.1171 to 0.6502
    y: -0.4083 to 0.2877
    area: ~0.543 m^2
```

Important interpretation:

- The accepted bbox is not automatically safe for uniform sampling because it
  can include holes/unreachable points.
- The `largest all-reachable rectangle` is the safer first uniform reset range.
- For the first minimal lab reset config, use the top-down-gripper rectangle as
  the conservative candidate workspace:

  ```text
  x: -0.1171 to 0.6502
  y: -0.4083 to 0.2877
  z_top: 0.84235
  ```

Next migration step:

- Add a lab reset/event config that samples object XY from either:
  - the all-reachable rectangle above, for minimal code changes; or
  - the CSV/mask accepted region, for better table coverage.
- Then run an Isaac/PhysX validation pass to reject points that collide with
  the Vention frame, robot, gripper, or objects.

### Cleanup After Reachability Pass

Removed generated leftovers that are no longer part of the current migration
path:

- Python `__pycache__` and `.pyc` files under the repo.
- Old raw preview USD folders:

  ```text
  /data/kanth042/converted_assets/lab_vention_preview
  /data/kanth042/converted_assets/lab_vention_preview_v60
  ```

- Temporary inspection copies under `/data/kanth042/tmp`:

  ```text
  Props/
  pat_vention.usd
  square_table_top.usd
  uwlab-assets-inspect/
  uwlab_object_usds/
  metadata.yaml
  ```

- Current IsaacLab/Kit logs and the old NvStreamer `.etli` file found under
  the Isaac-Sim Streaming data directory.
- Removed `scripts_v2/tools/view_usd_asset.py`; we are keeping one viewer path:

  ```text
  scripts_v2/tools/view_ur5e_camera_scene.py
  ```

Kept the files still needed for reproducibility:

```text
/data/kanth042/converted_assets/lab_vention_asset_v60/lab_vention_noinst_raw.usd
/data/kanth042/converted_assets/lab_vention_asset_v60/lab_vention.usd
/data/kanth042/converted_assets/lab_vention_asset_v60/reachability/
scripts_v2/tools/conversions/convert_step_to_usd.py
scripts_v2/tools/conversions/build_lab_vention_asset.py
scripts_v2/tools/analyze_lab_tabletop_reachability.py
```

### Lab Right-Arm Reset-State Variant

Added a lab-specific reset-state variant instead of changing the original
UWLab reset configs.

New registered task:

```text
OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-LabRightArm-v0
```

New/updated config pieces:

```text
LabRightArmResetStatesSceneCfg
LabRightArmObjectAnywhereEEAnywhereEventCfg
LabRightArmObjectAnywhereEEAnywhereResetStatesCfg
```

What changed for the lab variant:

- `scene.table` now spawns the processed lab Vention asset:

  ```text
  /data/kanth042/converted_assets/lab_vention_asset_v60/lab_vention.usd
  ```

- The robot root is fixed at the selected right-arm station:

  ```text
  pos: (0.177660, 0.377695, 1.466000)
  rot: (0.7071068, 0.0, 0.7071068, 0.0)
  ```

- `ur5_metal_support` remains only as a hidden logical scene entity. Objects no
  longer use it as their offset frame in this lab reset variant.
- Object XY sampling uses the shrunken reachable rectangle:

  ```text
  x: -0.100 to 0.625
  y: -0.385 to 0.265
  ```

- Receptive object z is the tabletop top:

  ```text
  z: 0.84235
  ```

- Insertive object z samples from tabletop top to 30 cm above it:

  ```text
  z: 0.84235 to 1.14235
  ```

Smoke test command used the config object explicitly with Gym and completed:

```text
SMOKE_RESET_OK
```

One sampled reset from the smoke test:

```text
table_pos     [1.793445, 0.340075, -0.030851]
robot_pos     [0.177660, 0.377695, 1.466000]
receptive_pos [0.430006, 0.237381, 0.857513]
insertive_pos [0.390627, 0.196426, 0.947239]
```

The object XY values are inside the reachable rectangle. The object z values
include each asset's bottom-offset correction, so the root z is slightly above
the raw tabletop z.

After the smoke test, generated Python caches and fresh Isaac logs were removed
again.

### Lab Right-Arm Reset Validation

Added a dedicated validation utility:

```text
scripts_v2/tools/validate_lab_right_arm_resets.py
```

Purpose:

- sample the new lab-right-arm reset environment repeatedly;
- step physics briefly after each reset;
- record whether the receptive/insertive objects remain on or above the lab
  tabletop plane;
- report object drift separately from real support/tabletop failures.

Command used:

```bash
CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 ./uwlab.sh -p \
  scripts_v2/tools/validate_lab_right_arm_resets.py \
  --headless --num-envs 4 --num-resets 3 --settle-steps 8
```

Output files:

```text
/data/kanth042/converted_assets/lab_vention_asset_v60/validation/lab_right_arm_reset_validation.csv
/data/kanth042/converted_assets/lab_vention_asset_v60/validation/lab_right_arm_reset_validation_summary.json
```

Important correction during this step:

- The first validation version treated receptive-object XY drift as a support
  failure.
- That was too strict for `ObjectAnywhereEEAnywhere`: the falling insertive
  object can bump the receptive object during settling, and UWLab's reset-state
  generation is allowed to record the final stable state.
- The script now reports receptive drift as telemetry unless explicitly run
  with `--fail-on-receptive-drift`.
- The script only treats objects falling below the tabletop, non-finite poses,
  or escaping far outside the sampled workspace as support failures.

The validation also exposed a lab-specific success-condition issue:

- The original reset-state success check only required objects to stay above a
  near-ground threshold.
- For the lab frame, an insertive object that falls off the tabletop can still
  be above world ground.
- The lab-only reset config now raises the success z threshold to:

  ```text
  LAB_TABLETOP_TOP_Z - 0.03 = 0.81235 m
  ```

- This change is only in:

  ```text
  LabRightArmObjectAnywhereEEAnywhereResetStatesCfg
  ```

- The original UWLab `ObjectAnywhereEEAnywhereResetStatesCfg` remains unchanged.

Latest small validation run:

```text
samples: 12
support failures: 0 / 12 = 0.00%
receptive drift events: 5 / 12 = 41.67%
min receptive end z: 0.857513
min insertive end z: 0.857350
max receptive xy drift: 0.350764
```

Interpretation:

- The processed lab Vention tabletop collision is behaving correctly in this
  small smoke validation.
- Some object-object interaction during settling is expected and should be
  handled by the reset-state success/recording pipeline.
- A larger validation run is still needed before generating real lab reset
  datasets.

Larger validation run:

```bash
CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 ./uwlab.sh -p \
  scripts_v2/tools/validate_lab_right_arm_resets.py \
  --headless --num-envs 16 --num-resets 50 --settle-steps 15
```

Result:

```text
samples: 800
support failures: 33 / 800 = 4.125%
receptive drift events: 276 / 800 = 34.5%
min receptive end z: 0.857513
min insertive end z: 0.015000
max receptive xy drift: 0.767891
```

Interpretation after the larger run:

- The receptive object consistently stayed on the tabletop in this sample.
- The support failures were insertive-object outcomes after settling.
- This is not evidence that the tabletop collision is missing; the failed
  examples show the insertive object being rejected because it ended below the
  lab tabletop threshold.
- The lab-only success threshold added above is therefore important: it prevents
  off-table insertive-object outcomes from being recorded as valid reset states.
- Next practical step is to run a small `record_reset_states.py` pass with the
  lab task and confirm that successful states are actually exported.

### Lab Reset-State Recorder Smoke Test

Ran UWLab's actual reset-state recorder against the lab task:

```bash
CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 ./uwlab.sh -p \
  scripts_v2/tools/record_reset_states.py \
  --headless \
  --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-LabRightArm-v0 \
  --num_envs 16 \
  --num_reset_states 20 \
  --reset_type ObjectAnywhereEEAnywhere \
  --dataset_dir /data/kanth042/converted_assets/lab_vention_asset_v60/reset_state_smoke_dataset
```

Recorder output:

```text
/data/kanth042/converted_assets/lab_vention_asset_v60/reset_state_smoke_dataset/Resets/Peg__PegHole/resets_ObjectAnywhereEEAnywhere.pt
```

Result:

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

Interpretation:

- The lab-specific task registration works with UWLab's real reset recorder.
- The exported states are above the lab tabletop threshold.
- The object XY values remain inside the conservative lab reset rectangle.
- The robot/table roots stay fixed across the exported states.
- This confirms the lab layout and tabletop success threshold are wired into
  the existing reset-state generation flow.

### Downstream Compatibility Deep Dive

Created a standalone deep-dive note:

```text
lab_vention_downstream_compatibility_deepdive.md
```

Purpose:

- list exactly what changed for the lab Vention migration;
- separate software-interface compatibility from physical/data-distribution
  compatibility;
- explain why original UWLab envs remain unchanged while lab variants are added;
- document which downstream stages are already compatible and which still need
  lab-specific configs/datasets before paper replication.

### Coordinate/Origin Clarification And D405 Guide Cleanup

Updated the lab Vention downstream compatibility note to make the coordinate
convention explicit:

- `+Z` is up.
- The horizontal workspace plane is `X/Y`.
- `X` is the long tabletop direction after the `LAB_VENTION_*` transform.
- `Y` is the short tabletop direction after the `LAB_VENTION_*` transform.
- The world origin is the Isaac scene origin, not the robot base, Vention CAD
  origin, or tabletop center.
- In multi-env runs, actual placement is:

  ```text
  world pose in env_i = env.scene.env_origins[i] + configured local/root pose
  ```

Also rewrote the D405 migration guide so today's hardware work is clear. The
guide now starts with the values we need from the real D405 and Spark
calibration:

- D405 serial number and selected Spark camera key.
- Color profile used for calibration and policy sim.
- `fx`, `fy`, `cx`, `cy`, image size, distortion coefficients.
- Spark `flange_from_camera` hand-eye result.
- Calibration quality numbers.
- Final UWLab local camera offset:

  ```text
  robotiq_base_link -> rgb_wrist_camera
  ```

The guide now explicitly separates the lab Vention scene-layout migration from
the D405 wrist-camera migration. Scene layout values stay in the lab layout and
task configs; D405 values stay in the robot/camera configs.
