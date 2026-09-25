# Thunder D405 R2 stock-hand assembly

This step replaces the D415 wrist-camera assembly on the calibrated,
wrist-limited Thunder robot with the existing R2 D405 printed mount. The task
inherits the centered two-cube patch, measured robot/table pose, stock Robotiq
fingers, 40 mm cubes, reset proposals, controller, rewards, and PPO settings.

Build the asset with the Isaac USD Python environment:

```bash
/data/kanth042/envs/uwlab-isaac51/bin/python scripts_v2/tools/thunder_d405/build_asset.py \
  --source-dir /data/kanth042/datasets/umi_reset_from_defaults_20260911/105_thunder_calibration_port_20260924/assets/thunder_wrist180 \
  --r2-stl /data/kanth042/repos/robotiq-2f85-umi-gripper/artifacts/d405_print_R2_20260905/D405_wrist_mount_R2_PRINT.stl \
  --mechanical-checks /data/kanth042/repos/robotiq-2f85-umi-gripper/artifacts/d405_print_R2_20260905/mechanical_checks.json \
  --output-dir /data/kanth042/datasets/umi_reset_from_defaults_20260911/108_thunder_d405_stock_20260925/assets/d405_r2_geometry_only
```

The output directory must be new and empty. The generated USD lives outside
Git and has SHA-256 `00eb01b7c169b223bfe3e76f8be1285bc008e784db43a4a21b6eef9f5372ee90`.
The metadata remains byte-identical to the D415 source. A second build from
this portable script reproduced the reviewed USD hash exactly. The builder
ports only the material/mesh routines of the previously validated R2 builder,
checks the R2 STL hash, and audits the full USD snapshot: three D415 camera
prims removed, nine R2 camera/mount/material/frame prims added, and all 339
common prims unchanged. The nine added prims match the prior R2 assembly USD.
No UMI finger, mimic, PickNik mass, or other dynamics edits are imported.

Set `UWLAB_THUNDER_CALIBRATION_ASSET_DIR` to the wrist180 source asset directory
for the inherited calibration check, `UWLAB_THUNDER_D405_ASSET_DIR` to the new
R2 asset directory, and `UWLAB_THUNDER_D405_DATASET_DIR` to an independent new
OmniReset dataset directory. The D405 tasks use the suffix `CubeEasyThunderD405`.
The centered D415 reset poses may be reused because arm kinematics, fingers,
cubes, and task coordinates are unchanged. Before training, revalidate every
reused pose under the D405 assembly, especially camera/mount contacts; generate
new banks only for any family that fails. The sampled native reload check alone
does not establish full-bank contact acceptance.

PhysX computes the gripper-base mass/inertia from the changed colliders because
the inherited D415 base has no authored MassAPI. In one-environment native
probes with robot mass randomization disabled, the D415 gripper-base mass is
**1.337838 kg** and D405 R2 is **0.709552 kg** (−0.628287 kg, −47.0%). The
robot collision-shape count changes 16→15, so the policy observation remains
215 wide while the critic observation changes 282→279. This is a coupled
assembly-swap comparison, not an isolated camera-visual effect. The known
D415 `d415_and_cable` safety collider alone contributes 0.5 kg of that
automatic mass (issue 38 verification in the training-regression audit).
This stage keeps upstream automatic inertia behavior; an explicit measured
D405 mass model would be a separate future change.

The R2 nominal optical frame is CAD geometry, not measured hand-eye
calibration. A CPU zero-pose screen found all six stock finger/knuckle collider
vertex sets outside the R2 mount hull (nearest sampled gap 11.69 mm); it does
not establish clearance throughout the full gripper stroke. The native mass
reports and clearance screen are under the external `108_thunder_d405_stock_20260925`
dataset directory.
