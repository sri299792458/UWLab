# Corrected stock-finger D405 model

This stage ports the existing documented stock-hand mass model onto the D405
geometry from commit `788af89`, following the rounded 60 mm task `6a71a42`.
It changes all nine hand bodies' mass, center of mass and inertia. The base
includes the stock housing, 16 g printed R2 mount, nominal 60 g D405 camera,
1.4 g camera screws and 3.4 g gripper-mount screws. Moving links use the pinned
PickNik reference. The whole modeled assembly is **1.001664273 kg**;
**0.257662303 kg** is just the eight moving links. This is the previously
documented model, not a new whole-hand weighing.

`mass_properties.json` preserves the body-frame tensors and provenance from
the existing `picknik_inertial_report.json` and `r2_mount_report.json` in
`/data/kanth042/converted_assets/thunder_d405_robot_asset`. Cable/ties, the
separate flange adapter, washers and tape remain documented unmodeled mass.

The standalone sampling hand is extracted with the established
`may_cubes_stock_umi/extract_hand.py` method: preserve the nine hand bodies and
internal joints, express their poses relative to the base, and fix that base
to the world. The full robot's mass properties are preserved exactly. The
sampling configuration clears its inherited 0.5 kg mass override.

Build using the existing Isaac environment:

```bash
/data/kanth042/envs/uwlab-isaac51/bin/python scripts_v2/tools/thunder_stock_corrected/build_asset.py \
  --source-dir /data/kanth042/datasets/umi_reset_from_defaults_20260911/108_thunder_d405_stock_20260925/assets/d405_r2_geometry_only \
  --output-dir /data/kanth042/datasets/umi_reset_from_defaults_20260911/111_thunder_stock_corrected_20260925/assets/mass_only
```

The output must be new. `build_report.json` records the input/output hashes
and verifies the full composed asset differs only in nine MassAPI definitions
(including the schema's unauthored density=0 default). Geometry, physical
joints, wrist limits and calibration are preserved. The robot and standalone
hand mass attributes are compared exactly after serialization.

Set `UWLAB_THUNDER_STOCK_CORRECTED_ASSET_DIR` to the build directory, retaining
the preceding calibration/D405/60 mm environment variables. Select the new
`CubeEasyThunderStockCorrected60` task IDs; the partial-assembly task remains
`CubeEasyThunderD40560-PartialAssemblies` because it contains no robot.

This first commit ports inertials. A separate following commit removes the
four invalid inner mimic declarations. Controller settings remain inherited:
arm Kp translation/rotation 200/3, damping ratios 3/1, hand stiffness/damping
17/5, and training physics/policy 120/10 Hz. The old stages and reset exports
remain independently selectable. No new reset distribution, fitted arm
profile, gain selection, reward change or PPO run is introduced here.
