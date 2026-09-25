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
  --output-dir /data/kanth042/datasets/umi_reset_from_defaults_20260911/111_thunder_stock_corrected_20260925/assets/corrected \
  --remove-inner-mimics
```

The output must be new. `build_report.json` records the input/output hashes
and verifies the inertial port differs only in nine MassAPI definitions
(including the schema's unauthored density=0 default). Geometry, physical
joints, wrist limits and calibration are preserved. The robot and standalone
hand mass attributes are compared exactly after serialization. The subsequent
linkage patch reuses `robotiq_linkage.py` from the established conversion tools:
remove the four left/right inner-knuckle and inner-finger-knuckle mimic APIs
and their properties, retain all physical joints and the right outer-knuckle
mimic. Its joint snapshot verifies every other joint property remains equal.

Set `UWLAB_THUNDER_STOCK_CORRECTED_ASSET_DIR` to the build directory, retaining
the preceding calibration/D405/60 mm asset environment variables. Set
`UWLAB_THUNDER_STOCK_CORRECTED_DATASET_DIR` to this stage's fresh data directory;
the corrected tasks do not inherit the preceding model's reset directory. Select the new
`CubeEasyThunderStockCorrected60` task IDs; the partial-assembly task remains
`CubeEasyThunderD40560-PartialAssemblies` because it contains no robot.

Commit `c77cbbc` ports inertials; the following commit removes the four invalid
inner mimic declarations. Controller settings remain inherited:
arm Kp translation/rotation 200/3, damping ratios 3/1, hand stiffness/damping
17/5, and training physics/policy 120/10 Hz. The old stages and reset exports
remain independently selectable. No new reset distribution, fitted arm
profile, gain selection, reward change or PPO run is introduced here.

## Native validation

`validate_native.py --kind robot|hand --output REPORT --headless --device cuda:0`
instantiates the selected task's robot configuration, with nominal masses and
no startup randomization. Both native readbacks report **1.001664290 kg**;
all nine masses, centers and full body-frame tensors match the documented
values within float precision. Only the intended opposite-jaw mimic remains.
The standalone hand also runs a two-second close and two-second reopen using
the inherited 17/5 actuator settings. These are model-loading/linkage checks,
not a controller-gain recommendation or evidence of successful grasping.

Evidence is saved under
`/data/kanth042/datasets/umi_reset_from_defaults_20260911/111_thunder_stock_corrected_20260925/native/`.
Headless PhysX completed despite the existing graphics-device initialization
messages; no visual rendering was exercised.

## Fresh data required before training

Regenerate grasps with the corrected standalone hand and regenerate all four
full-robot reset families with the corrected robot before training this stage.
The previous hand's grasp acceptance and settled states depend on its old
mass/linkage model. Preserve the approved placement and proposal settings and
keep the previous banks as historical artifacts. The cube-only partial-assembly
inputs can be reused because their geometry and physics are unchanged. A chirp
replay instead starts from the recorded hardware joint state and uses no reset
bank. No generation or training process is started by building these assets.
