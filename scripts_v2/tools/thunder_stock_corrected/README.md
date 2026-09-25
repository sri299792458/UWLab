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

## Fixed-gain chirp comparison

The subsequent `replay_chirp.py` experiment replays the existing eight-second
0.1–3 Hz half-amplitude hardware chirp with the already fitted armature/friction
parameters. It uses two environments per rate: UWLab 200/3 stiffness with
damping 84.853/3.464, and Thunder's recorded 1000/50 stiffness with damping
63.246/14.142. The controller implementation is the existing task action;
gains are installed per environment. Nothing is fitted or selected by a sweep.

Both rates start from exactly equal saved joint positions/velocities, body
masses/inertias/COMs and fitted arm parameters. The 500 Hz recording has one
target per servo step; the 120 Hz replay interpolates position and uses SLERP
for orientation at corresponding times. This preserves the eight-second
trajectory and frequency; it does not simulate the training policy's 10 Hz
target-hold behavior. Comparison remains on the original fit's nominal 2 ms
command-index timeline; original timestamp anomalies are not repaired.
The fitted 6 ms delay is three steps at 500 Hz and nearest one step (8.333 ms)
at 120 Hz. The rate comparison includes that quantization. The existing fit's
inactive 1000 rad/s simulation cap is retained to expose excessive speed.

Completed results (joint RMS pooled over time and six arm joints):

| Gains | 120-vs-500 Hz angle difference | Speed difference | 120 Hz error vs recorded hardware | 500 Hz error vs recorded hardware |
|---|---:|---:|---:|---:|
| UWLab | 0.00214 degrees | 0.0497 degrees/s | 2.73887 degrees | 2.73949 degrees |
| Thunder | 0.07982 degrees | 1.31724 degrees/s | 0.50562 degrees | 0.50391 degrees |

All four traces remain finite without reaching the torque clamps. UWLab gains
produce almost no motion in five joints under these fitted dynamics; only the
shoulder-lift joint moves appreciably. The small rate difference in that row
therefore does not imply good motion tracking. UWLab Stage 1 normally omits
this fitted friction, so this is not a rejection of its initial-training recipe.
Thunder gains preserve similar motion at 120 and 500 Hz on this trajectory.

Hardware comparison is descriptive: the physical hand had UMI fingers, and
this is the same recording used to fit dynamics. It is not an independent
stock-hand validation, a contact/grasp test, or evidence of PPO learning.
`analyze_chirp.py` independently checks paired inputs, computes the rate
difference at common times, and saves joint-position/velocity plots. Reports,
exact launch commands and hashes are in the external R111 folder. The first
attempt stopped before replay on an input float64/float32 mismatch; corrected
attempt 2 completed all four conditions. Both attempts are preserved.
