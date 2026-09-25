# Thunder calibration on the upstream stock robot

The simulation setup applies Thunder arm calibration and then the paper's
±180° wrist limits as separate changes. Commit `8b15b3d` is the calibration-only
reference. Real deployment must use compatible initialization and angle handling;
that follow-up does not block applying these limits in simulation.

`build_asset.py` copies the pinned upstream D415 UR5e/Robotiq USD and changes
only Thunder's six calibrated arm joint frames and the rigid-body zero poses
that depend on them. It retains the upstream gripper, camera, collision shapes,
mass and inertia, materials, mimic relationships, and joint limits. The adjacent
`metadata.yaml` keeps every source field and adds the calibrated joint transforms,
calibration provenance, and an explicit note that inherited UWLab dynamics values
have not been identified on Thunder.

The calibration geometry helpers were ported from the local tested
`build_thunder_d405_robot_asset.py`; the procedure mirrors the retained R86
pure-calibration build. The bundled `thunder_kinematics.yaml` comes from the
Thunder calibration in `RPM-lab-UMN/spark-data-collection`, hash
`calib_10185139869934003756`. No D405-specific mount, camera, mass, or gripper
repair is applied.

From the repository root, use the Isaac Sim Python environment:

```bash
/data/kanth042/envs/uwlab-isaac51/bin/python scripts_v2/tools/thunder_calibration/build_asset.py \
  --source-usd /home/kanth042/.cache/uwlab/assets/Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated/ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd \
  --output-dir /data/kanth042/datasets/umi_reset_from_defaults_20260911/105_thunder_calibration_port_20260924/assets/thunder_calibration
```

The output directory must initially be empty. The script checks the pinned
source USD and metadata SHA-256, every prim's attributes and relationships, and the calibrated
zero-pose chain before writing `robot.usd`, `metadata.yaml`, and
`calibration_report.json`. The report includes input/output hashes and every
allowed USD attribute change. The asset is generated outside Git.

The September 24 build passed with zero position and rotation error in the
authored zero pose. `robot.usd` SHA-256 is
`8bfe88e5ceefdbd4524e4426cba93be94629634b642176143636935cee53610a`,
and `metadata.yaml` SHA-256 is
`625bed4ccddbc19392fc7ea317ebd000d2217f3a936f7970bb979a36f7329242`.
Both exactly match the retained R86 pure-calibration outputs.

## Select the calibrated tasks

The calibration-only validation used the asset above with its original wrist
limits. The next wrist-limit task step is described below; use its separate
asset and reset directory when instantiating the current task. Importing or using
the existing upstream tasks does not require these variables.

The training task is
`OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-CubeEasyThunderCalibration-v0`.
Generation tasks use
`OmniReset-UR5eRobotiq2f85-CubeEasyThunderCalibration-<family>-v0`, where
`<family>` is `ObjectAnywhereEEAnywhere`, `ObjectRestingEEGrasped`,
`ObjectAnywhereEEGrasped`, or `ObjectPartiallyAssembledEEGrasped`.

The Cartesian controller passes the selected robot USD path to the analytical
Jacobian, so the controller reads the calibrated joint transforms beside that
asset. The analytical mass-matrix helper accepts the same optional path.
Calls without a path keep the original upstream calibration. The calibration
commit changes only the robot asset and reset dataset paths relative to the
upstream easy-task parents. The wrist-limit follow-up also enables reset checks
for the three wrist joints. New compatible reset banks are required before
training; the calibration validation did not generate them.

## Native simulator validation

On September 24, 2026, an eight-environment Isaac Sim 5.1 probe compared native
wrist poses with the existing independent double-precision Thunder kinematics
check. Across eight joint poses near the default pose, the maximum wrist position
error was `5.82e-7 m` (0.58 micrometres), the maximum rotation-matrix component
error was `1.05e-6`, and the maximum analytical-Jacobian component error was
`1.46e-7`. The controller forwarded the selected asset on all 36 observed calls
and completed three finite policy steps. Upstream Jacobian and analytical
mass-matrix results were bit-identical to parent commit `6335f83`, both with
the implicit default path and the explicit upstream path. Configuration
comparisons covered the training task and all four generation tasks.

The reproducible probe, launch record, saved configurations and result are in
`/data/kanth042/datasets/umi_reset_from_defaults_20260911/105_thunder_calibration_port_20260924/`
(`verify_calibration_port.py`, `native_launch.json`, `native_check/report.json`).
This checks calibration integration; it does not establish learning performance
or validate the later Thunder mount, D405 assembly or custom fingers.

## Wrist-limit step after calibration-only commit `8b15b3d`

Commit `8b15b3d` deliberately retained the upstream **[-360, 360] degree**
wrist limits. For the next experiment, `apply_wrist_limits.py` copies its
calibrated USD to a new empty output directory and changes only the lower and
upper limit attributes of wrist joints 1–3 to **[-180, 180] degrees**:

```bash
/data/kanth042/envs/uwlab-isaac51/bin/python scripts_v2/tools/thunder_calibration/apply_wrist_limits.py \
  --source-dir /data/kanth042/datasets/umi_reset_from_defaults_20260911/105_thunder_calibration_port_20260924/assets/thunder_calibration \
  --output-dir /data/kanth042/datasets/umi_reset_from_defaults_20260911/105_thunder_calibration_port_20260924/assets/thunder_wrist180
```

The generated `robot.usd` SHA-256 is
`b00829fbce65001dc3dc88e6078fc353b3df3946b069efcb276f7d9016f7b817`.
`metadata.yaml` remains byte-identical to the calibration-only asset. The
`wrist_limit_report.json` records the six allowed attribute changes and the
source/output hashes. All other USD attributes and prim relationships are
checked unchanged. The output is outside Git.

Set `UWLAB_THUNDER_CALIBRATION_ASSET_DIR` to the `thunder_wrist180` directory.
The same four generation task IDs and training task ID now require native USD
wrist limits of ±180 degrees. Their default dataset directory is distinct:
`Datasets/OmniResetCubeEasyThunderWrist180`. Each generation task requests a
wrist-limit acceptance check; training requests a wrist-limit check when it
loads reset banks. These checks use the three named wrist joints only and allow
`1e-6 rad` of floating-point roundoff. Invalid angles are rejected without
clamping or wrapping. Fresh full-arm reset banks are required.

`verify_wrist_limits.py` checks the PhysX limits, generates a small set of
accepted reset states, and tests the real success term and training-bank loader
with above-limit, below-limit and non-finite angles on each wrist. Its negative
acceptance tests temporarily change observation buffers and restore them before
recording or advancing physics. They are deliberate validation cases, not
naturally generated failures. This probe does not launch training or generate
a production reset bank.

The September 24 native probe passed: PhysX read back `[-180, 180]` degrees
for all three wrists; 27 successful states were exported after 20 policy steps.
All exported wrist angles were finite and inside their limits. All nine
deliberately invalid cases were rejected by generation acceptance, and all
nine were rejected by the training-bank loader. Valid bank rows loaded without
modification, and existing upstream tasks retained their original behavior.
The result and exact launch are recorded in
`105_thunder_calibration_port_20260924/wrist180_native_check_v2/report.json`
and `wrist180_native_launch_v2.json` under the dataset experiment root above.
The first probe attempt confirmed the limits but failed in its temporary test
hook's reset delegation; the corrected probe passed.
