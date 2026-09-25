# Thunder calibration on the upstream stock robot

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

Set `UWLAB_THUNDER_CALIBRATION_ASSET_DIR` to the generated asset directory.
Set `UWLAB_THUNDER_CALIBRATION_DATASET_DIR` to a separate directory for resets
generated with this calibration; its default is
`Datasets/OmniResetCubeEasyThunderCalibration`, relative to the working directory.
The task checks the calibration identity when instantiated. Importing or using
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
Calls without a path keep the original upstream calibration. These task
configurations change only the robot asset and reset dataset paths relative to
their upstream easy-task parents. New compatible reset banks are required
before training; the calibration validation did not generate them.

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

## Wrist-limit discrepancy

Native readback of all three wrist joints is still **[-360, 360] degrees**.
The paper's [Appendix A.3.1](https://arxiv.org/html/2603.15789v3#A3.SS1)
describes **[-180, 180] degrees** in simulation, but that restriction was not
found in the selected released asset or active task configuration. The section
does not identify an implementing file or a training-stage switch. Adopting
the narrower limits is a separate task change to decide before generating the
next bank; this calibration commit retains the upstream limits.
