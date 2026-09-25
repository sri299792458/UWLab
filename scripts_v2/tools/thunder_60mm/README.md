# Thunder D405 with stock fingers and 60 mm cubes

This stage inherits the centered D405 task and replaces both 40 mm cubes with
the rounded AprilCube60 R3 assets (60 mm sides, 3 mm edge radius). The approved
cube XY positions, raised IK seed, Thunder calibration, mount/table, stock hand,
controller, rewards and PPO settings are inherited.

The USDs author a 100 g mass and diagonal inertia of 0.00006 kg m². Clear the
inherited cube mass overrides in **every stage**: the old 40 mm asset lacks
MassAPI, but the 60 mm asset supports it, so leaving those overrides would make
the upper cube 1 g. The 100 g baseline is a model parameter, not a hardware
measurement. Training retains the upstream upper-cube absolute 20–200 g
randomization and lower-cube 0.5–1.5 scaling (now 50–150 g).

Grasp sampling uses the upstream standalone stock Robotiq hand and the cube's
authored friction (static 1.0, dynamic 0.9). Partial assemblies retain upstream
zero friction; full reset generation retains 0.3/0.2. Training retains the
upstream material randomization. The 30 mm metadata bottom offset moves the
resting cube centre to Z=0.87235 m on the existing table. All other sampling
ranges, native acceptance conditions and solver settings are inherited.

Required environment variables:

- `UWLAB_THUNDER_CALIBRATION_ASSET_DIR`: calibrated wrist-limited D415 asset,
  required by the inherited configuration.
- `UWLAB_THUNDER_D405_ASSET_DIR`: reviewed D405 R2 stock-hand asset.
- `UWLAB_THUNDER_60MM_ASSET_DIR`: directory containing `InsertiveAprilCube60`
  and `ReceptiveAprilCube60`, each with `aprilcube_60.usd` and `metadata.yaml`.
- `UWLAB_THUNDER_60MM_DATASET_DIR`: separate dataset output/input directory.

The USD and metadata hashes are pinned in `thunder_60mm_cfg.py`.

Generate fresh data with the existing recorders. Task prefix:
`OmniReset-UR5eRobotiq2f85-CubeEasyThunderD40560-`.

1. `GraspSampling-v0` with `scripts_v2/tools/record_grasps.py`.
2. `PartialAssemblies-v0` with `scripts_v2/tools/record_partial_assemblies.py`.
3. Each of the four reset-family task suffixes with
   `scripts_v2/tools/record_reset_states.py`. Table-grasp generation depends on
   the completed free-hand bank, as in the upstream task.

Output roles are `Grasps/InsertiveAprilCube60` and
`Resets/InsertiveAprilCube60__ReceptiveAprilCube60`. Old 40 mm or historical
60 mm banks are not inputs to this stage. The training task is
`OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-CubeEasyThunderD40560-v0`.
