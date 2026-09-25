# Thunder lab mount/table easy cube scene

`thunder_lab_cfg.py` adds the measured Thunder base and Vention tabletop to the
calibrated, wrist-limited easy-cube task. It inherits the D415 assembly, stock
Robotiq fingers, 40 mm cubes, control timing/gains, physics, rewards, PPO, and four
reset families. Generation tasks are registered as
`OmniReset-UR5eRobotiq2f85-CubeEasyThunderLab-<family>-v0`; training uses
`OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-CubeEasyThunderLab-v0`.

Set `UWLAB_THUNDER_CALIBRATION_ASSET_DIR` to the verified `thunder_wrist180`
asset directory containing `robot.usd` and `metadata.yaml`. The Vention USD is
read from `${UWLAB_ASSET_ROOT:-/data/kanth042/converted_assets}/lab_vention_asset_v60/lab_vention.usd`.
Set `UWLAB_THUNDER_LAB_DATASET_DIR` to a new OmniReset bank directory; its
independent default is `Datasets/OmniResetCubeEasyThunderLab`. All four banks
must be regenerated under this scene before training because saved robot and
object roots from the prior mount are in different world coordinates.

The easy task retains goal XY `(0.45, 0.15)` and upper-cube start patch
`x=(0.35, 0.37), y=(-0.01, 0.01)`. Both lie inside the measured lab tabletop
footprint `x=[-0.83484, 0.65016], y=[-0.43230, 0.28770]`. The old tabletop is
at Z `-0.013`; the lab tabletop is at `0.84235`, a rise of `0.85535` m.
The cube sampler retains its bottom offset and adds the new tabletop Z to its
original height range, removing the old support offset. Thus the lower 40 mm
cube center moves from `0.007` to `0.86235` m, flush with each tabletop.
The lab CAD contains the mount, so the upstream support is removed.

The free-hand end-effector sampler retains its original world XY/orientation
proposals and table-relative height, raising world Z by `0.85535` m and then
converting to offsets from the new robot root as its event function expects.
Root jitter is recentered on the measured base pose with its original widths:
X/Z +/-10 mm, Y +/-20 mm. The end-effector offset accounts for the original
Y-jitter center of -39 mm. The reset success minimum Z rises by `0.85535` m;
other acceptance thresholds are inherited.

The table remains scene entity `table` under `{ENV_REGEX_NS}/Table`. Its USD has
one rigid body and 33 collision pieces (upstream: seven). The inherited table
mass/material observation and randomization terms operate on that native body.
Because the material observation emits three values per collision piece, the
critic input grows from 204 to 282 values. The actor input remains 215. This is
an explicit consequence of porting the scene with the upstream observation
encoding unchanged.

Validation on 2026-09-24 used `verify_wrist_limits.py --lab-scene` in native
Isaac Sim: table top 0.84235003 m, lower cube center 0.86234999 m, all three
wrist limits +/-180 degrees, five accepted free-hand states after 20 steps at
32 environments, and nine invalid-angle cases rejected by both acceptance and
loading. A separate eight-environment training fixture produced three finite
policy steps and confirmed the observation dimensions and table shape count.
The fixture repeats those five states under four labels solely to test startup;
it is not a production reset bank or a learning result. Fresh production banks
receive full structural/pose/limit checks and sampled native reload/FK/dynamics
validation before training.

Artifacts and the four-GPU, 1,000-update launch pipeline are in
`/data/kanth042/datasets/umi_reset_from_defaults_20260911/106_thunder_lab_easy_20260924`.
