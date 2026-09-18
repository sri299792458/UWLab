# May hand-dynamics comparison

This isolated experiment keeps the current UMI geometry and reset bank, but restores the nine hand bodies' stock nominal masses, centers of mass, and full inertia tensors. It also restores May's arm gains: translation Kp/Kd 200/84.8528137424 and rotation 3/3.46410161514. These are values read back from the reconstructed working stock simulator, not a claim of byte-identical May asset provenance.

The baseline commit records all 556 source files from the original failed seed-42 training launch. The experiment adds a separate task, `OmniReset-UMI-MayHandDynamics-State-Train-v0`. The original task and training script are unchanged. The new task adds a read-only startup audit after the inherited mass randomization; it asserts actual default properties, randomized properties, gains, and timing on every rank.

The full property values and native-source hash are in `stock_hand_properties.json`. `reference_umi_properties.json` contains the earlier native UMI readback used to check that arm-body properties remain unchanged. `build_asset.py` decomposes each symmetric body-frame inertia tensor into principal moments and axes for USD, then checks that only the 36 approved mass-property attributes differ. Geometry, joint relationships, and adjacent kinematic/grasp metadata are retained. The new asset is a layer over the existing UMI USD; the original asset is never edited.

Robot mass randomization remains uniform scaling from 0.7 to 1.3, with inertia scaled proportionally from the new nominal values. Gripper gain/material randomization, 20–200 g upper-cube masses, fixed lower cube, reward/success definitions, reset probabilities, observations, action scales, torque limits, and PPO are inherited unchanged. Physics and torque feedback remain 120 Hz; policy actions remain 10 Hz.

The requested training uses seed 42, rank seeds 42–45, GPUs 3/4/5/7, 16,384 environments per rank, and 32 rollout steps: 65,536 environments and 2,097,152 samples per update. It starts from fresh weights. The seed-142 comparison was stopped at the user's request; its logs and available checkpoint remain archived. Seed 142 completed only 64 updates, which does not rule out seed sensitivity.

Use the Isaac Python environment at `/data/kanth042/envs/uwlab-isaac51`. Build into a fresh output folder:

```bash
python scripts_v2/tools/may_hand_dynamics/build_asset.py \
  --base-usd /data/kanth042/converted_assets/thunder_d405_umi_rigid_asset/ur5e_robotiq_d405_umi_rigid_thunder.usd \
  --output /absolute/path/to/new/asset
```

Set `MAY_HAND_DYNAMICS_USD` to the output USD and `MAY_HAND_DYNAMICS_REPORT_DIR` to a fresh audit folder. Use the worktree's four `source` package directories in `PYTHONPATH`. `validate_native.py --output /absolute/path/to/probe --headless --device cuda:0` checks native properties before and after 20 ordinary policy steps. This verifies implementation and finite execution, not learning or universal stability.

After committing a clean worktree, `prepare_launch.py` accepts `--reference-metadata`, `--output`, `--asset`, and `--native-validation`. It verifies retained baseline assets/data, records the Git revision, archives source, and creates an exact launch script with a distinct W&B identity. The run is launched only after native validation passes.

The retained September 18 run record is `/data/kanth042/datasets/umi_reset_from_defaults_20260911/62_may_hand_dynamics_20260918`. This comparison changes gains and hand inertial properties together, so learning recovery would establish an effect of that combined intervention, not identify either component alone.
