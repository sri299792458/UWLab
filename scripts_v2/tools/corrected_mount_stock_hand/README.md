# Corrected-mount stock-hand gain comparison

The user requested restarting both stock-hand experiments with Thunder's physical mounting correction and its regenerated reset bank. Both use root quaternion **[0.5, 0.5, 0.5, 0.5] in w,x,y,z order** and nominal root position [0.177660, 0.377695, 1.466000] m. The original +/-10 mm position jitter remains. This supersedes the older [0.7071068, 0, 0.7071068, 0] mounting used by the stopped comparisons.

The two corrected configuration files are ported from the published `thunder-umi-sim2real` commit `afa9924fe099710a5cedd86a9b91a05ecbcd419a`. The corrected 20 mm map and regenerated 40,194 reset states live under `/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917`. No old full-arm reset state is reused. The object-relative grasp and partial-assembly input files are mounting independent and retain their original hashes.

Both experiments keep **UMI geometry, TPU/PETG finger collision shapes, joint linkage, calibrated joint frames, grasp reference and contact settings**. “Stock hand” means only the nine hand bodies' nominal mass, center of mass and full rotational inertia tensors. Both use exactly the same stock-property USD overlay from experiment 62 and retain all existing randomization rules.

| Condition | Translation Kp / Kd | Rotation Kp / Kd | Physical GPUs |
|---|---|---|---|
| May gains | 200 / 84.8528137424 | 3 / 3.46410161514 | 3, 4, 5, 7 |
| Current UMI gains | 500 / 160 | 60 / 0.1 | 0, 1, 2, 6 |

Each run starts fresh at seed 42, with rank seeds 42–45, 16,384 environments per worker (65,536 total), 32 rollout steps, 2,097,152 samples per update and a 40,000-update maximum. Physics and torque feedback are 120 Hz; policy actions are 10 Hz. Observations, rewards, original alignment success, fixed lower cube, reset probabilities and PPO are matched. GPU 2 shares the existing unrelated allocation with prior explicit user authorization.

These two runs isolate the gain choice **on the corrected mounting and bank**. Comparisons against older failed UMI training also change the mount and reset bank and cannot isolate inertial properties alone. Earlier old-mount controller diagnostics remain historical evidence, not tests of the corrected reset distribution.

The native startup audit verifies mass/COM/inertia and gains on every worker. A separate one-time reset audit directly reads the native PhysX base pose after the first actual reset in all environments; it asserts the corrected quaternion and permitted position jitter. This adds no simulation-state or RNG changes. Its Python bookkeeping flag only suppresses repeated file writes. The dataset verifier checks the hash, finite values and corrected root quaternion in every saved state, including all four families.

Set `UWLAB_DATA_ROOT`, `UWLAB_ASSET_ROOT`, the task-specific stock-asset/audit variables, and `CORRECTED_MOUNT_REPORT_DIR`. Run the shared native validator for each registered task with 256 environments and 20 ordinary steps. Then commit a clean worktree and run `prepare_pair.py` with the two stopped runs' metadata, `--data-root`, and the new output directory. It checks both probe configurations against their predecessors, checks that the pair differs only in gains and audit output paths, and records Git/source/data identities before producing two launch scripts.

Run `scripts_v2/tools/may_hand_dynamics/verify_run.py --run CONDITION_DIRECTORY` after each process has initialized and completed initial PPO updates. Each report verifies all four native property and post-reset mounting checks, the saved runtime configuration, actual device placement, finite initial metrics and a finite model-0 checkpoint. These checks establish correct startup, not learning success. Both older runs' logs and checkpoints are preserved in experiments 62 and 64.

Results and exact commands: `/data/kanth042/datasets/umi_reset_from_defaults_20260911/65_corrected_mount_stock_hand_comparison_20260918`.

The robot-system handoff is a separate published core release, `thunder-lab-20260917-mount-v2`, with its own original UMI inertial model. These artificial stock-property learning interventions are not changes to that robot-side handoff.
