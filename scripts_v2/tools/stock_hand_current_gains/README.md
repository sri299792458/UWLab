# Stock hand properties with current UMI gains

This experiment completes the requested comparison at 120 Hz physics/feedback and 10 Hz policy actions. It uses the exact asset and current reset bank from the running stock-property/May-gain experiment. Only arm gains differ behaviorally: translation Kp/Kd 500/160 and rotation 60/0.1 on each axis, inherited from `UmiCubeTrainCfg`.

“Stock” refers to nominal mass, center of mass, and full rotational inertia tensors of the nine hand bodies. UMI fingers, TPU/PETG collision geometry, other collisions, joint frames, grasp reference, Thunder calibration and the +90-degree Y base mounting remain unchanged. The original 0.7–1.3 per-body mass/inertia randomization is retained, along with material and gripper-drive randomization. These fallback stock properties are an experimental intervention, not a claim of physical correctness.

Relative to the failed original UMI baseline, only the nine hand bodies' inertial properties change behaviorally. Relative to the concurrent stock-property/May-gain run, only the arm gains change. Both comparisons retain observations, action scales, torque limits, reward/success calculation, fixed lower cube, reset distribution, PPO and timing. Compare equal updates/environment samples rather than wall-clock duration.

Use task `OmniReset-UMI-StockHandCurrentGains-State-Train-v0`, with `STOCK_HAND_DYNAMICS_USD` pointing to the first experiment's asset and `STOCK_HAND_DYNAMICS_REPORT_DIR` pointing to a fresh run-specific audit directory. Its startup audit reads native values on all four workers without changing state or RNG. The shared validator accepts `--task` to select this configuration and checks 20 ordinary policy steps before launch.

The authorized run uses fresh seed 42 (rank seeds 42–45), four GPUs 0/1/2/6, 16,384 environments per worker, 32 rollout steps and a 40,000-update maximum. GPU 2 is shared with the existing unrelated process under the user's earlier explicit authorization. The other experiment continues on GPUs 3/4/5/7.

After a committed native validation passes, `prepare_launch.py --reference-metadata /absolute/path/to/62/run_metadata.json --output /absolute/path/to/64 --native-validation /absolute/path/to/64/native_validation/report.json` verifies asset/data hashes, compares saved runtime configurations, records the source difference, archives Git source and writes a distinct launch script/W&B identity. The retained result directory is `/data/kanth042/datasets/umi_reset_from_defaults_20260911/64_stock_hand_current_gains_20260918`.

The shared `scripts_v2/tools/may_hand_dynamics/verify_run.py --run /absolute/path/to/64` verifies runtime configuration, seed, all four native property/gain checks, matching stock properties with the May-gain run, process-to-GPU mapping and finite initial training output. Startup validation does not establish learning recovery.
