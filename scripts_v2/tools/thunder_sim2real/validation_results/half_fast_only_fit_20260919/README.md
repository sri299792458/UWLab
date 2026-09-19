# Model fitted only to the completed half-amplitude 0.1–3 Hz recording

The selected model reproduces the 4,000 samples of the latest hardware recording
with **0.503851° RMS joint-position error** and **3.030278°/s RMS joint-speed error**.
These are training errors: every sample was used, with no held-out validation.
`best_fit.json` contains the selected parameters and provenance;
`evaluation.json` contains a separate native-simulator replay of that checkpoint.

## Input and method

The sole recording is
`../half_fast_hardware_20260919/20260919T201242292903Z/thunder-fit.pt`, SHA-256
`4bb0140cfd3bb93a74a4d1040fd073f0529b886c5ab5a2837b68e7e7c4b71b36`.
It contains the completed eight-second run at a nominal 500 Hz. Older fits supplied
initial parameter guesses only; older recordings did not contribute to the objective.
The raw recording and all timestamps remain unchanged. As requested for this
planning experiment, comparison uses command order at nominal 2 ms spacing.
This does not repair the timing anomalies or bypass the production export validator.

Eight independent searches used physical GPUs 0–7, 128 simulated candidates each,
and completed 7,168 candidate evaluations. The optimizer changes 25 values:
six added joint inertias (armature), six static-friction values, six ratios of moving
to static friction, six velocity-dependent friction values, and one command delay.
The selected delay rounds to three 2 ms steps, or 6 ms.

The objective is the mean over samples of the sum over joints of
`position_error_rad² + (0.05 s × velocity_error_rad_s)²`.
Thus a speed error of 20°/s contributes the same amount as an angle error of 1°.
The controller gains, torque limits, calibrated robot geometry and inertial model
remain fixed. The offline simulator's speed cap is raised to 1,000 rad/s so it does
not hide excessive predicted speed; this changes no hardware limit.

Searches were stopped after retaining completed populations to spend the remaining
compute on pose checks. Global convergence is not claimed. `fit_selection.json`
records each run and its best score; `parallel_fit_plan.json` records seeds and
GPU assignment. `pose_models.json` retains the best model from each of the eight
independent searches for checking new motions. It is a sensitivity set, not a
statistical confidence interval. Different parameters can explain this one motion
similarly while predicting different behavior at another pose.

## Reproduce the selected search

From the repository root in the existing UWLab / Isaac Sim environment:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts_v2/tools/thunder_sim2real/fit_partial.py \
  --record scripts_v2/tools/thunder_sim2real/validation_results/half_fast_hardware_20260919/20260919T201242292903Z/thunder-fit.pt \
  --warm_start scripts_v2/tools/thunder_sim2real/validation_results/half_fast_only_fit_20260919/warm_start.json \
  --output /tmp/thunder-half-fast-only-fit \
  --num_envs 128 --iterations 7 --train_steps 0 --seed 44 \
  --parameterization logit --search_scale 0.5 --velocity_weight_s 0.05 \
  --headless --device cuda:0
```

Small numerical differences across native GPU replays are expected. The independent
replay's weighted score differs from the optimization checkpoint by about 0.11%.
`fit_angles.png` and `fit_speeds.png` compare measured samples with fitted responses.
Their orange band is the eight selected candidates from the winning search, whereas
`pose_models.json` uses one best candidate from each independent search.
This package is a planning estimate; no training deployment profile was promoted.
