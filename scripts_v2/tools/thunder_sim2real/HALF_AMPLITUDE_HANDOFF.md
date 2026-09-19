# Next collection: half amplitude, 0.1–1.5 Hz

Use `workstation/collection.half_slow_candidate.json`. The compatibility file
`workstation/collection.simulation_candidate.json` contains the same configuration.
This is a provisional collection to improve the model using more real data.

- Duration: 8 seconds; command rate: 500 Hz.
- Frequency sweep: 0.1–1.5 Hz.
- XYZ amplitudes: 0.05, 0.05, 0.075 m.
- XYZ axis-angle amplitudes: 0.25, 0.125, 0.25 rad.
- Same outward start pose, open empty gripper, UW controller gains and torque limits.
- Payload/connection values carried from the last physical recording: 1.04 kg,
  tool-frame CoG `[-0.002, 0.001, 0.047]` m. No new payload measurement is implied.

The start joints remain `[-171.50302, -30.67153, 73.26556, 42.38637, 60.00074,
69.07879]` degrees. Position the robot at this same reviewed start pose using
the established workstation procedure; the collector checks it and does not move
there automatically.

From the workstation checkout, using its existing collector environment:

```bash
git pull --ff-only origin thunder-umi-sim2real
python scripts_v2/tools/thunder_sim2real/workstation/collect_thunder.py \
  --config scripts_v2/tools/thunder_sim2real/workstation/collection.half_slow_candidate.json \
  --output data/thunder-half-slow.pt --execute
```

Omit `--execute` to print the plan without connecting. Use a fresh output filename
if that recording already exists. No simulator or model download is needed on
the workstation, and no hardware safety setting changes are part of this update.

## Why this motion

We optimized UWLab's 25 dynamics parameters against the first 1,600 measured
samples of the interrupted hardware run. The preliminary best model has training
angle RMSE 0.963 degrees and speed RMSE 5.022 degrees/second. This is enough to
inform a smaller collection; these are provisional parameters, not a final model
for policy training.

All eight retained fitted candidates completed the proposed 4,000-command sweep
in simulation. Peak predicted speed across them, by joint, is:

| Joint | Maximum predicted speed (degrees/s) |
|---|---:|
| Shoulder pan | 15.4 |
| Shoulder lift | 27.3 |
| Elbow | 54.2 |
| Wrist 1 | 65.8 |
| Wrist 2 | 40.5 |
| Wrist 3 | 42.5 |

These predictions used a loose simulation speed cap, with zero cap activity.
They are below the 120 degrees/second planning target; the displayed hardware
setting was 191 degrees/second. The earlier physical motion reached 205.9
degrees/second at Wrist 1.

All 4,001 saved states for each model passed source-hull checks against the
accepted lab geometry. Minimum moving-arm self separation was 15.986 mm; minimum
overall nonadjacent self separation was 5.200 mm at the gripper. Minimum world
clearance margin was 71.229 mm after the configured obstacle buffers. The complete
requested IK path also passed. These are simulated measurements, not a physical
trial of this new motion.

Evidence is in [`validation_results/partial_fit_20260919`](validation_results/partial_fit_20260919):
`partial_fit.json`, `fitted_variants.json`, `half_slow_simulation.json`,
`half_slow_hulls.json`, `requested_ik.json`, and `requested_hulls.json`.
The original partial hardware recording is retained unchanged.

## After this recording

First compare the new measured motion against the saved predictions. Then refit
using both real recordings. Test 75% amplitude at the same frequency range in
simulation before another physical increase. Consider full amplitude afterward
if the updated predictions leave adequate speed margin; assess frequency increases
separately. Accurate predictions across the intended motions are the goal.
