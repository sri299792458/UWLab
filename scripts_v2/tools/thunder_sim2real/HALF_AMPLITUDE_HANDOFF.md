# Next collection: half amplitude, 0.1–3 Hz

Use `workstation/collection.half_fast_candidate.json`. The compatibility file
`workstation/collection.simulation_candidate.json` contains the same configuration.
This extends the completed half-amplitude 0.1–1.5 Hz collection to higher frequencies.

- Duration: 8 seconds; command rate: 500 Hz; frequency sweep: 0.1–3 Hz.
- XYZ amplitudes: 0.05, 0.05, 0.075 m.
- XYZ axis-angle amplitudes: 0.25, 0.125, 0.25 rad.
- Same outward start pose and open, empty gripper.
- Stiffness: `[1000,1000,1000,50,50,50]`; damping ratio: `[1,1,1,1,1,1]`.
- Torque limits: `[150,150,150,28,28,28]` Nm.
- Configured payload carried from the previous recording: 1.04 kg,
  tool-frame CoG `[-0.002, 0.001, 0.047]` m.

These gains are retained to compare motions under the same controller. The model
checks below do not establish that these gains are optimal for Thunder. Controller
tuning should assess target tracking, oscillation, torque demand, and contact behavior.

The start joints remain `[-171.50302, -30.67153, 73.26556, 42.38637, 60.00074,
69.07879]` degrees. Position the robot at this reviewed pose using the established
workstation procedure. The collector checks the pose and does not move there automatically.

From the workstation checkout, using its existing collector environment:

```bash
git pull --ff-only origin thunder-umi-sim2real
python scripts_v2/tools/thunder_sim2real/workstation/collect_thunder.py \
  --config scripts_v2/tools/thunder_sim2real/workstation/collection.half_fast_candidate.json \
  --output data/thunder-half-fast.pt --execute
```

Omit `--execute` to print the plan without connecting. Use a fresh output filename
if the recording already exists. No simulator or model download is needed on the workstation.

## Existing evidence for this collection

The best model from the earlier training fit predicted the completed half-slow
hardware recording with 0.722 degree joint-position RMSE and 2.640 degree/second
joint-speed RMSE. Those predictions were saved before collection; the model was
not refitted to that recording. This measures simulation-to-hardware agreement,
not tracking error relative to the requested target.

For the new 0.1–3 Hz motion, peak predicted joint speeds across eight fitted models are:

| Joint | Maximum predicted speed (degrees/s) |
|---|---:|
| Shoulder pan | 17.3 |
| Shoulder lift | 47.8 |
| Elbow | 59.9 |
| Wrist 1 | 102.8 |
| Wrist 2 | 41.7 |
| Wrist 3 | 51.2 |

All models remain below the 120 degrees/second planning target with no simulation
speed-cap activity. All 4,001 saved states per model pass source-hull checks against
the accepted lab geometry. Minimum moving-arm self separation is 16.173 mm and
minimum world clearance margin is 71.229 mm after obstacle buffers. All 4,000
requested poses solve within joint limits and pass the same geometry checks.
Requested-path speeds describe ideal tracking and differ from the predicted
controller response above. These are simulation results; this configuration has
not yet been recorded on the robot.

The existing joint-excursion guards use maximum excursion across the eight
simulations and requested path, plus five degrees per joint. Evidence and method
are in [the published review](validation_results/half_fast_review_20260919/README.md).

## After this recording

Compare the measured response with the saved predictions, then fit using the
usable interrupted recording and both half-amplitude recordings, normalizing
each recording's contribution by sample count. Preserve raw recordings and their
timing and completion metadata. Replays must use the controller settings recorded
with each dataset. This run does not promote a final dynamics profile for training.
