# Half-amplitude 0.1–3 Hz collection review

These reports were computed before publication of the next workstation collection
configuration. The physical 0.1–3 Hz half-amplitude experiment has not been executed
as part of this review. The active configuration is
[`collection.half_fast_candidate.json`](../../workstation/collection.half_fast_candidate.json).

## Inputs and method

The eight dynamics candidates in `fitted_variants.json` were fitted to the saved
prefix of the interrupted full-amplitude recording. They were used without refitting
to predict the completed 0.1–1.5 Hz half-amplitude hardware recording published in
commit `301b64f1e0fc4c4493cdc8feb1256ab52510d892`. Its SHA-256 is
`7426bd05924f41dabe8cf5e38848525f099865dc5d8c30f1e43e2354b7a90d61`.

`half_slow_prediction_validation.json` compares those saved predictions with all
4,000 measured samples using pre-command state alignment by command index and the
nominal 2 ms interval. The best model selected by the earlier training fit achieved
0.721552 degree position RMSE and 2.639964 degree/second speed RMSE. These errors
measure simulation-to-hardware agreement, not tracking of the commanded target.
The source recording and its original timing metadata were not modified.

`half_fast_simulation.json` contains the next motion's controller, waveform, start
pose, model parameters, and speed results. Native simulation used `preview_motion.py`
with those eight dynamics variants, amplitude scale 0.5, frequency limits 0.1 and
3 Hz, duration 8 seconds, and the original collection controller. A loose simulation
speed cap of 1000 rad/s was verified to have no activity. The largest predicted
joint speed is 102.755653 degrees/second, below the 120 degrees/second planning target.
The eight-model spread is not a statistical confidence interval.

`half_fast_hulls.json` checks all 4,001 saved states for each model against the
accepted 33-box lab geometry using the robot's source convex hulls. All pass.
Minimum moving-arm self separation is 16.173 mm; minimum world margin after
obstacle buffers is 71.229 mm. `requested_ik.json` and `requested_hulls.json` check
all 4,000 requested poses separately using calibrated inverse kinematics (IK).
All solve within joint limits and pass the geometry check. IK assumes ideal
tracking; its speeds must not be substituted for the simulated response speeds.

The configuration's excursion guards take the maximum absolute joint excursion
across all eight dynamic predictions and the requested IK path, then add five
degrees per joint. The controller remains the one used in the preceding physical
recordings. This review does not establish that its gains are optimal for Thunder.

## Report integrity

The six JSON files are unchanged copies of the original reports. Their absolute
server paths document provenance; they are not workstation dependencies. Verify
the copies from this directory with `sha256sum -c SHA256SUMS`.

The full server outputs, including trajectories and target poses, are retained at
`/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917/75_full_amplitude_review_20260919`.
Use the [current handoff](../../HALF_AMPLITUDE_HANDOFF.md) for collection instructions.
