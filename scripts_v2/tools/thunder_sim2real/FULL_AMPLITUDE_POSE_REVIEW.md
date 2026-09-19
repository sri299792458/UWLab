# Full-amplitude candidate from the latest recording

The new pose reduces worst predicted joint speed from **223.8°/s to 147.9°/s**
for the original full-amplitude 0.1–3 Hz sweep, a 34% reduction using the same
controller and eight fitted models. Both the simulated responses and the complete
requested path pass the accepted geometry checks. **The candidate exceeds the
unchanged 120°/s planning target**, so it remains a review candidate rather than
an approved replacement for the collected half-amplitude configuration.

The model was fitted only to the completed half-fast recording. Its replay errors
are 0.504° RMS position and 3.030°/s RMS speed, with all 4,000 samples used for
training. [Model parameters, method and reproduction](validation_results/half_fast_only_fit_20260919/README.md)
include the eight independent fit results. No older recording contributes to this fit.

## Pose and waveform

`workstation/collection.full_amplitude_pose_candidate.json` is the concrete candidate.
Joint order is shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3.

| Joint | Start angle (degrees) | Worst predicted peak speed (degrees/s) |
|---|---:|---:|
| Shoulder pan | 13.34042 | 59.95 |
| Shoulder lift | -124.13324 | 97.86 |
| Elbow | -136.67602 | 133.92 |
| Wrist 1 | -99.11863 | 147.87 |
| Wrist 2 | -106.39103 | 144.29 |
| Wrist 3 | -90.86033 | 80.29 |

Use the exact radians stored in the configuration for reproduction. This is a
different arm configuration from the recorded pose. The path for moving the real
robot from its current pose to this starting pose has not been checked. The collector
only checks the starting angles; it does not reposition the arm.

The motion lasts 8 seconds at 500 Hz and sweeps 0.1–3 Hz. Full XYZ amplitudes are
`[0.10, 0.10, 0.15]` m; XYZ axis-angle amplitudes are `[0.50, 0.25, 0.50]` rad.
Stiffness stays `[1000,1000,1000,50,50,50]`, damping ratios stay all 1, and torque
limits stay `[150,150,150,28,28,28]` Nm. The open, empty tool and recorded payload
configuration are retained. These gains are a comparison baseline, not newly
optimized gains for Thunder.

Print the candidate plan without connecting to the robot:

```bash
python scripts_v2/tools/thunder_sim2real/workstation/collect_thunder.py \
  --config scripts_v2/tools/thunder_sim2real/workstation/collection.full_amplitude_pose_candidate.json
```

The compatibility configuration still contains the completed half-fast motion.
No hardware commands were sent while producing this review.

## What the checks establish

353 candidate entries were explored, followed by finalist checks and an eight-model
repeat of the 97 original atlas poses. Pose 65 is the best completely geometry-checked
candidate by worst-model peak speed. This is a sampled search, not proof of a global optimum.

The best fitted model predicts 139.7°/s. The eight independently fitted models predict
peak speeds from 114.1 to 147.9°/s. All lie below the displayed 191°/s hardware setting,
but that comparison is not a hardware safety guarantee and does not pass the 120°/s
planning screen. No simulation speed cap is active. The minimum total motion range
across models is 3.8° for shoulder pan and 20.0–34.0° for the other joints: this pose
still excites shoulder pan less strongly than the others.

Every 2 ms native state was checked: 4,001 states for each of eight models. Minimum
moving-arm self separation is 17.189 mm; minimum obstacle margin is 28.737 mm after
the existing buffers (50 mm at columns, 1 mm elsewhere). Separately, all 4,000
requested targets have calibrated inverse-kinematics solutions within joint limits.
Their minimum moving-arm separation is 17.123 mm and obstacle margin is 21.779 mm.
These checks use the source convex collision shapes and the accepted 33 lab boxes.
They do not cover unknown obstacles or the transition into the starting pose.

“Full amplitude” describes the commanded target. The compliant controller does not
track that target exactly, especially at high frequency. The reported speeds are
native simulated controller responses; the requested inverse-kinematics path checks
geometry under ideal tracking. Its much larger ideal speeds are not predictions of
actual robot speed. No full-amplitude recording at this pose has been collected.

[Detailed evidence and reproduction](validation_results/full_amplitude_pose_20260919/README.md)
include the pose images, speed plot, source-hull reports and candidate configuration.
The rendered motion video remains local because the repository LFS budget prevented upload.
