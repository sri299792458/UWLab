# Full-amplitude pose 65 review

See [the pose review](../../FULL_AMPLITUDE_POSE_REVIEW.md) for the result and its
limits. `review.json` is the computed summary. The 120°/s planning screen fails;
the geometry checks pass. No hardware execution is implied.

| File | Meaning |
|---|---|
| `candidate.json` | Exact tested starting joint angles |
| `simulation.json` | Eight final fitted-model responses, extracted from the atlas replay |
| `simulation_hulls.json` | Source-shape checks at every saved native 2 ms state |
| `requested_ik.json` | Full requested path solved using calibrated inverse kinematics |
| `requested_hulls.json` | Source-shape checks for every requested target |
| `predicted_speeds.png` | Computed speeds from all eight native replays |
| `start_pose.png`, `start_pose_second_view.png` | Two views of the selected pose |
| `render_report.json` | Render-to-recorded-state consistency check |
| `robust_atlas_summary.json` | All 97 atlas poses checked against eight final models |
| `local_search_summary.json`, `focused_search_summary.json` | Earlier search rounds with their recorded preliminary model sets |
| `finalists_speed_summary.json` | Eight focused finalists checked against all final models |

The video remains a local analysis artifact because the repository LFS budget
prevented upload. Its path and checksum are in `local_video.json`. Recreate it with
`render_collection_motion.py` from the reproduced native preview below.
The video replays saved joint states at 15 frames/second. It is a visualization of
computed simulator output, not a hardware recording and not the resolution used
for collision checks. Collision checks use every saved 500 Hz state.
Absolute paths in reports are provenance from the analysis server; the workstation
collector only needs the candidate configuration.

## Reproduce the motion review

On the existing calibrated UWLab simulation server, use the already configured
asset and geometry paths (or `UWLAB_DATA_ROOT` and `UWLAB_ASSET_ROOT`):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts_v2/tools/thunder_sim2real/preview_motion.py \
  --candidates scripts_v2/tools/thunder_sim2real/validation_results/full_amplitude_pose_20260919/candidate.json \
  --dynamics_variants scripts_v2/tools/thunder_sim2real/validation_results/half_fast_only_fit_20260919/pose_models.json \
  --output /tmp/thunder-full-pose-65 --amplitude_scale 1 --f0_hz 0.1 --f1_hz 3 \
  --velocity_limit_mode diagnostic --review_speed_deg_s 120 --headless --device cuda:0
python scripts_v2/tools/thunder_sim2real/check_motion_hulls.py \
  --preview /tmp/thunder-full-pose-65 --candidate 65 --include_sphere_failures
python scripts_v2/tools/thunder_sim2real/check_requested_motion.py \
  --preview /tmp/thunder-full-pose-65 --candidate 65 --output /tmp/thunder-full-pose-65-requested
python scripts_v2/tools/thunder_sim2real/check_motion_hulls.py \
  --preview /tmp/thunder-full-pose-65-requested --candidate 65 --include_sphere_failures
```

Joint-excursion stop thresholds are computed from the maximum excursion across all
eight native responses and requested inverse-kinematics states, plus five degrees
per joint. The configuration records both the measured maxima and the allowance.
This is the existing excursion guard, not a stopping-distance guarantee.
