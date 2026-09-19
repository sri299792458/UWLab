# Outward collection-pose handoff

The selected start is shifted **15 cm outward from the columns**, with the same
27 cm grasp-center height and wrist orientation. Use
[`workstation/collection.outward_candidate.json`](workstation/collection.outward_candidate.json).
`collection.simulation_candidate.json` is an identical compatibility copy.
The eight-second, 500 Hz, 0.1–3 Hz waveform and UW collection gains are unchanged;
all 4,000 × 6 offset samples match the pinned upstream generator exactly.

## Get the update

```bash
git switch thunder-umi-sim2real
git pull --ff-only origin thunder-umi-sim2real
```

The corrected-mount core model/map package remains compatible. No model or map
reinstallation is needed for this pose change. Download the supplemental evidence:

- [Release and complete artifact list](https://github.com/sri299792458/UWLab/releases/tag/thunder-sysid-outward-20260919)
- [Eight-second simulation video](https://github.com/sri299792458/UWLab/releases/download/thunder-sysid-outward-20260919/collection_sweep.mp4)
- [Full trajectories, reports, video and configuration](https://github.com/sri299792458/UWLab/releases/download/thunder-sysid-outward-20260919/thunder-sysid-outward-20260919.tar.gz)
- [SHA-256 checksums](https://github.com/sri299792458/UWLab/releases/download/thunder-sysid-outward-20260919/SHA256SUMS)

The archive extracts into `thunder-sysid-outward-20260919/`. Its
`artifact_manifest.json` records every included file's hash and the source commit.
The core package's older `lower_full` motion remains historical.

## Geometry and motion evidence

All 4,000 requested poses and 4,001 states in each of three simulated dynamics
cases passed source-convex-hull checks. Minimum modeled column clearance is
121.229 mm; minimum distance to other lab geometry is 93.729 mm. Minimum
nonadjacent moving-arm clearance is 12.699 mm on the ideal path and 14.446 mm
across simulated responses. Internal open-hand clearance is approximately 5.2 mm.
These are sampled model distances; no real robot recording or positioning path
is supplied. The video replays the UW-reference-dynamics case with 4 ms delay.

Joint travel is maximum minus minimum angle. UW calculated values use the original
pinned UW calibration and default pose `[0,-90,90,-90,-90,0]` degrees. Calculated
values assume perfect tracking; they are not UW hardware measurements. The simulated
column is the Thunder response shown in the video.

| Joint | UW calculated | Outward calculated | Outward simulated |
|---|---:|---:|---:|
| Shoulder pan | 30.9° | 24.4° | 5.9° |
| Shoulder lift | 30.5° | 32.1° | 21.4° |
| Elbow | 34.3° | 57.3° | 16.1° |
| Wrist 1 | 50.3° | 27.8° | 27.7° |
| Wrist 2 | 56.7° | 57.9° | 23.7° |
| Wrist 3 | 69.7° | 66.9° | 25.6° |

The outward pose requests more elbow travel and less wrist-1 travel. The simulated
shoulder moves only 5.88 degrees. In the separate zero-added-friction/inertia case,
wrist 3 moves only 0.54 degrees despite high speed. Collision clearance alone does
not establish identification quality or stable tracking. Real recorded response
and fitted replay remain the basis for evaluating the dynamics fit.

Complete min/max angles and peak speeds are in
[`validation_results/outward_20260919/joint_ranges.json`](validation_results/outward_20260919/joint_ranges.json).
That folder also contains requested-path IK, all collision checks, waveform parity
and rendering validation. Original server paths in these reports are provenance,
not required workstation paths.

The configuration retains unset robot IP, PolyScope version and measured payload
fields. The collector requires the robot already at the selected pose; it does
not move there. Follow the existing [robot-side handoff](portable/ROBOT_AGENT_HANDOFF.md)
for positioning, collection and data return.
