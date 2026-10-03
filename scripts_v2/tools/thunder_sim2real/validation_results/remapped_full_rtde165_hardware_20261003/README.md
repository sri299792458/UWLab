# Full-amplitude RTDE 1.6.5 hardware recording — October 3, 2026

Run `20261003T123920056588Z` completed all **4,000 commands** of the eight-second, 500 Hz,
0.1–3 Hz remapped full-amplitude test at 07:39 America/Chicago. It reports no
failure or cleanup errors and **passes strict record validation**. The original
config exactly matches `workstation/collection.remapped_full_amplitude_swe_candidate.json`
and the previous October 2 full run; amplitudes, gains, torque limits and payload
were unchanged.

The captured Git revision is `b638e3d8b03ee5d45c4a2ea57076dbae000fe56b` with local
changes. Capture provenance is preserved. Its collector hash matches the reviewed
implementation in commit `239daa093a6cbefcf3d6fc989aa33a9887a06500`.

## Sampling and effective scheduling

- SDK: ur-rtde 1.6.5, with explicit zero viscous/coulomb friction scales in normal commands and cleanup.
- Actual receive worker: FIFO 90; actual control worker: FIFO 85; application loop: FIFO 80.
- Robot timestamp span: 7.998 s; all 3,999 intervals are 2 ms to floating-point precision.
- Duplicate robot intervals: **0**; robot intervals above 2.4 ms: **0**; state reads spanning an update: **0**.
- Host sample-to-command latency: maximum **1.391 ms**, 99th percentile **0.707 ms**.
- Host command intervals: **0.571–3.608 ms**; **3** above 2.4 ms; 99th percentile **2.216 ms**.

The robot-state sampling issues in the October 2 record (82 duplicate intervals,
86 intervals above 2.4 ms, 15 crossed reads) are absent here. The host send
intervals still vary; strict state validation does not assert identical command
arrival times at the robot. The workstation kernel remains generic, without
PREEMPT_RT. The separate read-only preflight also passed 4,000 states under limit 99.

## Motion review

| Joint | This run peak (degrees/s) | Previous full run (degrees/s) | Reported R182 prediction (degrees/s) |
| --- | ---: | ---: | ---: |
| Shoulder pan | 124.75 | 122.19 | 53.10 |
| Shoulder lift | 100.83 | 100.25 | 85.20 |
| Elbow | 118.38 | 117.82 | 102.40 |
| Wrist 1 | 151.19 | 141.51 | 143.20 |
| Wrist 2 | 141.56 | 138.54 | 69.60 |
| Wrist 3 | 121.51 | 117.55 | 66.90 |

Peak recorded speed was **151.19 degrees/s**, at Wrist 1. No recorded
joint speed exceeds the previously reported 191 degrees/s setting. Shoulder pan,
Wrist 1, Wrist 2 and Wrist 3 exceed the unchanged 120 degrees/s planning target.
The measured base and wrist peaks still differ from the fitted-model prediction.
R182 trajectories and detailed simulation reports are absent from this checkout;
these are comparisons of maxima from the candidate's summary, not a trajectory
validation or statistical bound.

All numeric arrays are finite. The initial pose differs from the exact configured
start by at most **0.00139 degrees**. Joint excursions
remain below all configured guards, positions remain inside model joint limits,
and no commanded torque reaches its configured limit. No new dynamics fit or
measured-trajectory collision verification was performed in this review.

## Files and integrity

`20261003T123920056588Z/` contains the unchanged raw record, config, log, capture provenance,
and original validation output, plus a derived `recording_summary.json`.
`workstation_helpers_at_capture/` preserves the exact helpers whose hashes appear
in capture provenance. They are historical workstation snapshots with their
original local paths; they are not a portable installation entrypoint.
`os_setup_at_capture/` preserves the operator's permission configuration, and
`readonly_preflight_realtime.json` preserves the preceding read-only check.

The standalone portable launcher and setup instructions are in
[`../../workstation/REALTIME_SETUP.md`](../../workstation/REALTIME_SETUP.md).
Configuration notes retain their original pre-collection statements. Absolute
paths in logs/provenance are historical; no raw data or provenance was rewritten.

The raw file is stored directly in Git using a narrow attribute override because
the repository LFS budget is exhausted. Its SHA-256 is
`4f021dd094a4a625ef82638a7566371c80efcedd8467d075ab49e96753c31c49`.
Verify this package from its directory:

```bash
sha256sum -c SHA256SUMS
```
