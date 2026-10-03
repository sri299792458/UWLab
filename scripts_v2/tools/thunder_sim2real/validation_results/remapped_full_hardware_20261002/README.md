# Remapped full-amplitude hardware recording — October 2, 2026

Run `20261003T001719268791Z` completed all **4,000 commands** of the eight-second, 500 Hz,
0.1–3 Hz remapped full-amplitude sweep at the new S-W-E- start pose. The run
identifier is UTC; the collection occurred October 2 in America/Chicago.
The saved configuration matches `workstation/collection.remapped_full_amplitude_swe_candidate.json`
at source commit `0bd14897cc27b9bfbc9208515ad799b8e173b975`. The original recording
reports no failure and no controller cleanup errors. Its RTDE version is 1.6.2.

**The strict record validator failed on duplicate/missing robot timestamps.**
The raw recording is complete but does not meet the current fixed-step production
replay contract. No samples, timestamps, configuration or completion flags were changed.

The robot timestamp span is 8.012 s. There are **82 duplicate intervals**, **86
intervals longer than 2.4 ms** (maximum 10 ms), and **15 state reads spanning a
robot update**. Host timestamps are ordered sample-before-command.

| Joint | Measured peak speed (degrees/s) | Reported R182 prediction (degrees/s) |
| --- | ---: | ---: |
| Shoulder pan | 122.19 | 53.10 |
| Shoulder lift | 100.25 | 85.20 |
| Elbow | 117.82 | 102.40 |
| Wrist 1 | 141.51 | 143.20 |
| Wrist 2 | 138.54 | 69.60 |
| Wrist 3 | 117.55 | 66.90 |

Peak recorded speed was **141.51 degrees/s**, at Wrist 1. No recorded speed exceeded
the previously displayed 191 degrees/s setting. Shoulder pan, Wrist 1 and Wrist 2
exceeded the 120 degrees/s planning target. The base, Wrist 2 and Wrist 3 peaks
substantially exceed the reported prediction. These are comparisons of full-run
maxima, not a time-aligned trajectory evaluation or statistical bounds. The underlying
R182 simulation trajectories/reports are not present in this checkout, so the
prediction numbers above are taken from the candidate configuration.

All numeric arrays are finite. The initial pose is within 0.00112 degrees of the
configured start. Joint excursions remain below the configured guards, and positions
remain within the model's joint limits. No commanded torque reached its configured
limit. These checks do not establish collision clearance for the measured trajectory
or prove the candidate's dynamics predictions. Review the timing and prediction
mismatch before repeating the full-amplitude collection.

## Files and integrity

The [20261003T001719268791Z/](20261003T001719268791Z/) directory contains the unchanged workstation originals:
`thunder-fit.pt`, `collection.json`, `collector.log`, `provenance.json`, and
`validation.txt`, together with the derived `recording_summary.json`. Absolute paths
in logs and provenance are historical. Configuration notes retain their original
pre-collection status.

The raw file is stored directly in Git using a narrow attribute override because
the repository LFS budget was exhausted. Its SHA-256 is
`22198670b54725de43af36bc8de8adf77e7e9a9d36425dc48cc225d40eea4a5a`.
Verify the evidence from this directory with:

```bash
sha256sum -c SHA256SUMS
```
