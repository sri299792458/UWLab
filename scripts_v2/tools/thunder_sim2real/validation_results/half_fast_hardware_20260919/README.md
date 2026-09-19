# Half-amplitude 0.1–3 Hz hardware recording — September 19

Run `20260919T201242292903Z` completed all **4,000 commands** of the eight-second, 500 Hz,
0.1–3 Hz half-amplitude sweep at the outward pose. The recorded configuration
matches `workstation/collection.half_fast_candidate.json`; source commit was
`b5e2efd92cde47319e89248e8a2306a630ebb87e`. No failure or cleanup errors were recorded.

**The strict record validator failed on duplicate/missing robot timestamps.**
The run is complete but does not satisfy the current fixed-step production replay
contract. Preserve the actual timing when reviewing and fitting this recording.
No raw samples, timestamps, configuration metadata, or completion flags were changed.

The robot timestamp span is 8.002000 s, with
42 duplicate intervals,
44 intervals longer than 2.4 ms, and
6 reads spanning a robot update.

| Joint | Measured peak speed (degrees/s) | Maximum predicted across eight models (degrees/s) |
| --- | ---: | ---: |
| Shoulder pan | 11.96 | 17.25 |
| Shoulder lift | 52.54 | 47.78 |
| Elbow | 57.28 | 59.88 |
| Wrist 1 | 97.55 | 102.76 |
| Wrist 2 | 34.06 | 41.67 |
| Wrist 3 | 47.09 | 51.23 |

This compares full-run maxima, not time-aligned prediction accuracy or statistical
bounds. Compare the complete measured response with the saved predictions before
refitting. Predictions are in `../half_fast_review_20260919/half_fast_simulation.json`.

## Files

The [20260919T201242292903Z/](20260919T201242292903Z/) directory contains the original
`thunder-fit.pt`, `collection.json`, `collector.log`, `provenance.json`, and
`validation.txt`, plus a derived `recording_summary.json`. Paths in logs and
provenance are historical. Configuration notes preserve the pre-collection review
status as recorded at collection time.

The raw file is stored directly in Git using a narrow attribute override because
the repository's LFS budget was exhausted. Its bytes match the workstation original.
Verify the evidence from this directory with:

```bash
sha256sum -c SHA256SUMS
```
