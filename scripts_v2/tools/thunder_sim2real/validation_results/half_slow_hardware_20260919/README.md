# Half-amplitude hardware recording — September 19

Run `20260919T182507203720Z` completed all **4,000 commands** of the eight-second, 500 Hz,
0.1–1.5 Hz half-amplitude sweep at the outward pose. The recorded configuration
matches `workstation/collection.half_slow_candidate.json`; source commit was
`a6a6fc24125749c67a94f292f6dcaba656135ede`. The collector recorded no failure or cleanup errors.

**The strict record validator failed on duplicate/missing robot timestamps.**
The run is complete, but it is not eligible for the current fixed-step production
replay contract. Preserve its actual timing and review the raw measurements before
choosing a diagnostic fitting treatment. No resampling or completion flags were changed.

The robot timestamp span is 8.004000 s. There are
66 duplicate intervals,
69 intervals longer than 2.4 ms,
and 8 reads spanning a robot update.

| Joint | Measured peak speed (degrees/s) | Maximum predicted across eight models (degrees/s) |
| --- | ---: | ---: |
| Shoulder pan | 10.91 | 15.40 |
| Shoulder lift | 28.91 | 27.30 |
| Elbow | 51.91 | 54.20 |
| Wrist 1 | 62.15 | 65.81 |
| Wrist 2 | 35.76 | 40.51 |
| Wrist 3 | 40.65 | 42.50 |

This table compares full-run maxima, not time-aligned prediction accuracy or
statistical bounds. The next model review should compare complete measured and
predicted trajectories before refitting.

## Files

The [20260919T182507203720Z/](20260919T182507203720Z/) directory contains the original `thunder-fit.pt`,
`collection.json`, `collector.log`, `provenance.json`, and `validation.txt`, plus
`recording_summary.json`. Workstation paths in logs and provenance are historical.
The earlier manually stopped recording is excluded.

The raw file is stored directly in Git using a local attribute override because
the repository's LFS budget was exhausted. Its bytes match the workstation original.
From this directory, verify the evidence with:

```bash
sha256sum -c SHA256SUMS
```
