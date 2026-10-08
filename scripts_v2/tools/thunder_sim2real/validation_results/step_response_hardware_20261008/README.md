# Thunder step-response hardware collection — October 8, 2026

All five planned recordings completed with no failure, no cleanup errors, and no duplicate or missed
robot-state intervals. The three compensation-OFF records pass strict `records.validate_record` validation.
The two corrected compensation-ON records are for response comparison; the fitter rejects their nonzero scales.
No simulation comparison or new dynamics fit has been performed.

## Records to use

All paths below are under `run_20261008_nWQzjt/`. `collection_review.json` lists the valid files,
SHA-256 hashes, analysis paths and fitting eligibility.

| Recording | Test | Compensation | Samples | Steps/trials | Planned duration |
| --- | --- | --- | ---: | ---: | ---: |
| `step_full_off_retry2.pt` | Full held grid | OFF | 84,500 | 48 | 169 s |
| `step_full_ur_default_fixed_retry1.pt` | Full held grid | Corrected UR default ON | 84,500 | 48 | 169 s |
| `policy_off_retry2.pt` | 10 Hz re-anchored policy targets | OFF | 99,500 | 66 | 199 s |
| `policy_ur_default_fixed_retry1.pt` | 10 Hz re-anchored policy targets | Corrected UR default ON | 99,500 | 66 | 199 s |
| `fine_off_retry1.pt` | Fine held grid | OFF | 63,500 | 36 | 127 s |

Each valid record has a `.collection.json`, `.steps.json`, `.review.json` and `.provenance.json`.
Raw files were copied byte for byte; samples, targets, flags and metadata were not changed.
Each analysis confirms all scheduled return phases ran. This does not imply exact return to the center pose.
The policy records have observed policy-step durations of 0.100000 s throughout.

## Historical record excluded from ON comparisons

**`step_full_ur_default_retry1.pt` requested ON but actually used zero compensation.**
The ur_rtde 1.6.5 compiled-in PolyScope 5.26 script reads the requested scales into different variable names
from those passed to `direct_torque`. Its captured nonzero settings describe the request, not effective behavior.
The earlier archive description incorrectly called this an ON comparison. The original raw file and its
historical summaries are retained, and its provenance now records the correction. Use
`step_full_ur_default_fixed_retry1.pt` for the held ON comparison instead.

Both valid ON records capture the corrected script hash
`bad20358db9ca01fb10ae0a20740c2b9e12064e27e7e77b2a9229384e4d20b71`, checked against the vendored script.
For example, the +x 40 mm held target ends at 5.628 mm with OFF and 40.228 mm with corrected ON, measured
from each run's initial center. Policy responses remain dependent on axis, direction and preceding motion;
ON does not improve every trial uniformly.

## Timing and software

The three newest recordings (policy OFF/ON and fine OFF) use a preallocated, touched and memory-locked
NumPy recording buffer and request only `timestamp`, `actual_q` and `actual_qd` in the receive recipe.
All records capture receive/control/application FIFO priorities 90/85/80. The earlier held records use
the preceding logger; controller gains, payload, calibration and step targets were not changed by the
recording update. The workstation still runs a generic kernel.

The new `--allow-small-gaps` option is explicit and restricted to response comparisons: at most 6 ms
between states and five missing cycles across a run. It preserves actual timestamps and gap events.
The three newest runs selected this option but recorded zero gaps, so they did not use the tolerance.
Fixed-step sysid validation still rejects gaps; the 20 ms stream/computation guards and joint/torque limits remain.

`timing_diagnostics/` preserves receive-only evidence, including the preallocated/default-recipe audits
that still missed cycles, the clean minimal-recipe audit, and a native host-timer probe with training active.
These audits sent zero robot-control commands. The initial held receive-only audit is also retained.
This evidence does not establish the cause of every earlier miss or guarantee future deadline behavior.

`python -m unittest test_contract test_step_test test_move_to_start` passed all **53 tests**.
`verification.json` records current checks and source hashes; `verification.initial.json` retains the
initial archive verification. Source hashes identify code at archival time, not proof of the exact collector
source at capture. Raw metadata supplies the captured controller, calibration, compensation-script and timing details.

## Interpretation and integrity

All five valid `.steps.json` files use the current timestamp-aware analyzer. It measures endpoints at the
next pre-command boundary state and uses actual robot times for onset and the 0.1 s observation.
The earlier full-OFF table is preserved as `.steps.previous.json`; historical `.summary.json` files remain unchanged.
Translation values use millimeters and rotation values use degrees. Held `reached`, `stop_short` and
`return_left` are relative to the initial center, while `moved_0p1s` and onset use the pre-step pose.
Residual displacement can make a small center-relative endpoint look like motion that did not occur during that step.

The raw recordings are stored directly in Git with file-specific attribute overrides, following the
existing hardware archive convention because repository LFS capacity is exhausted.
From this package directory, check integrity with:

```bash
sha256sum -c SHA256SUMS
```

From `scripts_v2/tools/thunder_sim2real`, validate an OFF record or regenerate a step summary with:

```bash
python records.py validate validation_results/step_response_hardware_20261008/run_20261008_nWQzjt/policy_off_retry2.pt
python analyze_step_test.py validation_results/step_response_hardware_20261008/run_20261008_nWQzjt/policy_off_retry2.pt
```

Absolute paths in source provenance and derived summaries refer to the original workstation files.
