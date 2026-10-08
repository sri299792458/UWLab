# Thunder full step-response recordings — October 8, 2026

The latest full OFF and UR-default ON runs both completed all **84,500 samples / 48 steps** of the
169-second sequence. Both report no failure or cleanup errors, and every recorded robot-state interval is 2 ms
to floating-point precision. The OFF record passes strict `records.validate_record` validation. The ON record
is comparison-only: the fitting validator correctly rejects its nonzero friction compensation scales.

| Recording | Mode | File saved, America/Chicago | Strict fitting validation |
| --- | --- | --- | --- |
| `run_20261008_nWQzjt/step_full_off_retry2.pt` | OFF | 12:28:23 | Passed |
| `run_20261008_nWQzjt/step_full_ur_default_retry1.pt` | UR default ON | 12:31:48 | Expected rejection: nonzero friction scales |

Save times come from the original files' modification timestamps. The raw records are copied byte for byte;
no samples, targets, flags, or metadata were changed. Earlier failed attempts are outside this package.

## Timing and software changes

Both captures record the receive worker at FIFO 90, control worker at FIFO 85, and application loop at FIFO 80.
They include the updated collector's 16 computation-only warmup iterations, fresh first-frame wait,
suspended automatic cyclic GC during torque collection, and host timestamps after `directTorque` returns.
The update also rejects skipped robot cycles before sending the next command, rejects computation stalls
above 20 ms, and restores GC after controller, connection, and scheduler cleanup.

| Timing metric | OFF | ON |
| --- | ---: | ---: |
| Duplicate or missed robot-state intervals | 0 | 0 |
| Maximum host sample-to-command latency | 1.284 ms | 1.427 ms |
| Maximum `directTorque` call duration | 0.085 ms | 0.089 ms |
| Host command interval range | 1.234–2.862 ms | 1.144–2.882 ms |
| Host command intervals above 2.4 ms | 746 | 1,002 |

Clean robot-state timing does not mean identical command arrival times at the robot. Host send intervals still
vary, and the workstation uses a generic kernel. Full per-run metrics are in the `.summary.json` files.
The separate receive-only audit with GPU training active completed 84,500 samples without a missed cycle;
its JSON and NPZ are included. It sent no robot-control commands and did not test torque submission.

The updated code is committed alongside these recordings, including the nearby start-pose helper and its
PolyScope 5.25+ 3PE status-bit handling. The software suite
`python -m unittest test_contract test_step_test test_move_to_start` passed all 39 tests.

## Data and interpretation

Each raw record has an extracted `.collection.json`, derived `.summary.json`, `.steps.json`, and archive
`.provenance.json`. The collector uses the unchanged pinned controller, gains, payload, calibration and
step amplitudes. The raw records do not embed the collector source hash; provenance explicitly distinguishes
the code hashes calculated at archival time from the pinned controller metadata captured in the raw file.

Step summaries use millimeters for translations and degrees for rotations. `reached`, `stop_short`, and
`return_left` are relative to the initial center pose; `moved_0p1s` and onset detection are relative to each
step's preceding pose. The arm can retain displacement after a return, so a later small step's center-relative
value must not be treated as its incremental motion. No new dynamics fit was performed.

The original config's 40 N / 14 N m wording describes the initial ideal spring component from an exactly
centered pose. It is not a cap on dynamic wrench or grip force. The current instructions clarify this.

## Integrity and reproduction

The two roughly 21 MB raw files are stored directly in Git with narrow attribute overrides, following the
existing hardware archive convention because the repository's LFS capacity is exhausted.
From this package directory, verify all included files with:

```bash
sha256sum -c SHA256SUMS
```

From `scripts_v2/tools/thunder_sim2real`, validate the OFF recording and regenerate either step summary with:

```bash
python records.py validate validation_results/step_response_hardware_20261008/run_20261008_nWQzjt/step_full_off_retry2.pt
python analyze_step_test.py validation_results/step_response_hardware_20261008/run_20261008_nWQzjt/step_full_off_retry2.pt
python analyze_step_test.py validation_results/step_response_hardware_20261008/run_20261008_nWQzjt/step_full_ur_default_retry1.pt
```

Absolute paths in historical provenance and step summaries refer to the original workstation files.
