# Policy-regime chirp hardware recording (R231) — October 9, 2026

The 50 s policy-regime excitation from `workstation/collection.policy_regime_chirp.json` (commit `634df90`) on Thunder,
for refitting the arm dynamics in the regime R214 uses (R229: the R187 model moves the wrists 6-13x more than the real arm
for small, slow commands). It completed all **25,000 samples** with no failure or cleanup errors and **passes strict record
validation** (`recording/validation.txt`). No fit has been run.

Setup: gripper open and empty, payload 1.04 kg / CoG (-0.002, 0.001, 0.047) on the controller, cubes removed, UR friction
compensation off (zero viscous/Coulomb scales), same controller as the R187 recording. The arm was brought to the start pose
with a one-off `moveJ` (0.2 rad/s; the straight joint path kept the fingertips at least 148 mm above the table), since it was
up to 82 deg away and `move_to_start.py` accepts at most 10 deg. Initial pose error: 0.0015 deg.

## Sampling

- Robot timestamp span 49.998 s; all 24,999 intervals 2 ms (duplicates 0, above 2.4 ms 0, gap events 0).
- Receive/control/loop priorities FIFO 90/85/80 on the generic kernel (no PREEMPT_RT).
- Host sample-to-command latency: max 1.241 ms, p99 0.904 ms.
- Host command intervals 1.277-2.695 ms; **118** above 2.4 ms (the October 3 recording had 3); p99 2.327 ms.

## Motion

| Joint | Peak speed (deg/s) | R187-model prediction (deg/s) | Excursion (deg) | Guard (deg) | Max torque (N m) | Limit (N m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shoulder pan | 21.8 | 30.9 | 3.21 | 8.66 | 27.35 | 150 |
| Shoulder lift | 12.7 | 16.0 | 2.38 | 11.50 | 18.66 | 150 |
| Elbow | 18.9 | 20.0 | 3.23 | 14.08 | 16.88 | 150 |
| Wrist 1 | 32.5 | 43.9 | 7.13 | 13.29 | 4.84 | 28 |
| Wrist 2 | 44.8 | 57.5 | 8.45 | 18.63 | 5.79 | 28 |
| Wrist 3 | 41.6 | 48.1 | 8.31 | 24.18 | 5.26 | 28 |

Every joint peaked below the R187 model's prediction (71-94%). Excursions stayed within the guards and no torque reached
its limit.

## Files

`recording/` holds the raw record (`thunder-fit.pt`), the config (`collection.json`, identical to the one stored in the
record), the collector's console output, the strict validation output and a derived `recording_summary.json`.
`SHA256SUMS` lists every file.
