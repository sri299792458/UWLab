# Thunder hardware collection evidence — September 19

The outward full-amplitude collection triggered C283A7 for Wrist 1 overspeed.
Its last saved velocity was 3.593506 rad/s (205.893 degrees/s), above the
191 degrees/s setting shown on the pendant. **Do not repeat the unchanged sweep.**
Revise and validate the excitation and speed handling, and correct sample
synchronization before another collection. Keep the robot safety settings intact.

This recording is incomplete diagnostic evidence. Its original `thunder-fit.pt`
filename is preserved, but it is invalid input to the production fitter.
No controller behavior or robot safety setting was changed in this
evidence update.

| Run (workstation UTC identifier) | Pose | Saved samples | Robot timestamp span | Stop evidence |
| --- | --- | ---: | ---: | --- |
| [20260919T163903086675Z](20260919T163903086675Z/REVIEW.md) | 15 cm outward start | 2,031 / 4,000 | 4.064 s | Pendant C283A7, Wrist 1 speed violation. |

## Outward-run findings

The recorded configuration matches the published outward candidate with the
workstation hardware values filled in: PolyScope 5.26.1.140514, ur-rtde 1.6.2,
payload 1.04 kg and tool-frame CoG [-0.002, 0.001, 0.047] m. These payload values
are controller-configured values, not a new physical mass/CoG measurement.
The source commit was `169e6e0619689011cdd2c247104e434242cd0d73`.

All saved commanded torques remained below their clipping limits, and positions
remained within the configured joint-excursion limits. Neither check bounds
joint speed. The hardware collector has no runtime joint-speed guard. The
simulation asset sets `velocity_limit_sim` to 1.5708 rad/s for the first three
joints and 3.1415 rad/s for the wrists. Its Wrist 1 peak speeds were 3.127152,
2.405484 and 2.277676 rad/s in the three reference cases. Those cases did not
bound the measured 3.593506 rad/s response; their collision-clearance passes
did not establish physical speed compliance.

The recording also contains 45 duplicate robot timestamps, 47 approximately
4 ms intervals, and four reads spanning a robot update. These separately
violate the fixed 500 Hz fitting contract. The failed torque command and
subsequent braking state were not saved, and the pendant clock has not been
aligned with the workstation clock. The photo and recorded velocity are
consistent with the overspeed diagnosis; neither establishes a collision or
a payload mismatch.

## Contents and verification

The run directory contains the original recording, collection configuration,
collector log, source provenance and offline review. It also
contains the two pendant photos and a detailed [review](20260919T163903086675Z/REVIEW.md).
`validation.txt` records the expected validator rejection. Original JSON and
log paths refer to the collection workstation and are retained as provenance;
the files in these directories are portable evidence.

The JSON configuration notes saying "No real robot execution" describe the
candidate when it was prepared. They are preserved unchanged as part of the
recorded configuration; this report documents the subsequent physical attempts.

This 496 KB diagnostic `.pt` file is stored directly in Git through a local
attribute override because the repository's LFS budget was exhausted when the
evidence was published. Its original bytes are preserved. From this directory:

```bash
sha256sum -c SHA256SUMS
```

The standard validator intentionally rejects this recording. From the repository
root, for example:

```bash
python scripts_v2/tools/thunder_sim2real/records.py validate \
  scripts_v2/tools/thunder_sim2real/validation_results/hardware_20260919/20260919T163903086675Z/thunder-fit.pt
```

The expected first error is `ValueError: completed must be True`. Preserve that
failure status; do not mark the recording complete or silently resample it.

UR's [C283 documentation](https://www.universal-robots.com/manuals/EN/HTML/SW5_26/Content/prod-err-codes/topics/CODE_283.html)
identifies A7 as exceeding a joint's safety-speed setting. The
[direct-torque documentation](https://www.universal-robots.com/manuals/EN/HTML/SW5_26/Content/prod-scriptmanual/all_scripts/direct_torque.htm)
states that each call uses a timestep regardless of speed scaling; reducing the
pendant speed slider is not a substitute for revising this excitation/controller.
