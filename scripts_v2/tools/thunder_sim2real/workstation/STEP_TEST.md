# Controller step-response test (hardware counterpart of sim R223)

Measures how far Thunder's flange actually moves toward a **held** Cartesian target offset under the training controller
(OSC Kp 1000/1000/1000/50/50/50, damping ratio 1, torque limits 150/28): the static-friction dead band and "stop short" error.
Run it twice: UR friction compensation **off** (the fitted setting) and **on** (URScript default per-joint scales).

Setup: same as the clean R187 sysid recording. Gripper open and empty, payload 1.04 kg / CoG (-0.002, 0.001, 0.047) as
configured on the controller, start pose c675 (S-W-E-) from `collection.remapped_full_amplitude_swe_candidate.json`. Move the arm
to the start pose yourself; the script refuses to run outside 0.02 rad of it or if the arm is moving.

Sequence (169 s, 48 steps): for each base-frame axis x, y (vertical), z, rx, ry, rz and sizes 5/10/20/40 mm or 2/4/8/16 deg,
both signs: hold the target offset 2 s, return to zero for 1.5 s. Worst case 40 N / 14 N m of spring. The joint guard stops the
run if any joint leaves the IK-predicted range + 5 deg. Keep the e-stop in hand.

```bash
# 1) print the plan only (no robot connection)
python step_test_thunder.py --config collection.step_test.json --friction off
# 2) compensation off
./run_realtime.sh python step_test_thunder.py --config collection.step_test.json --friction off \
    --execute --output ~/thunder_step_test/step_off.pt
# 3) compensation on (UR default scales viscous 0.9/0.9/0.8/0.9/0.9/0.9, coulomb 0.8/0.8/0.7/0.8/0.8/0.8)
./run_realtime.sh python step_test_thunder.py --config collection.step_test.json --friction ur_default \
    --execute --output ~/thunder_step_test/step_ur_default.pt
# 4) summary table (also writes <record>.steps.json)
python ../analyze_step_test.py ~/thunder_step_test/step_off.pt
```

Requires PolyScope >= 5.25.1 for the per-joint scales (Thunder: 5.26.1) and ur-rtde 1.6.5. Records with compensation on are
for comparison only; `records.validate_record` rejects them for fitting.
