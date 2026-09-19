# Outward collection speed violation — 2026-09-19

The outward-pose run `20260919T163903086675Z` saved 2,031 of 4,000 samples, spanning 4.064 s of robot time, then reported `directTorque returned failure`. The user-supplied pendant photo reports C283A7 for Wrist 1 exceeding the configured safety speed limit. Its peak saved velocity was 3.593506 rad/s (205.893 deg/s), at the final saved sample. The failed command and subsequent braking state are not in this recording.

The user supplied a second pendant photo showing all six joints set to 191 deg/s in both Normal and Reduced mode, with a displayed -11 deg/s margin annotation. The recorded Wrist 1 peak exceeded even the displayed setting by 14.893 deg/s. The photo identifies an overspeed stop; it does not establish a collision or a payload mismatch. The recorded configuration retains 1.04 kg and CoG [-0.002, 0.001, 0.047] m. No recorded torque reached its configured clipping limit, and all saved positions were inside the configured excursion guards. Cleanup recorded no exceptions.

The simulated Wrist 1 peaks were 3.127152, 2.405484, and 2.277676 rad/s. Its asset configuration applies a `velocity_limit_sim` of 3.1415 rad/s to the wrists and 1.5708 rad/s to the first three joints. That physics constraint differs from the hardware collector, which limits torque and joint excursion but contains no runtime joint-speed guard. Thus the modeled collision pass did not validate physical speed compliance, and these three dynamics cases did not bound the actual response.

The record is rejected by the validator because `completed` is false. It also has 45 repeated robot timestamps, 47 approximately 4 ms gaps, and four robot updates crossing the sequential state reads. Those violate the fixed 500 Hz fitting contract independently of the stop. Preserve this data for diagnosis; do not pass it off as a completed fitting run.

Do not repeat the unchanged sweep. The next step is to revise and validate excitation and speed handling against the observed safety settings, with operating margin, and correct sample synchronization. Leave safety limits intact. UR documents that direct torque uses one timestep regardless of speed scaling, so reducing the pendant speed slider is not a substitute for correcting this command waveform/controller.

Detailed data: [offline-safety-stop-review.json](offline-safety-stop-review.json). Pendant evidence: [pendant-safety-message.png](pendant-safety-message.png).

Sources: [UR C283](https://www.universal-robots.com/manuals/EN/HTML/SW5_26/Content/prod-err-codes/topics/CODE_283.html), [UR direct_torque](https://www.universal-robots.com/manuals/EN/HTML/SW5_26/Content/prod-scriptmanual/all_scripts/direct_torque.htm).

Joint-speed settings evidence: [pendant-joint-speed-settings.png](pendant-joint-speed-settings.png).
