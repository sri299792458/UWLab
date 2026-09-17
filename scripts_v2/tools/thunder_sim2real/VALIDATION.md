# Validation and readiness

This package is ready to clone and configure on the robot workstation. No
physical Thunder recording or fitted Thunder dynamics profile exists yet.
The candidate keeps UW's collection gains and full excitation amplitudes;
the Thunder-specific start pose is 27 cm above the modeled table.

## Completed checks

- Eleven CPU tests pass: record/calibration/timing contracts, synthetic-data
  rejection, calibrated Jacobian comparison, mocked recorder success and abort
  paths, invalid starting-state rejection before creating a control interface,
  and generation of a quoted Stage-2 launch script without starting training.
- The isolated pinned `ur-rtde==1.6.2` wheel imports and exposes
  `directTorque(torque, friction_comp=True) -> bool`. The test instantiated no
  robot connection. Workstation/robot compatibility still requires checking
  the actual installed PolyScope version and controller availability.
- Native 500 Hz replay recovers a known two-step delay with the other dynamics
  fixed. The matched-delay joint-averaged error is approximately 0.000059
  degrees. The workstation and simulator wrist pose/Jacobian agree within
  the tolerances recorded in `validation_results/known_delay.json`.
- The optimizer completes a two-iteration/eight-environment synthetic smoke
  run, and its evaluation path writes metrics, replay arrays, and a joint plot.
  This checks execution and output generation, not a converged 25-parameter fit.
- The Stage-2 training environment resets and steps 64 environments at
  curriculum progress 0, 0.5, and 1. The evaluation task resets and steps at
  progress 1. These preserve the Thunder/UMI assets, current reset bank and
  probabilities, observations, rewards, 120 Hz physics and 10 Hz policy rate.
- The selected full-amplitude motion is checked at all 4,000 requested poses
  and 4,001 states in each of three simulated dynamics cases. All pass the
  source-hull checks, including the camera and 33 lab boxes. Minimum lab
  clearance is 93.7 mm, moving-arm self clearance is 15.0 mm, and the closest
  internal hand gap is 5.2 mm. Existing adjacency/mount exclusions and the
  map's 50 mm column requirement are retained.
- All 4,000 waveform samples match the pinned UW collector exactly.

Reports in `validation_results/` retain the original server artifact paths for
traceability. Their presence does not require those paths on the workstation.
The geometry checks are discrete samples in the modeled, cleared scene; the
three dynamics cases are assumptions rather than measured Thunder parameters.

## Workstation setup remaining

Copy `workstation/collection.simulation_candidate.json` and fill the robot IP,
installed PolyScope version, and actual mounted tool mass/center of gravity.
The pose, amplitudes, gains, and simulation-derived excursion stop limits are
already specified. Install the pinned dependencies and print the plan.
Position the robot at the selected configuration before starting collection;
this recorder checks the position and stationary state, and does not issue
the initial positioning move itself.

After real collection, validate the timestamps, fit the dynamics, and evaluate
against a separate real recording before using a selected profile for Stage 2.
The simulation evidence does not establish physical performance or uniqueness
of every fitted parameter. Controller tracking error is expected in this
experiment; the fitting objective is the actual recorded joint response.
