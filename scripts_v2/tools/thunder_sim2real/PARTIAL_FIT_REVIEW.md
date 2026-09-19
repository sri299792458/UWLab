# Fit the interrupted Thunder recording and design the next collection

The recording's measured prefix is usable for dynamics optimization. The original
file remains unchanged, including its incomplete status and stop information.

## Method

Input: `validation_results/hardware_20260919/20260919T163903086675Z/thunder-fit.pt`.
It contains 2,031 measured states, commanded targets, joint speeds, and the exact
controller settings used on the robot. Its SHA-256 is
`3d4a4ced946af45d706f8a53966b711e52e6c15f6b764b98869c472d693b9e8b`.

`fit_partial.py` runs UWLab's existing controller in the native Thunder Isaac
simulation. CMA-ES, an optimizer that samples parameters and concentrates its
search around better candidates, adjusts the existing 25 dynamics parameters:
six added joint inertias, six static-friction values, six sliding/static-friction
ratios, six viscous-friction values, and one discrete actuator delay.

The first 1,600 samples (3.2 nominal seconds) determine the parameters. The final
431 samples are held out: simulation continues through them without resetting
to measured states. This tests a prediction of the rise toward the hardware stop.
Samples are compared before the corresponding command, using their command index.

The objective is the sum over six joints of mean squared angle error plus
`0.05² × mean squared speed error`, in radians and radians/second. Thus a
20 degrees/second speed error has the same weight as a one-degree angle error.
Including speed error discourages models with small angle errors but spurious
rapid oscillations. The existing replay's default angle-only objective remains
available with zero velocity weight.

Search bounds are 0.0001–10 for added inertia, 0–20 for static friction, 0–1 for
sliding/static ratio, 0–60 for viscous friction, and 0–5 delay steps. A step is
2 ms. The search starts from half the reference added dynamics, then refines the
best result using a smooth logistic mapping into these bounds. Each generation
evaluates 128 candidates. The controller gains and torque limits remain fixed.

The local fitting/preview simulation uses a 1,000 rad/s speed cap, checked directly
in PhysX, so low asset caps cannot hide speed predictions. This setting applies
only to these offline simulation instances.

The eight best candidates by training score are replayed for the full proposed
collection. Their spread shows disagreement among fitted candidates; it is not
a statistical confidence interval. Collision distances use every saved 2 ms body
state and the source convex hulls against the accepted 33-box lab model. The
existing complete requested-path IK/hull check is reused only if its start pose,
controller, and waveform match exactly.

## Completed optimization

The broad search completed 10 generations, and local refinement completed 12:
2,816 candidate evaluations in total. The refined model's training angle RMSE is
0.648 degrees and speed RMSE is 3.72 degrees/second. On the reserved suffix, these
are 1.880 degrees and 12.95 degrees/second. The earlier half-reference model's
corresponding held-out errors were 3.196 degrees and 26.47 degrees/second.
The refined model predicts a held-out Wrist 1 peak of 222.7 degrees/second versus
the measured 205.9 degrees/second, reproducing an overspeed in the original motion.

The next collection configuration uses the **already checked eight broad-search
candidates**. They predict a maximum of 65.8 degrees/second on the half-amplitude,
0.1–1.5 Hz sweep and all pass the complete source-hull check. The refined fit is
retained as additional evidence; waiting for more optimization is not a prerequisite
for this provisional next collection. No physical execution occurred on the
simulation server.

## Collection sequence

1. Use this partial-run fit to predict an eight-second sweep at half amplitude
   and 0.1–1.5 Hz, with the same start pose and controller.
2. Collect that proposed physical sweep. Compare its measured angles and speeds
   against the saved predictions **before** using the new recording for fitting.
   This is a stronger test than another segment of the original recording.
3. Refit using both the original measured prefix and the new complete recording.
   Normalize each recording's loss by its sample count so the longer, slower run
   does not automatically overwhelm the earlier, faster motion.
4. Predict 75% amplitude at the same 0.1–1.5 Hz range. Increase amplitude on the
   robot only after the updated model predicts adequate speed margin. Evaluate
   full amplitude the same way; assess higher frequencies separately afterward.

The target is predictive accuracy across the intended robot motions. Returning
to the original full-amplitude, 0.1–3 Hz sweep is not a required milestone. That
combination already produced a measured speed-limit violation. A short single
trajectory also cannot establish unique physical values for all 25 parameters;
the fitted values are a provisional dynamics model for the next experiment.

## Artifacts

Experiment outputs are in
`/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917/74_partial_hardware_fit_20260919`.
The broad search, refinement, held-out evaluation, full-motion predictions, source
collision checks, and collection configuration are retained separately so the
recording, fitted model, and proposed physical motion can be traced to their inputs.
