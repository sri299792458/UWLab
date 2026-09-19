# Thunder / UMI state-policy sim-to-real

For the **complete portable lab, calibrated assets, cuRobo map and software setup**, start with
[portable/README.md](portable/README.md). The default download is the **150 MB core package**; the full
map/reset dataset is optional and remains prepared locally. The robot-side agent handoff is included there.
The physical mounting correction and rebuilt 20 mm map are documented in
[MOUNTING_CORRECTION.md](MOUNTING_CORRECTION.md). Use the matching corrected bundle.

This branch prepares UWLab's Stage-2 state-policy fine-tuning for Thunder and
provides a standalone package for the robot workstation. The current Stage-1
training run remains in its original checkout. No real-robot data has been
collected and no Stage-2 training has been launched by this implementation.

## What is implemented

| Component | Purpose |
| --- | --- |
| `workstation/collect_thunder.py` | Plan and collect a 500 Hz Cartesian frequency sweep using Thunder calibration and UWLab's torque controller. |
| `workstation/thunder_calibration.json` | Thunder's six calibrated joint transforms and arm inertials, copied from the current robot metadata. |
| `workstation/vendor/` | Unmodified NumPy controller/kinematics from WEIRDLabUW/diffusion_policy commit `3cd87c830b3a46967fb2291f5bc18e8d746ab4b6`, with provenance and MIT license. |
| `records.py` | Validate data shape, robot identity, controller settings, coordinate conventions, timestamps, and completion; export fitted parameters for Stage 2. |
| `replay.py` | Run closed-loop replay through UWLab's existing simulated controller, with explicit sample alignment and candidate motor delays. |
| `fit.py` | Optimize 25 dynamics parameters using UWLab's CMA-ES method, or evaluate a saved fit on another recording and plot joint trajectories. |
| `prepare_finetune.py` | Check a Stage-1 checkpoint and a real-data profile, then write a launch script and provenance record. It does not start training. |
| `run_sim.sh` | Select this checkout's Python packages without changing the shared Python installation. |
| `test_contract.py`, `validate_sim.py` | CPU validation and native simulation checks, including known-delay recovery and Stage-2 reset/step checks. |

The registered environments are:

- `OmniReset-Thunder-UMI-Sysid-v0`: 500 Hz free-space fitting with Thunder's
  calibrated robot, UMI hand, and unscaled Cartesian target corrections.
- `OmniReset-UMI-Defaults-State-Finetune-v0`: Stage-2 training with the current
  Thunder/UMI task and an explicit fitted dynamics profile.
- `OmniReset-UMI-Defaults-State-Finetune-Play-v0`: evaluation at the end of the
  fine-tuning curriculum.

These configurations live in
`source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/umi_sim2real_cfg.py`.
The Thunder dynamics event is in `mdp/thunder_sysid.py`. It uses an explicit
profile rather than altering the shared robot asset's `metadata.yaml`.

## Modeling choices

The controller formula remains UWLab's explicit Cartesian PD torque law:
`torque = J.T @ (Kp * pose_error - Kd * wrist_velocity)`, with
`Kd = 2 * sqrt(Kp) * damping_ratio` and joint torque limits. `J` maps joint
velocities to wrist motion. This controller has no learned gains and no added
inertial compensation. The real command uses `directTorque(...,
friction_comp=False)` as in the pinned collector. Robot gravity compensation
remains the robot controller's responsibility; the fitting robot retains
UWLab's disabled-gravity configuration.

Fitting uses a free-space robot with an open, empty hand. The real collector
does not command the hand; set that condition on the workstation. The
simulator holds it open. The full lab scene and task objects remain in
Stage-2 task training, where their contacts are relevant.

The frequencies follow UWLab:

| Loop | Frequency |
| --- | --- |
| Real collection/controller | 500 Hz |
| System-identification simulation/controller | 500 Hz |
| Stage-1 and Stage-2 simulation/controller | 120 Hz |
| State policy decisions | 10 Hz |

The 25 fitted parameters are six joint armatures (effective rotational
inertias), six static-friction values, six dynamic/static friction ratios,
six viscous-friction values, and one integer motor delay. CMA-ES searches the
same default bounds as UWLab: armature 0–10, static friction 0–20, ratios 0–1,
viscous friction 0–20, and delay 0–5 fitting steps. The upper bounds are CLI
options; inspect solutions at the bounds before interpreting a fit.

Stage 2 preserves the current 60 mm cube assets, fixed lower cube, 20–200 g
cube mass randomization, regenerated reset banks, four equally weighted
reset families, rewards, observations, policy architecture, and PPO defaults.
It adds the upstream curriculum for arm dynamics, controller gains, and
action scale. Gains begin at Stage 1's XYZ `Kp=500, Kd=160` and rotation
`Kp=60, Kd=0.1`, and move toward the actual collection controller settings.
The example collection settings are UWLab's `Kp=[1000,1000,1000,50,50,50]`,
with damping ratio 1. They are **proposed collection settings, not gains
validated on Thunder hardware**.

The action-scale endpoint stays UWLab's
`[0.01, 0.01, 0.002, 0.02, 0.02, 0.2]` in meters/radians per unit action.
Stage-2 dynamics/gain randomization retains the upstream 0.8–1.2 range and
success-driven curriculum. Evaluation uses that full randomization range.

Fitted delay is stored both in 500 Hz steps and in seconds. Stage-2 delay
randomization remains UWLab's independent 0–1 physics-step range at 120 Hz
(0–8.33 ms). The export explicitly records this choice; it does not copy a
500 Hz step count into a 120 Hz actuator. Compare the measured delay with
that range when reviewing the real fit.

## Sample alignment and its validation

The collector stores `joint_positions[t]` before computing/sending that row's
torque command. The replay compares it with the simulated position before
applying the corresponding command, then advances one 2 ms step. The first
recorded position and velocity initialize the simulation.

The pinned upstream replay instead compares the post-step position with that
row. This implementation also computes that comparison as a diagnostic; it
does not use it as the fitting objective. A native synthetic test generates
motion with known dynamics and a two-step delay, holds the other 24 dynamics
parameters fixed, and compares candidates with delays 0 through 5. This checks
sample indexing and delay application. It does not prove that all 25
parameters can be uniquely identified from one real trajectory, or quantify
the effect of the older convention on UWLab's real experiments.

Candidate parameters and delay lags are applied after environment reset and
settling. Delayed actuator resets clear their buffers and reset their lags,
so this order is required to test the intended candidate delay.

Real records contain robot timestamps before and after reading Q/Qd, plus host
sample and command times. The validator rejects duplicate/missing samples or
state reads spanning a robot update. There is no silent resampling. If the
workstation cannot maintain the recording cadence, address that before
fitting. Host and robot clock values are not assumed to share an epoch.

## Robot workstation: clone, install, and plan

Only the `workstation` directory and its dependencies are needed for collection.
The workstation does not need Isaac Sim, the reset dataset, robot USD assets,
or a separate diffusion_policy installation.

```bash
git clone --branch thunder-umi-sim2real git@github.com:sri299792458/UWLab.git UWLab-thunder
cd UWLab-thunder
python3.11 -m venv .venv-thunder
source .venv-thunder/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r scripts_v2/tools/thunder_sim2real/workstation/requirements.txt
cp scripts_v2/tools/thunder_sim2real/workstation/collection.outward_candidate.json /tmp/thunder-collection.json
```

Fill `/tmp/thunder-collection.json` with Thunder's IP, installed PolyScope
version, and configured tool payload mass and center of gravity.
The selected pose is shifted 15 cm outward from the columns at the same 27 cm
height. Its full UW motion amplitudes and
UW collection gains are already filled in. Preserve them to reproduce the
checked candidate. `collection.example.json` is a separate blank motion template.
The candidate's joint excursion stop limits are the largest deviations seen
across the three simulated responses and requested path, plus 5 degrees per
joint. They stop collection after excessive departure from the start pose;
they are not an online collision checker.
Angles are radians. Amplitudes are explicit XYZ meters followed by XYZ
axis-angle radians in `base_link`; there are no hidden per-axis multipliers.
The eight-second 0.1–3 Hz sweep has a two-second ramp-up and three-second
ramp-down. Commands are issued every 2 ms; the waveform uses UW's original
`linspace(0, duration, N)` phase grid and sample-based ramp construction.

The collector uses the `directTorque` interface expected by UWLab's pinned
`ur-rtde==1.6.2`; the robot must support that interface. Newer RTDE releases
have changed its signature, so the recorder checks the installed version.
The supplied payload fields are deliberately empty; they must describe the
actual mounted Thunder tool rather than UWLab's reference payload.

For comparison, the current simulation's open, empty tool has mass
**1.002607 kg** and center of gravity **[-0.00027272, 0.00714345, 0.06427591] m**
in the UR tool-mounting (`tool0`) frame. This sums the nine tool bodies and
includes the camera, printed mount, and modeled screws once. It excludes the
robot wrist/arm. The asset's mass report lists the separate flange adapter,
USB cable/ties, camera washers, and tape as unmodeled. This is a model-derived
reference, not a measurement of the complete mounted hardware; it does not
automatically fill the collection payload fields. The per-body calculation
and native-pose cross-check are in
[`workstation/simulation_payload_reference.json`](workstation/simulation_payload_reference.json).

Print the plan without connecting:

```bash
python scripts_v2/tools/thunder_sim2real/workstation/collect_thunder.py \
  --config /tmp/thunder-collection.json
```

With Thunder positioned at the configured start pose and the planned
excitation checked in the physical workspace, run collection:

```bash
python scripts_v2/tools/thunder_sim2real/workstation/collect_thunder.py \
  --config /tmp/thunder-collection.json \
  --output /path/to/data/thunder-fit.pt --execute
```

The script does not move to the start pose. It checks position tolerance and
stationary joints before enabling collection. Ctrl-C aborts; cleanup follows
the UWLab torque-to-hold handoff. Partial/error recordings are saved when
samples exist, marked incomplete, and rejected by the fitter. Existing output
files are not overwritten. Following UWLab's published procedure, use this same
recording for fitting and for the subsequent simulated-versus-real overlay.
A separate held-out trajectory is not required by that procedure.

Transfer the `.pt` recordings to the simulation server. They include the
controller gains, torque limits, payload configuration, calibration, initial
state, targets, joint positions/velocities, commanded torques, and timing.

## Simulation server: validate, fit, evaluate, and export

Activate the existing Isaac Sim/UWLab environment. These commands run from
this checkout's root. The simulation assets and reset dataset remain external
files at the paths used by the current Stage-1 run.
Those server assets are not needed on the robot workstation and are not bundled
in the Git repository.

```bash
python scripts_v2/tools/thunder_sim2real/records.py validate /path/to/thunder-fit.pt
CUDA_VISIBLE_DEVICES=0 bash scripts_v2/tools/thunder_sim2real/run_sim.sh \
  scripts_v2/tools/thunder_sim2real/fit.py --headless --device cuda:0 \
  --record /path/to/thunder-fit.pt --output /path/to/new-fit-directory \
  --num_envs 512 --iterations 200
```

Use an available GPU; the active Stage-1 run currently uses physical GPUs
3, 4, 5, and 7. The fitted result is `best_fit.json`, updated after each
completed optimizer iteration. It records source hashes, controller settings,
search bounds, parameter values, delay, and optimization history.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts_v2/tools/thunder_sim2real/run_sim.sh \
  scripts_v2/tools/thunder_sim2real/fit.py --headless --device cuda:0 \
  --record /path/to/thunder-fit.pt \
  --evaluate /path/to/new-fit-directory/best_fit.json \
  --output /path/to/new-evaluation-directory
```

Evaluation writes `evaluation.json`, `replay.npz`, and `joint_overlay.png`.
Inspect all six joint errors and the plotted trajectories. UWLab's guide
suggests less than 2 degrees RMS error per joint as a fit check; that is a
reference criterion, not evidence of successful Thunder task transfer.
UWLab then calls for teleoperating the physical robot to verify sensible motion
before proceeding to fine-tuning. See their linked guide for that robot-side step.
The report identifies whether the evaluated recording differs from the
fitting recording.

```bash
python scripts_v2/tools/thunder_sim2real/records.py export-profile \
  --fit /path/to/new-fit-directory/best_fit.json \
  --output /path/to/sysid_profile.json
```

This writes a candidate profile without modifying shared robot metadata.
Synthetic records/profiles are rejected by the production fitting/export and
Stage-2 loading paths.

## Prepare Stage 2 after selecting a Stage-1 checkpoint

Select a successful checkpoint using task evaluation and UWLab's action-noise
robustness evaluation. Geometry and calibration must match this branch.
Then choose the Stage-2 iteration budget and available GPUs explicitly:

```bash
python scripts_v2/tools/thunder_sim2real/prepare_finetune.py \
  --checkpoint /path/to/selected-stage1-model.pt \
  --profile /path/to/sysid_profile.json --output /path/to/new-stage2-launch \
  --gpus 3,4,5,7 --num_envs 16384 --iterations 40000 --logger wandb
```

The iteration count above is an example budget. The helper writes `launch.sh`,
a copy of the dynamics profile, and `launch_metadata.json` with hashes and the
exact command. It does not launch. Run the resulting script after reviewing
the selected checkpoint, measured profile, fit evaluation, and resource budget.
W&B authentication/entity configuration remains the server's existing setup.

For Stage-2 evaluation:

```bash
THUNDER_SYSID_PROFILE=/path/to/sysid_profile.json CUDA_VISIBLE_DEVICES=0 \
  bash scripts_v2/tools/thunder_sim2real/run_sim.sh \
  scripts/reinforcement_learning/rsl_rl/play.py --headless \
  --task OmniReset-UMI-Defaults-State-Finetune-Play-v0 \
  --num_envs 256 --checkpoint /path/to/stage2-model.pt
```

## Reproduce implementation checks

```bash
python scripts_v2/tools/thunder_sim2real/test_contract.py
CUDA_VISIBLE_DEVICES=0 bash scripts_v2/tools/thunder_sim2real/run_sim.sh \
  scripts_v2/tools/thunder_sim2real/validate_sim.py --mode replay --headless \
  --device cuda:0 --output /tmp/thunder-replay-check
CUDA_VISIBLE_DEVICES=0 bash scripts_v2/tools/thunder_sim2real/run_sim.sh \
  scripts_v2/tools/thunder_sim2real/validate_sim.py --mode stage2 --headless \
  --device cuda:0 --output /tmp/thunder-stage2-check
CUDA_VISIBLE_DEVICES=0 bash scripts_v2/tools/thunder_sim2real/run_sim.sh \
  scripts_v2/tools/thunder_sim2real/validate_sim.py --mode stage2eval --headless \
  --device cuda:0 --output /tmp/thunder-stage2-eval-check
```

Native tests write explicit PASS/FAIL reports. Check those reports, not just
the process exit code, because simulator shutdown can obscure Python errors.
The native Stage-2 check injects synthetic parameters only inside its test
process; it does not create an admissible production profile.

The committed `validation_results/` directory contains the native replay,
Stage-2 reset/step, and selected-motion reports. See `VALIDATION.md` for their
scope. These are simulation/software checks; they do not claim a real Thunder
recording, a fitted Thunder dynamics profile, or successful physical transfer.

## Sources

- [UWLab system identification and fine-tuning guide](https://uw-lab.github.io/UWLab/main/source/publications/omnireset/sim2real.html)
- [Pinned UWLab real-robot collector](https://github.com/WEIRDLabUW/diffusion_policy/blob/3cd87c830b3a46967fb2291f5bc18e8d746ab4b6/scripts/sim2real/collect_sysid_data.py)
- [Pinned UWLab robot environment](https://github.com/WEIRDLabUW/diffusion_policy/blob/3cd87c830b3a46967fb2291f5bc18e8d746ab4b6/conda_environment_real.yaml)
# Current motion candidate — 15 cm outward, unchanged UW excitation

Use `workstation/collection.outward_candidate.json`. The compatibility filename
`collection.simulation_candidate.json` contains the same candidate. The grasp center
moves 15 cm outward along world -Y, keeping its height 27 cm above the table and
its orientation unchanged. The corrected mounting remains `[0.5, 0.5, 0.5, 0.5]`.
Starting joint angles are `[-171.50302, -30.67153, 73.26556, 42.38637, 60.00074, 69.07879]` degrees.

The eight-second 0.1–3 Hz waveform, full amplitudes and collection gains remain exactly
UW's. All 4,000 six-axis offset samples are identical to the pinned collector.
The 4,000 requested-path poses and 4,001 states in each of three simulated responses
pass the source-hull checks. Minimum modeled column clearance is 121.229 mm;
minimum distance to other lab geometry is 93.729 mm. Minimum nonadjacent moving-arm
separation is 12.699 mm on the requested path and 14.446 mm across simulated responses.

The [outward-pose handoff](OUTWARD_POSE_HANDOFF.md) links the video, raw trajectories,
checksummed download and UW-versus-outward joint-range comparison. The selected video
shows the UW-reference-dynamics case with 4 ms delay. The UW joint comparison is
calculated ideal tracking with UW's default pose and calibration, not UW hardware data.
Collision clearance does not establish good identification data: in the shown simulation,
shoulder pan travels only 5.88 degrees. The zero-added-friction/inertia scenario has
only 0.54 degrees of wrist-3 travel despite high speed. Real recording and fit verification
remain necessary; the simulated scenarios are not measured Thunder dynamics.

The existing corrected-mount core bundle remains compatible. Its `lower_full` motion
is historical; download the supplemental outward-pose evidence linked in the handoff.
No physical robot connection or motion command was made for this update.
