# UMI training handoff for the MSI agent

This branch supplies the corrected physical mounting, full reset bank, robot/lab assets,
and both stock-hand gain experiments currently running on the development server.
The user will discuss hypotheses in the existing task and run selected experiments on
MSI. The MSI agent owns environment installation and scheduler setup. No MSI job has
been submitted by this delivery.

Start from branch **`codex/msi-training-handoff-20260918`** of
<https://github.com/sri299792458/UWLab>. The data release is **`umi-training-20260918`**.
Use a separate checkout for new experiment changes so the delivered reference stays available.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --single-branch \
  --branch codex/msi-training-handoff-20260918 \
  https://github.com/sri299792458/UWLab.git UWLab-msi
cd UWLab-msi
```

The delivered tasks load their large assets from the release, so this checkout does not
need to download the repository's unrelated Git LFS media. The tiny stock USD layer is
ordinary Git text and is included even when LFS downloads are skipped.

## Storage first: install the simulator in project space

The user's MSI home is already at its 200 GB quota. Clearing a small cache may restore
Codex, but the simulator stack needs a separate storage plan. MSI documents independent
home and project quotas. Before installation, run `echo "$SHARED"` and `groupquota` to
identify the primary project's shared directory and its free allocation; these do not
recursively scan the home directory. See [MSI quota guidance](https://msi.umn.edu/storage/data-storage-faqs/how-can-i-check-my-storage-quota).

Use a per-user project directory such as `$SHARED/$USER/umi-training` for the repository,
Python/Conda environment, package caches, data and run outputs:

```text
umi-training/
  repo/
  envs/isaac51/
  cache/pip/
  cache/conda/
  cache/runtime/
  data/
  runs/
```

NVIDIA lists 50 GB as the minimum storage for Isaac Sim 5.1. Budgeting roughly **100 GB
free for initial software, downloads and caches**, with additional space for accumulated
checkpoints/videos, is a planning allowance rather than a measured MSI requirement.
The released task data is only ~896 MB compressed/~1.86 GB unpacked; it is not the
simulator installation. [NVIDIA requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

The linked generic environment recipe uses `$HOME/venvs`; **replace that with the project
environment path above on MSI**. Before invoking pip/Conda, the MSI agent should direct
their package caches to project storage using `PIP_CACHE_DIR` and `CONDA_PKGS_DIRS`.
Configure simulator/Kit, W&B, and container caches in project or allocated job-local
storage too, and inspect the actual paths at first launch. Do not assume moving only
the environment moves all caches. Keep home-directory configuration/authentication
separate; do not redefine `HOME` to point at the project.

Global scratch is useful for temporary staging, but MSI deletes data older than 30 days
and provides no backups there. Retain environments, required inputs and checkpoints in
project storage with the group's retention plan. If that project is also full, the MSI
agent/user must choose another authorized project or arrange capacity with the PI/MSI
before installing. [MSI storage locations](https://userdocs.msi.umn.edu/storage/storage.html)

## Download the complete training inputs

From this repository root, using Python 3.11 or newer:

```bash
python scripts_v2/tools/training_handoff/install.py \
  --destination /YOUR/PERSISTENT/STORAGE/umi-training-data
source /YOUR/PERSISTENT/STORAGE/umi-training-data/activate.sh
```

Replace the destination with a path on MSI. The installer refuses an existing destination.
Use project storage with available quota. By default its download cache is created beside
the destination under `.uwlab-downloads`, rather than consuming the home-directory quota.
Allow at least **4 GB free** for the download, temporary archive and extraction, in addition
to the separately installed simulator environment and future training outputs.
It downloads the full **895,637,069-byte archive (~896 MB decimal, ~854 MiB)** and verifies
its SHA-256 plus every extracted file. The full bundle contains **34,315 files, 1,859,058,012
uncompressed bytes**, before small manifests and the stock-hand layer. It includes:

- Corrected 20 mm map, orientation lookup and all **40,194** final reset states.
- The 591 object-relative grasp inputs and 5,536 partial-assembly inputs used to build that bank.
- Calibrated UMI robot, finger collision geometry, cubes, lab geometry and cached task assets.
- Exported kinematics/collision models, source meshes, map/reset audit records and generation tools.
- Exact IsaacLab source with its existing launcher patch and exact cuRobo source, with licenses.
- The portable stock-hand mass-property overlay and its input/build provenance, added from Git.

The installer relocates configuration paths to the chosen directory. It retains original
configuration bytes and records path substitutions in `relocation.json`. Geometry and reset
tensors keep their original hashes. The stock overlay changes only its base-USD reference
to a relative path; the 36 authored mass-property values are unchanged. The original
`build_report.json` remains historical provenance with development-machine paths.

This is a complete input package for the current corrected training tasks. Simulator binaries,
credentials, historical duplicate banks, live training logs and checkpoints are not in it.
Training starts fresh by default. Reference W&B runs and their exact configurations are included
under [`scripts_v2/tools/training_handoff/`](scripts_v2/tools/training_handoff/).

## Environment contract

Use Python 3.11, Isaac Sim **5.1.0.0**, Torch **2.7.0+cu128**, NumPy **1.26.0**,
Warp **1.13.0**, the bundled IsaacLab revision **`3e73d6dd79080fd7632488c061052a6edd52e230`**
with its supplied patch, and UW-Lab RSL-RL at **`959ccbc4712400a75efaa63cb827fd8939776825`**.
The [existing installation recipe](scripts_v2/tools/thunder_sim2real/portable/README.md#isaac-sim-replay-reset-generation-and-training)
and [constraints](scripts_v2/tools/thunder_sim2real/portable/environment/isaac-constraints.txt)
record the simulator stack. The recipe's old full-bundle transfer limitation does not apply
to this training release: use the installer above. The freeze inventories are provenance,
not a portable requirements file. cuRobo uses a separate environment and is not needed to
train from the delivered bank.

The MSI agent should check local GPU/driver/container compatibility. NVIDIA's
[Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
exclude A100/H100 because they lack RT cores. MSI lists A40/L40S options in its
[GPU partitions](https://userdocs.msi.umn.edu/compute/shared_partitions.html).
No particular MSI node, driver or container setup is claimed tested here. Keep the supplied
simulator version unless a version change is deliberately recorded as another experimental factor.

## W&B: same project, new run IDs

Use:

```text
entity:  srinivas299792458-university-of-minnesota
project: uwlab-lab-cube-stack
```

Authenticate on MSI with the user's own W&B access, for example `wandb login`. No API key is
included. The run preparer sets the entity/project and creates a unique ID for every new
experiment. Do not reuse either reference run's ID. `--offline` supports later `wandb sync`
if compute nodes cannot reach W&B; keep the run directory on persistent storage.

## Prepare and validate an experiment

After activating the simulator Python environment:

```bash
python scripts_v2/tools/training_handoff/verify_inputs.py \
  --bundle /YOUR/PERSISTENT/STORAGE/umi-training-data \
  --output /YOUR/PERSISTENT/STORAGE/input-verification.json

python scripts_v2/tools/training_handoff/prepare_run.py \
  --bundle /YOUR/PERSISTENT/STORAGE/umi-training-data \
  --output /YOUR/PERSISTENT/STORAGE/runs/stock-current-seed42 \
  --condition stock-current --nproc 4 --envs-per-rank 16384 --seed 42
```

This creates commands and metadata; it does not launch training. Available conditions:

| Condition | Nominal hand mass/COM/inertia | Arm gains | Task |
| --- | --- | --- | --- |
| `stock-may` | Stock nine-body hand properties | May | `OmniReset-UMI-MayHandDynamics-State-Train-v0` |
| `stock-current` | Same stock hand properties | Current UMI | `OmniReset-UMI-StockHandCurrentGains-State-Train-v0` |
| `umi-current` | Physical UMI model | Current UMI | `OmniReset-UMI-Defaults-State-Train-v0` |

All three retain UMI finger geometry, collision shapes, joints and linkage. Stock refers to
the mass properties, not a replacement gripper mesh. `umi-current` is supplied as the
corrected physical-UMI reference for future comparisons; it is not a third training job
launched by this handoff. The fixed stock task audits intentionally reject arbitrary gain
changes: define and document a new task/audit expectation for a new gain hypothesis.

On an allocated compatible GPU, run the generated `validate.sh`, then inspect
`validation/report.json` **and** `validation/mount/rank_0.json` for `PASS`. A simulator process
exit alone is insufficient. The check uses 256 environments and 20 ordinary policy steps;
it verifies the actual post-reset mounting, finite execution and, for stock tasks, the native
mass/COM/inertia and controller gains. It does not optimize a policy.

Then run the generated `launch.sh` under the site's chosen scheduler allocation. It inherits
`CUDA_VISIBLE_DEVICES` and does not overwrite the scheduler's GPU assignment. Each torchrun
worker uses its local device index. The launcher writes `train.log`, `train.exit`, and
`logs/rsl_rl/.../{params,model_*.pt,events...}` inside the chosen run directory. Both stock
tasks also write per-worker `native_training/` and `mount_training/` reports.
Commit source changes before preparing a real experiment; the preparer records the commit,
source archive, tracked patch, full command, input hashes and settings. It never stores W&B credentials.

For a small end-to-end installation test, prepare a separate output with
`--nproc 1 --envs-per-rank 64 --max-iterations 3 --logger tensorboard`. Such a test is an
installation check and is not comparable to the full-size learning runs.

An explicit `--resume-path /path/to/model_N.pt` loads that checkpoint and creates a new W&B
run with its parent checkpoint hash recorded. Here `--max-iterations` is passed to the
upstream runner as the number of additional updates after loading. Checkpoint resumes
restore policy/optimizer state, not the complete simulator/RNG trajectory. There is no
automatic preemption/requeue behavior in these scripts; the MSI agent can build that around
the explicit resume operation if required.

## Preserve the comparison

The reference pair shares seed 42 (rank seeds 42–45), four workers × 16,384 environments,
32 rollout steps, **2,097,152 samples per update**, 120 Hz physics/torque feedback, 10 Hz
policy actions, and 40,000 configured updates. The mounting is quaternion
**`[0.5, 0.5, 0.5, 0.5]` in w,x,y,z order**, nominal position
`[0.177660, 0.377695, 1.466000]` m, with the existing ±1 cm XYZ reset jitter.
Reaching/NearObject/Grasped/NearGoal counts are **10,128 / 10,011 / 10,036 / 10,019**,
with equal sampling probabilities. Rewards, observations, original termination rules,
fixed lower cube, randomization and PPO remain inherited from the recorded reference.

| Arm gains | Translation Kp / Kd | Rotation Kp / Kd |
| --- | --- | --- |
| May | 200 / 84.8528137424 | 3 / 3.46410161514 |
| Current UMI | 500 / 160 | 60 / 0.1 |

Kp sets the response to position/orientation error; Kd sets the response to velocity.
The controller computes `Kd = 2 * sqrt(Kp) * motion_damping_ratio`.

Changing GPU count or environments per worker changes the rollout batch and optimization
conditions. Record that as a separate factor instead of calling it an identical reproduction.
The old failed runs used a different mount and bank: comparing them directly to new runs
does not isolate inertia or gains. For root-cause tests, keep the corrected bank/mount fixed,
change one stated factor, record its expected mechanism, and compare at matched sample counts
with repeated seeds before making a learning claim.

Current corrected reference runs:

- [Stock + May gains](https://wandb.ai/srinivas299792458-university-of-minnesota/uwlab-lab-cube-stack/runs/eab459b8)
- [Stock + current UMI gains](https://wandb.ai/srinivas299792458-university-of-minnesota/uwlab-lab-cube-stack/runs/8b0d44e5)

[`reference_runs.json`](scripts_v2/tools/training_handoff/reference_runs.json) records their
source/input identities and verified startup status. The two development-server runs continue
independently of this delivery. Return new W&B URLs, commit, hypothesis, command, startup
reports and any setup deviations to the existing discussion task so subsequent experiments
can be selected using the same evidence.

## Delivery checks

[Validation reports](scripts_v2/tools/training_handoff/validation/summary.json) record a
fresh relocation of the complete archive, all input/asset checks, and 256-environment,
20-step native checks for each of the three conditions. The two stock conditions match
the reference runtime configurations after accounting for paths and validation size/device.
Masses, full inertias, COM positions, gains, action scale and timing match the native
reference values; principal-axis quaternion rounding differs by at most 1.47e-7.

A separate two-worker × 64-environment test completed three training updates, with all
40 scalar series and saved model tensors finite. It used TensorBoard and created no W&B
experiment. These tests used the existing local simulator installation and the relocated
bundle's IsaacLab source. They do not establish a fresh MSI installation or training success.
