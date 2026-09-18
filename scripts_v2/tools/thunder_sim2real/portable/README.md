# Complete Thunder lab handoff

This release uses the physically corrected Thunder mounting quaternion `[0.5, 0.5, 0.5, 0.5]`
(`w,x,y,z`) and a newly computed **20 mm** reachability map. It supplies the current lab and
Thunder assets, calibration, cuRobo collision model, coverage data, and pinned software sources.
The optional full profile adds the complete joint lookup and freshly recorded reset banks.
Path planning and physical execution belong to the robot-system agent. The previous release
remains historical; its mounting and map must not be mixed with this release.

## Download the code and data

```bash
git clone --branch thunder-umi-sim2real https://github.com/sri299792458/UWLab.git
cd UWLab
python3.11 scripts_v2/tools/thunder_sim2real/portable/install_bundle.py \
  --destination "$HOME/thunder-lab-mount-v2"
source "$HOME/thunder-lab-mount-v2/activate.sh"
```

For an existing clone, switch to `thunder-umi-sim2real` and pull before running the installer.
Use Python 3.11.8 or later in the 3.11 series (the server uses 3.11.15).
By default, the installer downloads the **150 MB core** package (149,883,745 bytes).
It contains 3,053 files totaling 260,349,299 bytes before installation metadata. Exact sizes
and hashes are recorded in `release_manifest.json`. Allow 1 GB for core download/extraction,
plus space for Python environments. It includes the calibrated
robot, lab/collision models, map coverage, dependency sources and validation evidence.

**Only the core package is published in this release.** The full archive remains on the
development server. The instructions below describe that prepared archive and require it
to be transferred or published before `--profile full` can be used.

The **full** profile also includes every saved joint solution and reset tensor:
895,637,069 download bytes (896 MB), with 34,315 files totaling 1,859,058,012 bytes
(1.86 GB) before installation metadata. Its hashes are in `release_manifest.full.json`.
It is optional for robot-side planning. For complete simulation/reset workflows:

```bash
python3.11 scripts_v2/tools/thunder_sim2real/portable/install_bundle.py \
  --profile full --destination "$HOME/thunder-lab-mount-v2-full"
source "$HOME/thunder-lab-mount-v2-full/activate.sh"
```

Allow at least **5 GB** during full download/extraction, plus simulator/environment space.
Already downloaded parts are reused within a cache specific to the release tag.
The destination must be new; an existing installation is never overwritten.

All parts, the joined archive, and every extracted file have SHA-256 checksums.
After checking the original bytes, the installer changes server paths in text configurations
to the chosen installation directory. It keeps the original text in `.original-configs/`
and records both hashes in `relocation.json`. USD geometry, meshes, arrays and tensors keep
their original bytes. Embedded historical hashes in reports still refer to original inputs.
Source `activate.sh` in each new shell and after activating a Python environment.
To move the data later, install again at the new location so embedded paths are updated.

## Contents and entry points

Paths below are relative to `$HOME/thunder-lab-mount-v2` unless marked as repository files.

| Purpose | Location |
| --- | --- |
| Full Thunder arm, D405, mount and UMI hand | `assets/thunder_d405_umi_rigid_asset/` |
| Extracted UMI hand for grasp generation | `assets/thunder_d405_umi_gripper_asset/` |
| Thunder factory calibration and inertial metadata | Robot directory: `thunder_kinematics.yaml`, `metadata.yaml`, calibrated URDF and mass reports |
| Lab CAD and table | `assets/lab_vention_asset_v60/lab_vention.usd` |
| AprilCube60 pair and tag textures | `assets/aprilcube_60mm_rounded/` |
| Original convex robot meshes and calibrated joint tree | `data/15_table_reachability/model/` |
| Lab collision scene (33 oriented boxes) | `data/15_table_reachability/model/lab_scene.yml` |
| cuRobo robot model (239 spheres on 17 collision links) | `data/49_clearance_and_placement/atlas/robot_open_spheres.yml` |
| Map axes, frames, margins and masks | `data/49_clearance_and_placement/atlas/config.json` |
| Complete map and all per-height/orientation solutions | `data/49_clearance_and_placement/atlas/atlas.npz`, `atlas/slices/` (**full**) |
| Memory-mapped joint lookup | `data/49_clearance_and_placement/lookup/atlas_joint_seeds.npy` (**full**) |
| Current grasp, partial-assembly and four reset banks | `data/49_clearance_and_placement/OmniReset/` (**full**) |
| Selected motion's simulation trajectories, reports and preview | `data/53_thunder_sim2real/motion_preview/lower_full/` |
| Exact IsaacLab and cuRobo sources with licenses | `sources/IsaacLab/`, `sources/curobo/` |
| Applied IsaacLab modification and source commits | `sources/IsaacLab.patch`, `bundle_manifest.json` |
| Collection pose, waveform and real-workstation collector | Repository: `scripts_v2/tools/thunder_sim2real/workstation/` |
| cuRobo export, IK, map and hull-check tools | Repository: `scripts_v2/tools/curobo_umi/` |
| Current reset generation and packing tools | Repository: `scripts_v2/tools/map_reset/` |

The final bank contains **40,194 states**: 10,128 ordinary starts, 10,011 table-grasp
recipe starts, 10,036 airborne-grasp starts and 10,019 partial-assembly starts.
`dataset_audit.json` records their hashes and validation. Raw recordings and any explicit native-recheck exclusions are
preserved on the development server; the installed bank contains the validated final rows.
The map has **31 heights × 37 Y positions × 75 X positions × 504 orientations**:
**43,356,600 targets and 13,259,706 accepted pose solutions**. There are 60,276 spatial
positions with at least one solution. The six-joint lookup uses non-finite entries for missing
solutions. The map belongs to the corrected mounting and supersedes the previous 10 mm map.
The installer checks the complete archive, and `check_bundle.py` rejects map/bank mounting
orientations that differ from this checkout.

Historical scripts are included for inspection. Scripts named `run_sphere_atlas.py`,
`run_stage.py`, `prepare_lookup.py`, `audit_bank.py` and their older summarizers target
superseded datasets; use the **clearance** scripts and paths above for this handoff.
Historical logs, duplicate intermediate banks, superseded assets/maps, credentials, live
training checkpoints and installed simulator binaries are excluded. No trained policy or
real-robot dynamics fit is implied by the supplied reset datasets.

## Software setup

The server is Linux x86-64, Python 3.11.15, with NVIDIA RTX A6000 GPUs. Use separate
environments: the validated simulator uses NumPy 1.26 and Warp 1.13; the current cuRobo
environment uses NumPy 2.4 and Warp 1.17. `environment/versions.json` and the two freeze
inventories record the observed software. The full freeze files are provenance inventories,
not installation recipes: they contain unrelated packages and server editable paths.

### cuRobo, map inspection and source-hull checks

From the repository root, with Python 3.11 available:

```bash
python3.11 -m venv "$HOME/venvs/thunder-curobo"
source "$HOME/venvs/thunder-curobo/bin/activate"
source "$HOME/thunder-lab-mount-v2/activate.sh"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r scripts_v2/tools/thunder_sim2real/portable/environment/curobo-requirements.txt
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.8.0.post1.dev42 \
  python -m pip install --no-deps --no-build-isolation "$CUROBO_SOURCE"
python scripts_v2/tools/thunder_sim2real/portable/check_bundle.py --gpu
python scripts_v2/tools/curobo_umi/query_sphere_atlas.py \
  --x 0 --y 0 --height 0.30 --tilt 0
```

The cuRobo revision is `8e734f3ced1df898990bcd92de40abce475907db`, using its current
`curobo.kinematics`, `curobo.inverse_kinematics`, `curobo.collision_checking` and
`curobo.motion_planner` API. Older examples importing `curobo.wrap.reacher` target a
different API. The bundled source includes this version's examples and tests.
The focused requirements cover the exercised map/IK/collision workflows; optional cuRobo
features may need their extra dependencies. CUDA kernels compile on first use.

Without `--gpu`, `check_bundle.py` checks model references, USD loading, reset tensors
and the memory-mapped lookup when installed, and the collection endpoint's original convex hulls on CPU.
With `--gpu`, it also compares cuRobo forward kinematics to the calibrated joint tree,
constructs the cuRobo scene checker and tests the accepted sphere geometry on CUDA.
None of these commands opens a robot connection.

### Isaac Sim, replay, reset generation and training

Install the **full** data profile above for these workflows. Use the pinned IsaacLab source supplied in the bundle:
`3e73d6dd79080fd7632488c061052a6edd52e230` with the included launcher patch already applied.
The patch allows additional Kit arguments through `ISAACLAB_KIT_ARGS`.
Install Isaac Sim 5.1 from NVIDIA in a separate Python 3.11 environment; NVIDIA's simulator
license and hardware requirements apply. This bundle does not redistribute its binaries.

```bash
python3.11 -m venv "$HOME/venvs/thunder-isaac"
source "$HOME/venvs/thunder-isaac/bin/activate"
source "$HOME/thunder-lab-mount-v2-full/activate.sh"
python -m pip install --upgrade pip setuptools wheel toml
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaacsim[all,extscache]==5.1.0.0' \
  --extra-index-url https://pypi.nvidia.com
python -m pip install -c scripts_v2/tools/thunder_sim2real/portable/environment/isaac-constraints.txt \
  -e "$ISAACLAB_PATH/source/isaaclab" -e "$ISAACLAB_PATH/source/isaaclab_assets" \
  -e "$ISAACLAB_PATH/source/isaaclab_tasks" -e "$ISAACLAB_PATH/source/isaaclab_rl" \
  -e source/uwlab -e source/uwlab_assets -e source/uwlab_tasks -e source/uwlab_rl \
  'rsl-rl-lib @ git+https://github.com/UW-Lab/rsl_rl.git@959ccbc4712400a75efaa63cb827fd8939776825' \
  numpy==1.26.0 scipy==1.15.3 trimesh==4.5.1 warp-lang==1.13.0 PyYAML==6.0.2 \
  cmaes==0.13.0 python-fcl==0.7.0.11
export THUNDER_PYTHON="$HOME/venvs/thunder-isaac/bin/python"
bash scripts_v2/tools/thunder_sim2real/run_sim.sh \
  scripts_v2/tools/thunder_sim2real/validate_sim.py \
  --mode stage2eval --output "$HOME/thunder-stage2-smoke" --headless --device cuda:0
```

Isaac's first launch requests its license acceptance. The robot USDs retain their original
NVIDIA material references; rendering can require internet access for those materials.
All local physics geometry and cube textures are included. The bundle is not an air-gapped
simulator installer. Check the JSON validation report, not just the simulator process exit.
`stage2eval` uses a clearly labeled synthetic dynamics profile solely for the smoke test.
The real-record → dynamics fit → replay verification → fine-tuning procedure is in
[the parent README](../README.md).

The current reset pipeline uses `run_clearance_stage.py`, `run_clearance_atlas.py` and
`pack_clearance_atlas.py`. Keep the delivered inputs intact. Recompute into a separate
data root and prepare its configuration anew; the delivered slice/report hashes describe
the original generation, not a resumed generation with relocated or edited source files.
The current pilot and production drivers accept `--gpus`, for example
`run_clearance_production.py --gpus 0` on a single GPU. Direct `run_clearance_stage.py` accepts
`--gpu`, `--dataset-dir`, `--input-dir` and `--output-subdir`. Use
`audit_clearance_bank.py` for final structural/hash checks; combine disjoint native reports
with `merge_native_validation.py --reports REPORT...`. Both default to the current data root.

## Handoff to the robot-system agent

Start with [ROBOT_AGENT_HANDOFF.md](ROBOT_AGENT_HANDOFF.md). It defines the coordinate
frames, tool configuration and collision margins, and points to the selected collection pose.
The collector still requires actual robot IP, firmware version and physical payload values.
The simulation's payload reference is supplied separately and is not labeled a measurement.

## Delivery validation

The `validation/` reports identify the extracted file counts, bundle paths, corrected mount,
lookup dimensions and final bank counts. Geometry verification covers calibrated cuRobo
kinematics, scene construction, sphere clearance and original convex hulls. The native
Stage-2 smoke test uses 64 environments, 120 Hz simulation and 10 Hz policy rate with a
clearly labeled synthetic dynamics profile. It verifies reset/step integration, not a trained
policy. Simulator checks use the existing Isaac environment, not a new simulator binary
installation. The collector/data-contract suite has 11 checks.

Both profiles were installed separately and passed these checks. cuRobo forward-kinematics
position disagreement was below 0.06 micrometers. The cuRobo check imported each installation’s
bundled source; the native check imported the full installation’s bundled IsaacLab source.
The dedicated cuRobo environment was created during the previous delivery and reused here.

Reports are in `validation/`. The GPU tests validate the exercised geometry and APIs;
physical hardware, workstation-specific drivers and a planned route remain robot-side work.
The historical controller-sweep and inertia-ablation source tools are also included in the
repository for inspection; their old experimental output datasets are not runtime inputs
and are not part of this release.
