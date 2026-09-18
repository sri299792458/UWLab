# Physical Thunder mounting correction

The robot workstation reported that Thunder's physical base cable opening faces
downward, while the Vention CAD opening faces horizontally. The previous server
check established agreement between CAD and simulation, not agreement with the
physical installation. The workstation's mesh-cutout and accelerometer evidence
resolved the mounting rotation.

The simulation root is `base_link`. Its corrected world quaternion, in `w,x,y,z`
order, is `[0.5, 0.5, 0.5, 0.5]`. Compared with `[0.7071068, 0, 0.7071068, 0]`,
this is +90 degrees about the previous local Z axis (world +X). The mounting
position remains `[0.177660, 0.377695, 1.466000]` meters.

The calibrated URDF's fixed `base_link` to controller `Base` conversion is still
180 degrees about Z. The calibrated joint transforms, controller conversion,
robot geometry and lab/table geometry are unchanged. Do not add another frame
conversion to compensate for the mounting correction.

## Regenerated data

The active server data directory is
`/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917`.
`UWLAB_DATA_ROOT` can select a relocated installation. The previous data and
training checkout are preserved separately.

The new map uses exact 20 mm spacing in world X, world Y and height above the
table, with 75 × 37 × 31 positions and the same 504 orientations: 43,356,600
targets. X samples end 5 mm before the physical table edge. Heights include
0 and 600 mm. IK search settings, fitted spheres, 50 mm column clearance,
1 mm other-world clearance and 50 mm cube-footprint edge clearance retain
their previous definitions.

All 33 lab collision boxes are freshly expressed in the corrected `base_link`
coordinates. Every map query is rerun; no previous world-space IK results or
full-arm reset banks are reused. The 591 object-relative grasps and 5,536
object-relative partial-assembly records are mounting-independent inputs.
`20_dense_reachability` retains the historical robot-relative sphere-fit inputs;
its old world-frame metadata is not the active map configuration. Use `49`'s atlas.
The task configuration rejects an atlas whose mounting pose differs from its
nominal robot root pose. Reset generation retains the original ±10 mm root-position
jitter; native checks use each recorded pose. The root orientation is fixed to the
corrected quaternion.

The complete map contains **13,259,706 collision-clear pose solutions** at **60,276
spatial positions**. The regenerated final reset banks contain **40,194 states**:
10,128 ordinary, 10,011 table-grasp recipe, 10,036 airborne-grasp and 10,019
partial-assembly states. Every retained state passed native geometry reload checks.
Three original rows failed the unchanged 50 mm column margin by approximately
4–24 micrometers and were excluded after independent full-joint forward-kinematics
and 256-placement reload checks. Raw banks, exact row mappings and reports are
preserved; no clearance threshold was relaxed.

## Validation and reproduction

`validate_mounting.py` reads the actual base pose and tabletop vertices back
from Isaac Sim, compares native body transforms with the exported joint graph,
checks that an old-mount atlas is rejected, and renders the scene. Its static
pilot pose illustrates the mount; it is not a collection-motion certificate.

The map is prepared with:

```bash
python scripts_v2/tools/curobo_umi/run_clearance_atlas.py --prepare --grid-spacing-m 0.02
```

Run one worker per GPU using `--worker INDEX --workers COUNT`, then pack and
validate all slices with `scripts_v2/tools/map_reset/pack_clearance_atlas.py`.
The preparation refuses to overwrite an existing map configuration. Its input
hashes and each slice's configuration hash identify the exact geometry and
mounting used for the computation.

The map remains a finite search over endpoint poses. Reset validation reloads
the resulting full-arm states in Isaac Sim and checks the actual body geometry.
The collection waveform receives separate dynamics and source-hull checks.
Robot-side route planning remains the workstation's responsibility.
