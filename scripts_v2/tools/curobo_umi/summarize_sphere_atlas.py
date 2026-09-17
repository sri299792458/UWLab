"""Check every saved slice, aggregate orientation coverage, and render results."""
import base64
import gzip
import hashlib
import json
import importlib.metadata
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from reachability_geometry import ROOT

out=ROOT/'21_sphere_reachability'
cfg=json.loads((out/'config.json').read_text())
sha=hashlib.sha256((out/'config.json').read_bytes()).hexdigest()
H,O,Y,X=cfg['shape']
status=np.zeros((H,O,Y,X),np.uint8)
counts=np.zeros((8,H,Y,X),np.uint16)
best_q=np.full((H,Y,X,6),np.nan,np.float32)
best_o=np.full((H,Y,X),-1,np.int16)
max_p=max_r=0.;totals={str(k):0 for k in [2,3,4]};slice_seconds=[]
limits=np.asarray(cfg['joint_limits_rad'])
for h in range(H):
    for o in range(O):
        path=out/'slices'/f'h{h:02d}_o{o:03d}'
        meta=json.loads(path.with_suffix('.json').read_text());assert meta['config_sha256']==sha
        with np.load(path.with_suffix('.npz')) as data:
            s=data['status'];assert s.shape==(Y,X) and np.isin(s,[2,3,4]).all()
            status[h,o]=s;good=s==4;q=data['chosen_q']
            assert np.isfinite(q[good]).all() and np.isnan(q[~good]).all()
            if good.any():
                assert (q[good]>=limits[0]-2e-6).all() and (q[good]<=limits[1]+2e-6).all()
                max_p=max(max_p,float(data['position_error_m'][good].max()))
                max_r=max(max_r,float(data['rotation_error_rad'][good].max()))
            assert (data['clear_solution_count'][good]>0).all()
            assert (data['clear_solution_count'][~good]==0).all()
            for key in totals:
                n=int((s==int(key)).sum());assert n==meta['status_counts'][key];totals[key]+=n
            counts[0,h]+=good
            counts[1+cfg['orientations'][o]['tilt_deg']//30,h]+=good
            first=good&(best_o[h]<0);best_q[h,first]=q[first];best_o[h,first]=o
        slice_seconds.append(meta['elapsed_s'])
    print('HEIGHT',h+1,'/',H,flush=True)
assert sum(totals.values())==cfg['total_targets']
assert max_p<=cfg['ik_position_tolerance_m']+2e-6 and max_r<=cfg['ik_rotation_tolerance_rad']+2e-6
assert np.array_equal(counts[0],counts[1:].sum(axis=0))
np.savez_compressed(out/'atlas.npz',status=status,coverage_counts=counts,
    best_q=best_q,best_orientation_index=best_o,x_world_m=cfg['x_world_m'],y_world_m=cfg['y_world_m'],
    height_above_table_m=cfg['height_above_table_m'])
per_height=[]
for h,z in enumerate(cfg['height_above_table_m']):
    per_height.append(dict(height_above_table_m=z,positions_with_any_solution=int((counts[0,h]>0).sum()),
        positions_total=Y*X,sphere_clear_poses=int(counts[0,h].sum()),
        top_down_positions_with_any_solution=int((counts[1,h]>0).sum())))
workers=[json.loads((out/'slices'/f'worker_{i}_progress.json').read_text()) for i in range(4)]
assert all(w['complete'] for w in workers)
summary=dict(complete=True,total_targets=cfg['total_targets'],shape=cfg['shape'],status_counts=totals,
    positions_total=H*Y*X,positions_with_any_solution=int((counts[0]>0).sum()),
    max_position_error_m=max_p,max_rotation_error_rad=max_r,heights=per_height,
    longest_worker_elapsed_s=max(w['elapsed_s'] for w in workers),
    summed_ik_s=sum(w['ik_s'] for w in workers),summed_collision_s=sum(w['collision_s'] for w in workers),
    config_sha256=sha,known_comparison=cfg['known_comparison'],modeling_notes=cfg['modeling_notes'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

xs=np.asarray(cfg['x_world_m']);ys=np.asarray(cfg['y_world_m'])
xe=np.r_[xs[0],(xs[1:]+xs[:-1])/2,xs[-1]];ye=np.r_[ys[0],(ys[1:]+ys[:-1])/2,ys[-1]]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
for ax,h in zip(axes.flat,[0,4,9,14,19,29]):
    mesh=ax.pcolormesh(xe,ye,counts[0,h]/O*100,cmap='Blues',vmin=0,vmax=100,rasterized=True)
    ax.set_aspect('equal');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)')
    ax.set_title(f'{cfg["height_above_table_m"][h]*100:.0f} cm above table · {(counts[0,h]>0).sum():,}/{Y*X:,} positions')
fig.colorbar(mesh,ax=axes.ravel().tolist(),label='Sampled orientations with sphere-clear IK (%)',shrink=.8)
fig.suptitle('Full-table reachability · same 504 orientations at every point',fontsize=15)
fig.savefig(out/'reachability_height_slices.png',dpi=170)
fig.savefig(out/'reachability_height_slices.pdf')
plt.close(fig)
payload=dict(x=cfg['x_world_m'],y=cfg['y_world_m'],heights=cfg['height_above_table_m'],
    orientation_counts=[504,12,96,96,96,96,96,12],
    counts_gzip_base64=base64.b64encode(gzip.compress(counts.astype('<u2').tobytes())).decode())
template=Path(__file__).with_name('sphere_atlas_view.html').read_text()
html=template.replace('ATLAS_DATA_PLACEHOLDER',json.dumps(payload,separators=(',',':')))
assert len(html.encode())<1_000_000 and 'ATLAS_DATA_PLACEHOLDER' not in html
visual=Path('/home/kanth042/.codex/visualizations/2026/09/08/01a0820a-4d51-74f0-93ad-6437d2e8935c/umi-sphere-reachability.html')
visual.write_text(html)
(out/'visualization_path.txt').write_text(str(visual)+'\n')
versions={name:importlib.metadata.version(name) for name in ['nvidia-curobo','torch','warp-lang','numpy','scipy']}
versions['curobo_source_commit']=subprocess.check_output(['git','-C',str(ROOT/'15_table_reachability/curobo'),'rev-parse','HEAD'],text=True).strip()
(out/'software_versions.json').write_text(json.dumps(versions,indent=2)+'\n')
height_rows='\n'.join(f'| {100*r["height_above_table_m"]:.0f} | {r["positions_with_any_solution"]:,} | {r["top_down_positions_with_any_solution"]:,} | {r["sphere_clear_poses"]:,} |' for r in per_height)
scripts=Path(__file__).parent
report=f'''# Full-table sphere reachability map

Completed all **{cfg['total_targets']:,} sampled poses** using the sphere model accepted on 2026-09-11.
**{totals['4']:,} poses** have a sphere-clear inverse-kinematics solution. Across the {H*Y*X:,}
sampled XYZ positions, **{summary['positions_with_any_solution']:,}** have at least one accepted orientation.
Inverse kinematics (IK) finds joint angles that place the hand at a requested position and orientation.

## Scope and sampling

- The full physical tabletop: X [{xs[0]:.9f}, {xs[-1]:.9f}] m and Y [{ys[0]:.9f}, {ys[-1]:.9f}] m in world coordinates.
- Tabletop Z: {cfg['tabletop_z_m']:.9f} m. Hand-reference heights: 2, 4, …, 60 cm above it.
- Uniform axes: 76 X samples, 37 Y samples, 30 heights. Actual spacing is {(xs[1]-xs[0])*1000:.2f} mm in X, 20 mm in Y, and 20 mm in Z; boundary samples are included.
- Identical 504 orientation samples at every XYZ position. Tilt from downward is 0°, 30°, …, 180°; non-pole azimuth is sampled every 45° and roll every 30°. Each pole has 12 unique orientations; each other tilt has 96.
- Open empty hand, all six finger coordinates fixed at zero. No cubes, payload, stacking targets, rewards, policy, or controller gains enter the classification.
- Target frame: `{cfg['tool_frame']}`. Its exported offset and robot-to-world transform are frozen in `config.json`.

## Computation and acceptance

1. Transform every world-frame target into the robot base frame.
2. Run cuRobo batched IK with 32 seeds per target, a 1 mm position tolerance, and a 0.01 rad orientation tolerance. Returned seeds may converge to duplicate configurations.
3. Transform the accepted 239 fitted spheres to each candidate's joint configuration on the GPU. Add 1 mm to each radius, exactly as in the accepted comparison. This gives 2 mm combined padding for self-pairs and 1 mm against the world.
4. Use cuRobo's native sphere self-collision and sphere/world checks. The world consists of the 33 exported lab boxes. Retain the existing self-pair exclusions and fixed-base/world handling from the comparison. These are inspectable in `robot_open_spheres.yml` and `config.json`.
5. Accept a target if at least one converged candidate passes both sphere checks. Save one passing joint configuration and the candidate counts. Collision checking filters the IK results; it does not guide the IK optimization in this run.
6. Independently recompute the achieved hand pose from the exported Isaac articulation transforms for every saved accepted solution.

| Classification | Sampled poses |
| --- | ---: |
| Sphere-clear IK found | {totals['4']:,} |
| Converged IK found; all returned candidates sphere-blocked | {totals['3']:,} |
| No converged IK found | {totals['2']:,} |

All {H*O:,} height/orientation slices passed aggregate shape, count, joint-limit,
finite-value, model-hash, and pose-error checks. The independent transform check allows 2 micrometres of numerical difference around the native 1 mm IK tolerance. Maximum independently computed position
error: **{max_p*1000:.6f} mm**; orientation error: **{max_r:.8f} rad**.
The longest of four worker runs took **{summary['longest_worker_elapsed_s']/60:.2f} minutes**,
excluding the small pilot and aggregation. Physical GPUs: 3, 4, 5, and 7.

## Interpretation

The heatmap is the fraction of the selected orientation samples with a passing solution.
White means no passing solution was found at that sampled position for the selected orientation set.
These orientations are not uniformly weighted over all possible rotations, so the fraction is
not a random-orientation success probability. A negative result is limited by this IK search;
it is not a proof that every joint solution is impossible.

This is an endpoint geometry map on a finite grid. It contains no path-from-home or approach-path
validation and makes no dynamics or controller claim. It uses the user-accepted sphere approximation:
on the saved 2,048-configuration comparison, 157 hull-clear poses were rejected by spheres and one
hull collision was missed (the 4.96 mm finger/upper-arm overlap shown in Isaac Lab).
That known mismatch remains part of this map's model. See the
[comparison](../20_dense_reachability/raw_fit_comparison.md) and
[native Isaac Lab capture](../20_dense_reachability/contact_5mm/isaaclab_reverse.png).

The partial hull-classified map in `20_dense_reachability/endpoint_slices` was stopped and is
superseded by this completed sphere-classified map. Its outputs are preserved separately.

## Height summary

Each height has {Y*X:,} XYZ grid positions and {Y*X*O:,} requested poses.

| Height above table (cm) | Positions with any passing orientation | Positions with a passing downward orientation | Passing poses |
| ---: | ---: | ---: | ---: |
{height_rows}

## Files and reproduction

- `config.json`: exact axes, orientations, transforms, settings, and SHA-256 input hashes.
- `robot_open_spheres.yml`: accepted raw-fit sphere model, unchanged from the compared candidate; the query adds the documented 1 mm radius padding.
- `atlas.npz`: full status array `[height, orientation, y, x]`, coverage counts `[tilt_group, height, y, x]`, and one representative joint solution/orientation per XYZ position.
- `slices/hHH_oOOO.npz`: chosen joint configuration for every accepted pose in that orientation/height slice, candidate counts, and independent pose errors.
- `summary.json`, `software_versions.json`: totals, validation, runtime, and pinned software.
- `reachability_height_slices.png` and `.pdf`: six directly comparable height slices.

Run with `/data/kanth042/envs/curobo-umi-audit/bin/python`.
The input preparation script is `{scripts/'run_sphere_atlas.py'}` with `--prepare`.
Run that script with `--worker 0 --workers 4` through `--worker 3 --workers 4`, assigning
`CUDA_VISIBLE_DEVICES=3`, `4`, `5`, and `7` respectively and `OMP_NUM_THREADS=1`.
Completed slices are resumed after validating the input hashes. To recompute from scratch,
use a fresh output directory or move the completed `slices` directory before launching.
Each worker starts with random seed 1709 plus its worker index; restarting a worker resumes
saved slices but resets its generator sequence for remaining solves.

After every worker finishes, run `{scripts/'summarize_sphere_atlas.py'}`.
The nearest-grid lookup below prints the sampled coordinates and one saved passing joint solution:

```bash
/data/kanth042/envs/curobo-umi-audit/bin/python {scripts/'query_sphere_atlas.py'} --x 0 --y 0 --height 0.30 --tilt 0
```

Add `--azimuth` and `--roll` to select a particular sampled orientation. Omitting orientation
filters searches all 504 samples at the nearest XYZ point. Queries outside the sampled volume
are rejected rather than extrapolated. This lookup only reads the map; it does not command a robot.
'''
(out/'README.md').write_text(report)
print(json.dumps({k:v for k,v in summary.items() if k not in ['heights','modeling_notes']},indent=2))
print('VISUAL',visual,'bytes',len(html.encode()))
