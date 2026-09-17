"""Build the final comparison only from corrected, independently checked runs."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911/22_explicit_gain_sweep')
SCRIPT = Path(__file__).parent


def read(name):
    return json.loads((ROOT / name).read_text())


screen = read('corrected_screen_summary.json')
holdout = read('corrected_holdout_summary.json')
grasp = read('grasp_summary.json')
collisions = {r['gain_id']: r for r in read('corrected_holdout_sphere_collision_audit.json')['results']}
validation = read('initial_pose_validation.json')
assert {'corrected_screen', 'corrected_holdout', 'grasp'} <= {v['phase'] for v in validation['results']}
assert screen['gain_settings'] == 296 and screen['poses_per_setting'] == 16
assert holdout['gain_settings'] == grasp['gain_settings'] == 5
assert holdout['poses_per_setting'] == 80 and grasp['poses_per_setting'] == 72
srows, hrows, grows = [{r['id']: r for r in data['results']} for data in (screen, holdout, grasp)]
limits = holdout['ranking_limits']


def controller_pass(r):
    good = np.array(r['per_pose']['finite'], bool) & ~np.array(r['per_pose']['abnormal'], bool)
    for key, limit in limits.items():
        good &= np.array(r['per_pose'][key]) <= limit
    assert int(good.sum()) == r['passed']
    return good


rows = []
for gain_id in [0, 292, 276, 289, 288]:
    s, h, g, c = srows[gain_id], hrows[gain_id], grows[gain_id], collisions[gain_id]
    clear = np.ones(80, bool)
    clear[c['blocked_pose_indices']] = False
    hg, gg = controller_pass(h), controller_pass(g)
    m = g['per_pose']
    retained = (np.array(m['finite'], bool) & (np.array(m['cube_relative_error_max_m']) < .015)
        & (np.array(m['cube_rotation_error_max_rad']) < math.radians(5))
        & (np.array(m['cube_clearance_min_m']) > .02)
        & (np.array(m['bilateral_contact_fraction']) >= .8))
    assert retained.sum() == g['grasp_passed']
    assert (gg & retained).sum() == g['controller_and_grasp_passed']
    mass_results = {str(mass): dict(count=24, controller=int(gg[i::3].sum()),
        grasp=int(retained[i::3].sum()), combined=int((gg & retained)[i::3].sum()))
        for i, mass in enumerate([.02, .1, .2])}
    rows.append(dict(id=gain_id, kp=h['kp'], kd=h['kd'],
        screen_passed=s['passed'], holdout_passed=h['passed'],
        holdout_passed_and_sphere_clear=int((hg & clear).sum()), holdout_sphere_flagged=int((~clear).sum()),
        holdout_position_rms_mm=float(np.median(h['per_pose']['motion_position_rms_m'])*1000),
        holdout_rotation_rms_deg=float(np.degrees(np.median(h['per_pose']['motion_rotation_rms_rad']))),
        holdout_final_wrist_speed_rms_rad_s=float(np.median(h['per_pose']['final_wrist_speed_rms_rad_s'])),
        grasp_retained=g['grasp_passed'], grasp_controller_passed=g['passed'],
        grasp_combined_passed=g['controller_and_grasp_passed'], grasp_by_mass=mass_results,
        grasp_position_drift_max_mm=max(m['cube_relative_error_max_m'])*1000,
        grasp_rotation_drift_max_deg=float(np.degrees(max(m['cube_rotation_error_max_rad']))),
        grasp_minimum_both_finger_contact_fraction=min(m['bilateral_contact_fraction']),
        holdout_failed_pose_indices=np.flatnonzero(~hg).tolist(),
        grasp_failed_controller_case_indices=np.flatnonzero(~gg).tolist(),
        holdout_failed_counts_by_criterion={key:int((np.array(h['per_pose'][key])>limit).sum()) for key,limit in limits.items()},
        grasp_failed_counts_by_criterion={key:int((np.array(g['per_pose'][key])>limit).sum()) for key,limit in limits.items()}))

report = dict(status='Gain comparison complete; no setting passes all tested cases; training configuration not changed.',
    physics_hz=120, decimation=12, action_hz=10,
    recommended_candidate_for_further_validation=292,
    candidate_basis='Highest controller pass count on the reserved open-hand poses, with contact results reported separately; no claim of universal validity.',
    controller_limits=limits, results=rows)
(ROOT/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')

plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
    'axes.labelcolor':'#253347', 'text.color':'#253347', 'axes.edgecolor':'#b7c0cd',
    'figure.facecolor':'white', 'axes.facecolor':'white'})
fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.4))
names = ['Default / UMI', 'A · 292', 'B · 276', 'C · 289', 'D · 288']
x = np.arange(5)
passing = np.array([r['holdout_passed'] for r in rows])
clear = np.array([r['holdout_passed_and_sphere_clear'] for r in rows])
axes[0].bar(x, clear, color='#278568', label='Tracking pass + sphere clear')
axes[0].bar(x, passing-clear, bottom=clear, color='#a7c5e3', label='Tracking pass; sphere flagged')
for i, n in enumerate(passing):
    axes[0].text(i,n+1,str(n),ha='center',fontsize=10)
axes[0].set(ylim=(0,85), title='Reserved open-hand poses', ylabel='Cases passing / 80', xticks=x, xticklabels=names)
axes[0].legend(frameon=False,fontsize=8,loc='upper left')

axes[1].bar(x, [r['grasp_retained'] for r in rows], color='#d5deea', label='Grasp retained')
axes[1].bar(x, [r['grasp_combined_passed'] for r in rows], color='#377cae', label='Grasp retained + tracking pass')
for i,r in enumerate(rows):
    axes[1].text(i,r['grasp_combined_passed']+.8,str(r['grasp_combined_passed']),ha='center',fontsize=10)
axes[1].set(ylim=(0,85),title='24 grasps × 3 cube masses',ylabel='Cases passing / 72',xticks=x,xticklabels=names)
axes[1].legend(frameon=False,fontsize=8,loc='upper left')

for gain_id,color,label in [(0,'#a44b50','Default / UMI'),(292,'#278568','A · 292'),(276,'#377cae','B · 276')]:
    values=np.sort(hrows[gain_id]['per_pose']['final_wrist_speed_rms_rad_s'])
    axes[2].step(values,np.arange(1,81)/80*100,where='post',color=color,label=label,lw=1.8)
axes[2].axvline(.1,color='#4e5968',ls='--',lw=1,label='0.1 rad/s limit')
axes[2].set(xscale='log',ylim=(0,103),title='Wrist motion during final hold',xlabel='RMS wrist-joint speed norm (rad/s)',ylabel='Reserved poses at or below value (%)')
axes[2].legend(frameon=False,fontsize=8,loc='upper left')
for ax in axes:
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Explicit Cartesian gain comparison · 120 Hz physics · decimation 12',fontsize=15,y=.99)
fig.text(.5,.016,'Computed from corrected Isaac Lab runs. Provisional screening limits; sphere flags use the accepted fixed-open-hand model. No training or hardware validation.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.065,1,.93))
fig.savefig(ROOT/'gain_comparison.png',dpi=180)
fig.savefig(ROOT/'gain_comparison.svg')
plt.close(fig)

lines = [
'# Explicit Cartesian gain sweep at 120 Hz',
'',
'The comparison is complete. Candidate A (gain 292) improves tracking and final wrist settling substantially, but no tested setting passes every case. These results do not establish a validated production gain setting, training readiness, or hardware performance. The active controller and training configuration were not changed.',
'',
'**Benchmark clarification:** every row in this original gain sweep uses the current UMI model with nominal inertias. “Default / UMI” is not the complete original stock model with its fallback properties. The provisional tracking limits below were not derived from stock performance. The subsequently added [default-gain reference comparison](fallback_reference.md) measures the stock reference and a separate UMI knuckle-inertia intervention. Stock settles below the 0.1 rad/s wrist criterion on 80/80 reserved poses versus A’s 77/80, although it also fails the earlier combined strict tracking cutoff. Read that comparison before interpreting these pass counts as readiness.',
'',
'## Corrected results',
'',
'All numbers below exclude the invalid initial runs. A gain is written as `translation Kp / Kd; rotation Kp / Kd`; each value is shared by its three Cartesian axes. The runtime arrays are in `comparison.json`.',
'',
'| Setting | Gains | Screen /16 | Reserved tracking /80 | Tracking + sphere clear /80 | Cube retained /72 | Cube retained + tracking /72 |',
'|---|---|---:|---:|---:|---:|---:|',
]
for label,r in zip(names,rows):
    lines.append(f"| {label} | {r['kp'][0]:g} / {r['kd'][0]:.5g}; {r['kp'][3]:g} / {r['kd'][3]:.5g} | {r['screen_passed']} | {r['holdout_passed']} | {r['holdout_passed_and_sphere_clear']} | {r['grasp_retained']} | {r['grasp_combined_passed']} |")
lines += ['', '| Setting | Reserved motion position RMS, median | Reserved motion rotation RMS, median | Final wrist-speed RMS, median |',
    '|---|---:|---:|---:|']
for label,r in zip(names,rows):
    lines.append(f"| {label} | {r['holdout_position_rms_mm']:.3f} mm | {r['holdout_rotation_rms_deg']:.3f}° | {r['holdout_final_wrist_speed_rms_rad_s']:.6f} rad/s |")
lines += ['', 'Medians in this report use the usual average of the middle two values. Raw worker summaries use PyTorch’s lower-middle median for even sample counts; this small presentation difference does not affect any pass count.',
    '', '## Loaded results by mass', '', '| Setting | 20 g combined /24 | 100 g combined /24 | 200 g combined /24 | Maximum cube position slip | Maximum cube rotation slip |',
    '|---|---:|---:|---:|---:|---:|']
for label,r in zip(names,rows):
    v=r['grasp_by_mass']
    lines.append(f"| {label} | {v['0.02']['combined']} | {v['0.1']['combined']} | {v['0.2']['combined']} | {r['grasp_position_drift_max_mm']:.3f} mm | {r['grasp_rotation_drift_max_deg']:.3f}° |")
lines += [
'', '## What was held fixed and measured', '',
'- The existing `RelCartesianOSCAction.apply_actions` executes `tau = J.T @ (Kp * pose_error - Kd * (J @ qdot))`, then applies the existing 150/28 Nm joint torque limits. The recorded command matched this formula exactly in every completed run. No implicit damping, mass-matrix solve, inertia addition, controller reformulation, or source gain override was introduced.',
'- Physics and feedback remain 120 Hz. Decimation remains 12, so targets change at 10 Hz. Each trial has a 2 s initial hold, a 6 s windowed 0.5 Hz motion (20 mm and 0.1 rad component amplitudes), then a 2 s final hold. Position and orientation targets are referenced to the initial wrist pose in the robot base frame. The diagnostic bypasses policy action clipping to prescribe the same absolute trajectory for every gain.',
'- Arm masses, inertias, zero joint-drive Kp/Kd, armature, joint limits, gravity setting, and solver settings remain inherited nominal values. The gripper retains its existing 17/5 joint-drive gains. Startup/reset randomization is disabled, so authored/default material properties are used. This is a nominal dynamics comparison, not a randomized robustness test.',
'- The 296 settings comprise four references, the initial 108-point grid, and 184 additional unique settings. All 296 run on the same 16 screening poses. Only the leading four plus the original baseline run on the 80 reserved poses. The 96 starting arm poses come from the accepted reachability atlas, spanning seven tilt bins from downward to upward orientations. `plan_expanded.json` and `poses.json` contain the exact grid, seeds, inputs, and poses.',
'- The loaded test uses 24 airborne saved grasp geometries, each paired at 20, 100, and 200 g. Robot and cube are transformed together to the intended mount; all initial velocities are zeroed. Cube inertia scales with mass. The saved generation controller is not replayed, and its dynamics are not considered validation. All gains see the same initial states. These are different arm postures from the open-hand test, so differences between the two phases cannot be attributed solely to payload.',
'- Every physics step records target errors, all six arm joint positions/velocities, commanded torques, and abnormal-speed flags. The loaded phase also records cube slip, cube height above the table, both finger-body contact forces filtered to the cube, driver angle, cube speed, and cube angular slip. Each finger-body sensor aggregates its TPU pad and PETG adapter; it does not isolate the soft pad shape.',
'', '## Provisional pass criteria', '',
'An arm trial must remain finite and have no speed above twice its configured joint limit. During the 6 s motion, position RMS must be at most 5 mm and orientation RMS at most 0.035 rad (2.01°). During the final 1 s, position RMS must be at most 2 mm, orientation RMS at most 0.02 rad (1.15°), and the RMS norm of the three wrist joint speeds at most 0.1 rad/s. RMS means root mean square over the stated interval.',
'',
'A retained grasp must, after the initial 2 s, keep cube position slip below 15 mm and orientation slip below 5° relative to its initial pose in the hand, keep the cube bottom above the table by more than 20 mm, and have force above 0.02 N on both finger bodies for at least 80% of the remaining time. “Combined” requires both retained grasp and all arm criteria. These screening thresholds are explicit diagnostic choices, not proof of policy success or real robot safety.',
'', '## Geometry and execution validation', '',
f"The intended base pose is explicitly written and asserted before each trial. Every saved initial body pose, including the six finger coordinates, is checked against independent forward kinematics from the exported USD articulation tree. Across the final phases the maximum body position discrepancy is {max(x['body_position_max_error_m'] for x in validation['results'])*1e6:.3f} micrometers and the maximum orientation discrepancy is {max(x['body_rotation_max_error_rad'] for x in validation['results']):.3g} rad. Clone origins increase floating-point position error for distant clones. Cube initial transforms and the analytical Jacobian have separate checks in `initial_pose_validation.json` and `jacobian_audit.json`.",
'',
'The motion audit replays all recorded 120 Hz arm configurations through the accepted cuRobo sphere checker with 1 mm added to each sphere and the same base-world omission and self-pair masks as the atlas. It uses fixed-open-hand geometry, tests sampled configurations rather than continuous swept motion, and is not physical mesh collision proof. The starting poses are clear; some prescribed motions leave that clear region. The loaded phase has actual Isaac contact dynamics but has not received an independent full closed-hand/payload motion collision audit.',
'', '## Discarded diagnostic runs and correction', '',
'The first diagnostic disabled all reset events but omitted the normal explicit base-body pose write. The USD asset’s authored base rotation therefore remained in addition to the requested spawn rotation. This misaligned the simulated robot with the intended world used for cube placement and collision checking. It was found when the first cube pilot free-fell with no finger force. Native body-pose inspection identified the mismatch. The diagnostic was corrected to write and verify the root pose; all 296 screen settings, selected reserved-pose tests, and loaded comparisons were rerun. Earlier provisional counts must not be used. `invalidated_runs.json` identifies the retained but excluded output folders. This correction is confined to the diagnostic runner.',
'', '## Interpretation and remaining limits', '',
'A is the strongest candidate by reserved-pose arm pass count. B gives lower median motion position error and led the 16-pose screen; the difference illustrates why the reserved poses matter. Both still fail cases. Retaining a cube can coexist with unacceptable wrist oscillation, which is why retention and tracking are reported separately. Failure indices and per-criterion counts remain in `comparison.json`; no failing poses were removed.',
'',
'For A, the same 12 of the 24 grasp postures fail at all three masses. This points to a posture-dependent limitation rather than evidence that increased payload alone caused the failures; it does not identify the cause. Across its 72 loaded trials, 33 exceed the motion position limit, 24 exceed the final wrist-speed limit, 23 exceed the final position limit, and 18 each exceed the motion/final orientation limits. These failure counts overlap. Maximum final wrist-speed RMS is 1.97 rad/s, so a low median does not imply all trials settle.',
'',
'No policy was trained, no gains were activated in the training configuration, no timing or controller-form change was made, and no hardware performance is inferred. A follow-up should examine the remaining failed motions and their collision/kinematic conditions before deciding whether a fixed 120 Hz gain set is adequate for the task. These reserved-pose results have now been observed; further tuning against them would require a fresh final validation set.',
'', '## Reproduce', '',
f'The scripts are in `{SCRIPT}`. The Isaac interpreter is `/data/kanth042/envs/uwlab-isaac51/bin/python`; add this repository’s `source/uwlab_tasks`, `source/uwlab_assets`, `source/uwlab`, and `source/uwlab_rl` to `PYTHONPATH`, and set `OMP_NUM_THREADS=1`. Native Isaac uses `--device cuda:G` without `CUDA_VISIBLE_DEVICES`. Each worker directory preserves its runner source and runtime inputs.',
'', '```bash',
'python run_gain_sweep.py --phase screen --worker W --workers 4 --plan /data/kanth042/datasets/umi_reset_from_defaults_20260911/22_explicit_gain_sweep/plan_expanded.json --output-name NEW_SCREEN_NAME --headless --device cuda:G',
'python run_gain_sweep.py --phase holdout --gain-ids 292,276,289,288,0 --plan /data/kanth042/datasets/umi_reset_from_defaults_20260911/22_explicit_gain_sweep/plan_expanded.json --output-name NEW_HOLDOUT_NAME --headless --device cuda:G',
'python run_gain_sweep.py --phase grasp --gain-ids 292,276,289,288,0 --plan /data/kanth042/datasets/umi_reset_from_defaults_20260911/22_explicit_gain_sweep/plan_expanded.json --output-name NEW_GRASP_NAME --headless --device cuda:G',
'```', '',
'Use fresh output names: the runner refuses to overwrite completed summaries. `summarize_gains.py --phase NAME` writes ranked JSON and CSV. Run `verify_initial_poses.py --phases NAME ...` with the cuRobo environment’s Python. `audit_trajectory_collisions.py --phase NAME` requires the cuRobo environment and a selected GPU. `report_results.py` reads the finalized corrected phase names and creates this report, `comparison.json`, and the PNG/SVG plot. Runtime snapshots include source hashes, actual masses/inertias, joint settings, initial poses, and gain arrays.',
]
(ROOT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))
