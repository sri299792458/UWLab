"""Summarize paired gain trials without discarding failed poses."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--phase',default='screen');args=parser.parse_args()
ROOT=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911/22_explicit_gain_sweep')
plan=json.loads((ROOT/'plan.json').read_text())
summaries=[json.loads(p.read_text()) for p in sorted((ROOT/args.phase).glob('worker_*/summary.json'))]
assert summaries,'No completed workers'
rows=[]
for summary in summaries:
    assert summary['physics_hz']==120 and summary['decimation']==12 and summary['action_hz']==10
    assert summary['controller_formula_max_error_nm']<1e-5
    for row in summary['results']:
        ratios=[]
        for name,limit in plan['ranking_limits'].items():
            values=np.array([np.inf if x is None else x for x in row['per_pose'][name]])
            ratios.append(values/limit)
        worst=np.max(ratios,axis=0)
        row['normalized_worst_metric_p90']=float(np.quantile(worst,.9)) if np.isfinite(worst).all() else None
        rows.append(row)
assert len({r['id'] for r in rows})==len(rows)
rows.sort(key=lambda r:(-r['passed'],r['count']-r['finite'],r['normalized_worst_metric_p90'] if r['normalized_worst_metric_p90'] is not None else float('inf')))
report=dict(phase=args.phase,gain_settings=len(rows),physics_hz=120,decimation=12,action_hz=10,
    poses_per_setting=summaries[0]['pose_count'],trial_count=sum(r['count'] for r in rows),
    ranking='Passing pose count first, then finite count, then 90th percentile across poses of the worst error/speed metric divided by its stated screening limit.',
    ranking_limits=plan['ranking_limits'],results=rows)
(ROOT/f'{args.phase}_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
fields=['id','label','count','passed','finite','abnormal','normalized_worst_metric_p90']
if 'grasp_passed' in rows[0]:fields+=['grasp_passed','controller_and_grasp_passed']
fields+=sorted(k for k in rows[0] if k.endswith('_median') or k.endswith('_max'))
with (ROOT/f'{args.phase}_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    w.writerows({k:r.get(k) for k in fields} for r in rows)
print('COMPLETED',len(rows),'settings;',sum(r['count'] for r in rows),'trials')
for r in rows[:12]:
    print(json.dumps({k:r[k] for k in ['id','label','passed','count','motion_position_rms_m_median','motion_rotation_rms_rad_median','final_wrist_speed_rms_rad_s_median','normalized_worst_metric_p90']}))
