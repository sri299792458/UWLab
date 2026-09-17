"""Look up the nearest sampled XYZ point and optional orientation subset."""
import os
import argparse
import json
from pathlib import Path
import numpy as np

root=Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/umi_reset_from_defaults_20260911') + '/49_clearance_and_placement/atlas')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--x',type=float,required=True,help='World X in metres')
parser.add_argument('--y',type=float,required=True,help='World Y in metres')
parser.add_argument('--height',type=float,required=True,help='Height above tabletop in metres')
parser.add_argument('--tilt',type=int,choices=range(0,181,30))
parser.add_argument('--azimuth',type=int)
parser.add_argument('--roll',type=int)
parser.add_argument('--atlas-dir',type=Path,default=root,help='Current clearance atlas by default')
args=parser.parse_args()
root=args.atlas_dir
cfg=json.loads((root/'config.json').read_text())
for value,key in [(args.x,'x_world_m'),(args.y,'y_world_m'),(args.height,'height_above_table_m')]:
    if not cfg[key][0]-1e-9<=value<=cfg[key][-1]+1e-9:
        parser.error(f'Requested {key}={value} outside sampled range [{cfg[key][0]}, {cfg[key][-1]}]')
ix=int(np.argmin(np.abs(np.asarray(cfg['x_world_m'])-args.x)))
iy=int(np.argmin(np.abs(np.asarray(cfg['y_world_m'])-args.y)))
ih=int(np.argmin(np.abs(np.asarray(cfg['height_above_table_m'])-args.height)))
options=[o for o in cfg['orientations'] if all(value is None or o[key]==value for key,value in
    [('tilt_deg',args.tilt),('azimuth_deg',args.azimuth),('roll_deg',args.roll)])]
if not options:parser.error('That orientation is not in the sampled orientation grid')
with np.load(root/'atlas.npz') as data:
    selected=data['status'][ih,[o['index'] for o in options],iy,ix]
good=np.flatnonzero(selected==4)
result=dict(requested=dict(world_x_m=args.x,world_y_m=args.y,height_above_table_m=args.height),
    sampled=dict(world_x_m=cfg['x_world_m'][ix],world_y_m=cfg['y_world_m'][iy],height_above_table_m=cfg['height_above_table_m'][ih]),
    sampled_orientation_count=len(options),sphere_clear_orientation_count=len(good),
    status='sphere_clear_IK_found' if len(good) else 'no_solution_found_in_sampled_candidates')
if len(good):
    orientation=options[int(good[0])]
    path=root/'slices'/f'h{ih:02d}_o{orientation["index"]:03d}.npz'
    if path.exists():
        with np.load(path) as data:
            result.update(orientation=orientation,joint_names=cfg['joint_names'],
                joint_positions_rad=data['chosen_q'][iy,ix].tolist(),open_hand_joint_values=cfg['open_hand_joint_values'],
                position_error_m=float(data['position_error_m'][iy,ix]),rotation_error_rad=float(data['rotation_error_rad'][iy,ix]),
                source_slice=str(path))
    else:
        result.update(orientation=orientation, joint_solution_available=False,
            note='Core package includes map coverage. Install the full package for saved joint solutions, or solve IK with the included cuRobo model.')
print(json.dumps(result,indent=2))
