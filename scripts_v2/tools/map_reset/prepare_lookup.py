"""Pack the completed atlas into an exact, dense joint-seed lookup."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911')
SOURCE = ROOT / '21_sphere_reachability'
OUT = ROOT / '26_map_reset_regeneration'


def main():
    cfg = json.loads((SOURCE / 'config.json').read_text())
    nh, no, ny, nx = cfg['shape']
    path = OUT / 'atlas_joint_seeds.npy'
    temporary = path.with_suffix('.npy.tmp')
    q = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                 shape=(nh, ny, nx, no, 6))
    occupied = np.zeros((nh, ny, nx), dtype=bool)
    count = 0
    for h in range(nh):
        for o in range(no):
            with np.load(SOURCE / 'slices' / f'h{h:02d}_o{o:03d}.npz') as d:
                good = d['status'] == 4
                assert np.array_equal(good, np.isfinite(d['chosen_q']).all(axis=-1))
                q[h, :, :, o] = d['chosen_q']
                occupied[h] |= good
                count += int(good.sum())
        q.flush()
        print(f'Packed height {h+1}/{nh}, clear seeds {count}', flush=True)
    del q
    temporary.replace(path)
    nearest = distance_transform_edt(~occupied, return_distances=False, return_indices=True)
    nearest_flat = np.ravel_multi_index(nearest, occupied.shape).astype(np.int64)
    np.save(OUT / 'nearest_occupied_cell.npy', nearest_flat)
    result = dict(source=str(SOURCE), source_config_sha256=hashlib.sha256(
        (SOURCE / 'config.json').read_bytes()).hexdigest(), shape=[nh, ny, nx, no, 6],
        clear_seeds=count, occupied_xyz=int(occupied.sum()),
        joint_seeds_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        use='Nearest occupied XYZ cell, then closest available orientation. Seed only; exact pose IK and final collision checks still required.')
    (OUT / 'lookup_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
