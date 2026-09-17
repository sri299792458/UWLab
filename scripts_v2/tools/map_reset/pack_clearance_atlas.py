"""Validate completed map slices and pack their exact joint seeds.

With --wait, process each height as its workers finish. Publish the runtime
lookup only when every slice and its pose-error checks have passed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/umi_reset_from_defaults_20260911') + '/49_clearance_and_placement')
SOURCE, OUT = ROOT / 'atlas', ROOT / 'lookup'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((SOURCE / 'config.json').read_text())
    config_sha = sha256(SOURCE / 'config.json')
    for name, expected in cfg['input_hashes'].items():
        assert sha256(Path(name)) == expected, name
    nh, no, ny, nx = cfg['shape']
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = OUT / 'atlas_joint_seeds.npy.tmp'
    seeds = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                     shape=(nh, ny, nx, no, 6))
    status = np.zeros((nh, no, ny, nx), dtype=np.uint8)
    counts = np.zeros((8, nh, ny, nx), dtype=np.uint16)
    best_q = np.full((nh, ny, nx, 6), np.nan, dtype=np.float32)
    best_o = np.full((nh, ny, nx), -1, dtype=np.int16)
    totals = {str(k): 0 for k in (2, 3, 4)}
    limits = np.asarray(cfg['joint_limits_rad'])
    max_p = max_r = 0.
    began = time.time()
    for h in range(nh):
        paths = [SOURCE / 'slices' / f'h{h:02d}_o{o:03d}' for o in range(no)]
        while not all(p.with_suffix('.json').exists() and p.with_suffix('.npz').exists() for p in paths):
            if not args.wait:
                raise RuntimeError(f'Height {h} is incomplete')
            if time.time() - began > 7200:
                raise TimeoutError('Map packing waited over two hours for slices')
            time.sleep(10)
        for o, path in enumerate(paths):
            meta = json.loads(path.with_suffix('.json').read_text())
            assert meta['config_sha256'] == config_sha
            with np.load(path.with_suffix('.npz')) as data:
                s, q = data['status'], data['chosen_q']
                assert s.shape == (ny, nx) and np.isin(s, [2, 3, 4]).all()
                assert q.shape == (ny, nx, 6)
                good = s == 4
                assert np.isfinite(q[good]).all() and np.isnan(q[~good]).all()
                assert (data['clear_solution_count'][good] > 0).all()
                assert (data['clear_solution_count'][~good] == 0).all()
                if good.any():
                    assert (q[good] >= limits[0] - 2e-6).all() and (q[good] <= limits[1] + 2e-6).all()
                    pe, re = data['position_error_m'][good], data['rotation_error_rad'][good]
                    assert np.isfinite(pe).all() and np.isfinite(re).all()
                    max_p, max_r = max(max_p, float(pe.max())), max(max_r, float(re.max()))
                    assert max_p <= cfg['ik_position_tolerance_m'] + 2e-6
                    assert max_r <= cfg['ik_rotation_tolerance_rad'] + 2e-6
                for code in totals:
                    count = int((s == int(code)).sum())
                    assert count == meta['status_counts'][code]
                    totals[code] += count
                status[h, o] = s
                seeds[h, :, :, o] = q
                counts[0, h] += good
                counts[1 + cfg['orientations'][o]['tilt_deg'] // 30, h] += good
                first = good & (best_o[h] < 0)
                best_q[h, first], best_o[h, first] = q[first], o
        seeds.flush()
        progress = dict(heights_packed=h + 1, total_heights=nh, status_counts=totals,
                        elapsed_s=time.time() - began, complete=False)
        save_json(OUT / 'packing_progress.json', progress)
        print(json.dumps(progress), flush=True)
    assert sum(totals.values()) == cfg['total_targets']
    assert np.array_equal(counts[0], counts[1:].sum(axis=0))
    occupied = counts[0] > 0
    assert occupied.any()
    nearest = distance_transform_edt(~occupied, return_distances=False, return_indices=True)
    nearest_flat = np.ravel_multi_index(nearest, occupied.shape).astype(np.int64)
    assert occupied.ravel()[nearest_flat].all()
    np.save(OUT / 'nearest_occupied_cell.npy', nearest_flat)
    np.savez_compressed(SOURCE / 'atlas.npz', status=status, coverage_counts=counts,
        best_q=best_q, best_orientation_index=best_o, x_world_m=cfg['x_world_m'],
        y_world_m=cfg['y_world_m'], height_above_table_m=cfg['height_above_table_m'])
    del seeds
    temporary.replace(OUT / 'atlas_joint_seeds.npy')
    manifest = dict(source=str(SOURCE), source_config_sha256=config_sha,
        shape=[nh, ny, nx, no, 6], clear_seeds=totals['4'], occupied_xyz=int(occupied.sum()),
        joint_seeds_sha256=sha256(OUT / 'atlas_joint_seeds.npy'),
        nearest_occupied_sha256=sha256(OUT / 'nearest_occupied_cell.npy'),
        use='Nearest occupied XYZ cell and closest available orientation for IK seeds. '
            'Cube/free-hand positions use nearest-cell occupancy across any orientation, without spatial fallback. '
            'Sampled cube and hand angles are preserved; cube-center coverage is an approximate workspace screen. '
            'Final simulated robot clearance is checked separately.')
    save_json(OUT / 'lookup_manifest.json', manifest)
    summary = dict(complete=True, shape=cfg['shape'], total_targets=cfg['total_targets'],
        status_counts=totals, positions_total=nh*ny*nx, positions_with_any_solution=int(occupied.sum()),
        max_position_error_m=max_p, max_rotation_error_rad=max_r, config_sha256=config_sha,
        modeling_notes=cfg['modeling_notes'], heights=[dict(height_above_table_m=z,
        positions_with_any_solution=int(occupied[h].sum()), sphere_clear_poses=int(counts[0,h].sum()))
        for h,z in enumerate(cfg['height_above_table_m'])])
    save_json(SOURCE / 'summary.json', summary)
    progress['complete'] = True
    save_json(OUT / 'packing_progress.json', progress)
    print('COMPLETE ' + json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
