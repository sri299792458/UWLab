"""Maintainer: snapshot the current Thunder inputs without modifying their sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[4]
DATA = Path(os.environ.get('UWLAB_DATA_ROOT', '/data/kanth042/datasets/thunder_mount_corrected_20mm_20260917'))
ASSETS = Path('/data/kanth042/converted_assets')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tag', default='thunder-lab-20260917-mount-v2')
    parser.add_argument('--curobo-source', type=Path,
                        default=Path('/data/kanth042/datasets/umi_reset_from_defaults_20260911/15_table_reachability/curobo'))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stage = out / 'thunder-lab'
    stage.mkdir()
    files = {}

    def add(source, relative):
        source = Path(source)
        if source.is_dir():
            for child in sorted(source.rglob('*')):
                if child.is_file() and '__pycache__' not in child.parts:
                    add(child, Path(relative) / child.relative_to(source))
            return
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[str(relative)] = dict(bytes=target.stat().st_size, sha256=sha(target), source=str(source))

    for name in ['thunder_d405_umi_rigid_asset', 'thunder_d405_umi_gripper_asset',
                 'aprilcube_60mm_rounded', 'lab_vention_asset_v60']:
        for path in sorted((ASSETS / name).rglob('*')):
            if path.is_file() and not any('before' in part for part in path.relative_to(ASSETS / name).parts):
                add(path, Path('assets') / path.relative_to(ASSETS))
    add('/home/kanth042/.cache/uwlab/assets', 'cache_assets')
    add(DATA / '15_table_reachability/model', 'data/15_table_reachability/model')
    add(DATA / '14_lab_setup_audit/lab_geometry_audit.json', 'data/14_lab_setup_audit/lab_geometry_audit.json')
    for path in sorted((DATA / '20_dense_reachability').iterdir()):
        if path.is_file() and path.suffix in {'.json', '.yml', '.md', '.npz'}:
            add(path, Path('data/20_dense_reachability') / path.name)
    current = DATA / '49_clearance_and_placement'
    for name in ['atlas', 'lookup', 'OmniReset']:
        add(current / name, Path('data/49_clearance_and_placement') / name)
    for path in sorted(current.iterdir()):
        if path.is_file() and path.suffix in {'.json', '.md', '.py', '.patch'}:
            add(path, Path('data/49_clearance_and_placement') / path.name)
    add(DATA/'rebuild_inputs.json', 'data/rebuild_inputs.json')
    for name in ['mounting_validation.json', 'world_boxes.json', 'corrected_mount_front.png', 'corrected_mount_other_angle.png']:
        add(DATA/'mounting_validation'/name, Path('data/mounting_validation')/name)
    preview = DATA / '53_thunder_sim2real/motion_preview/lower_full'
    add(preview, 'data/53_thunder_sim2real/motion_preview/lower_full')
    sources = {}
    for name, source in [('IsaacLab', Path('/data/kanth042/repos/IsaacLab')),
                         ('curobo', args.curobo_source)]:
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        archive = out / (name + '.tar')
        subprocess.run(['git', '-C', str(source), 'archive', '--format=tar', '-o', str(archive), 'HEAD'], check=True)
        target = stage / 'sources' / name
        target.mkdir(parents=True)
        with tarfile.open(archive) as stream:
            stream.extractall(target, filter='data')
        archive.unlink()
        patch = subprocess.check_output(['git', '-C', str(source), 'diff', '--binary', 'HEAD'])
        patch_path = stage / 'sources' / (name + '.patch')
        patch_path.write_bytes(patch)
        if patch:
            subprocess.run(['git', 'apply', str(patch_path)], cwd=target, check=True)
        for path in sorted(target.rglob('*')):
            if path.is_file():
                rel = str(path.relative_to(stage))
                files[rel] = dict(bytes=path.stat().st_size, sha256=sha(path), source=f'{name}@{commit}+local-patch')
        files[str(patch_path.relative_to(stage))] = dict(bytes=len(patch), sha256=sha(patch_path), source=str(source))
        sources[name] = dict(commit=commit, patch_sha256=sha(patch_path))
    manifest = dict(schema=1, bundle=args.tag, files=files,
                    source_repositories=sources, total_bytes=sum(v['bytes'] for v in files.values()),
                    scope='Current lab/robot/assets, full clearance atlas+slices+lookup, current reset banks, validation, pinned dependency sources. No robot path or robot execution.',
                    exclusions=['Historical superseded assets and maps', 'Intermediate duplicate reset banks and logs',
                                'Live training checkpoints and credentials', 'Installed proprietary simulator binaries (install from NVIDIA)'])
    (stage / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(staging=str(stage), files=len(files), bytes=manifest['total_bytes']), indent=2), flush=True)
    archive = out / 'thunder-lab.tar.gz'
    with archive.open('wb') as stream:
        subprocess.run(['tar', '--use-compress-program=pigz -1 -p 4', '-cf', '-', '-C', str(out), stage.name], stdout=stream, check=True)
    parts = []
    with archive.open('rb') as source:
        index = 0
        while True:
            block = source.read(8 << 20)
            if not block:
                break
            name = f'thunder-lab.tar.gz.part{index:03d}'
            part = out / name
            size = 0
            with part.open('wb') as stream:
                while block:
                    stream.write(block)
                    size += len(block)
                    if size >= 1536 << 20:
                        break
                    block = source.read(min(8 << 20, (1536 << 20) - size))
            parts.append(dict(name=name, bytes=size, sha256=sha(part)))
            index += 1
    release = dict(schema=1, tag=args.tag, repository='sri299792458/UWLab',
                   archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                   manifest_sha256=sha(stage / 'bundle_manifest.json'), extracted_bytes=manifest['total_bytes'], parts=parts)
    (out / 'release_manifest.json').write_text(json.dumps(release, indent=2)+'\n')
    print(json.dumps(release, indent=2), flush=True)


if __name__ == '__main__':
    main()
