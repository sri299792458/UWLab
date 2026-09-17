"""Download, verify and relocate the Thunder lab data. No robot connections."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root, manifest):
    for index, (name, entry) in enumerate(manifest['files'].items()):
        path = root / name
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError(f'Bundle file mismatch: {name}')
        if index % 10000 == 0:
            print(f'Verified {index}/{len(manifest["files"])} files', flush=True)


def relocate(root, manifest):
    mappings = {
        '/data/kanth042/datasets/umi_reset_from_defaults_20260911/15_table_reachability/curobo': str(root / 'sources/curobo'),
        '/data/kanth042/datasets/umi_reset_from_defaults_20260911': str(root / 'data'),
        '/data/kanth042/converted_assets': str(root / 'assets'),
        '/home/kanth042/.cache/uwlab/assets': str(root / 'cache_assets'),
        '/data/kanth042/repos/UWLab-reset-from-defaults': str(REPO),
        '/data/kanth042/repos/UWLab-thunder-sim2real': str(REPO),
        '/data/kanth042/repos/IsaacLab': str(root / 'sources/IsaacLab'),
    }
    changed = {}
    # Geometry arrays, USD assets and saved tensors retain their original bytes.
    # Original configuration/report bytes remain available for provenance checks.
    for name, entry in manifest['files'].items():
        path = root / name
        if name.startswith('sources/') or path.suffix not in {'.json', '.yaml', '.yml', '.urdf', '.md'}:
            continue
        old = path.read_text()
        new = old
        for source, target in mappings.items():
            new = new.replace(source, target)
        if new != old:
            backup = root / '.original-configs' / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            path.write_text(new)
            changed[name] = dict(original_sha256=entry['sha256'], installed_sha256=sha(path))
    (root / 'relocation.json').write_text(json.dumps(dict(mappings=mappings, changed=changed,
        provenance='Embedded historical hashes still describe the original inputs; original bytes are in .original-configs.'), indent=2)+'\n')
    values = dict(UWLAB_DATA_ROOT=root / 'data', UWLAB_ASSET_ROOT=root / 'assets',
                  UWLAB_CACHE_ASSETS_ROOT=root / 'cache_assets', ISAACLAB_PATH=root / 'sources/IsaacLab',
                  CUROBO_SOURCE=root / 'sources/curobo')
    text = '# Source this after activating the chosen Python environment.\n'
    text += '\n'.join('export '+key+'='+shlex.quote(str(value)) for key, value in values.items())+'\n'
    text += 'export PYTHONPATH='+shlex.quote(':'.join(str(REPO / 'source' / n) for n in
              ['uwlab', 'uwlab_assets', 'uwlab_tasks', 'uwlab_rl']))+'${PYTHONPATH:+:$PYTHONPATH}\n'
    (root / 'activate.sh').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True, help='New directory; refuses to overwrite')
    parser.add_argument('--cache', type=Path, default=Path.home()/'.cache/uwlab/thunder-lab-20260917-v1')
    parser.add_argument('--local-parts', type=Path, help='Use existing downloaded parts without network')
    parser.add_argument('--profile', choices=['core', 'full'], default='core',
                        help='Core lab/model/coverage, or complete map lookup and reset datasets')
    parser.add_argument('--manifest', type=Path, help='Override the selected release manifest')
    args = parser.parse_args()
    manifest_path = args.manifest or HERE / ('release_manifest.json' if args.profile == 'core' else 'release_manifest.full.json')
    release = json.loads(manifest_path.read_text())
    if release.get('published') is False and not args.local_parts:
        raise ValueError('The full archive is prepared but not published. Use the default core profile; full data requires a separate transfer.')
    dest = args.destination.expanduser().resolve()
    if dest.exists():
        raise ValueError(f'Refusing to overwrite {dest}; choose a new directory')
    cache = (args.local_parts or args.cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for entry in release['parts']:
        path = cache / entry['name']
        if not path.exists():
            if args.local_parts:
                raise FileNotFoundError(path)
            url = f'https://github.com/{release["repository"]}/releases/download/{release["tag"]}/{entry["name"]}'
            print(f'Downloading {entry["name"]} ({entry["bytes"]:,} bytes)', flush=True)
            tmp = path.with_suffix(path.suffix+'.partial')
            with urllib.request.urlopen(url) as source, tmp.open('wb') as target:
                shutil.copyfileobj(source, target, 8 << 20)
            tmp.replace(path)
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError(f'Corrupt download: {path}; remove it and retry')
        parts.append(path)
    archive = cache / 'verified-thunder-lab.tar.gz'
    with archive.open('wb') as stream:
        for part in parts:
            with part.open('rb') as source:
                shutil.copyfileobj(source, stream, 8 << 20)
    if sha(archive) != release['archive_sha256']:
        raise ValueError('Concatenated archive checksum mismatch')
    staging = dest.with_name(dest.name+'.extracting')
    staging.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, 'r:gz') as stream:
        stream.extractall(staging, filter='data')
    root = staging / 'thunder-lab'
    manifest_path = root/'bundle_manifest.json'
    if sha(manifest_path) != release['manifest_sha256']:
        raise ValueError('Internal manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_text())
    verify(root, manifest)
    root.rename(dest)
    staging.rmdir()
    relocate(dest, manifest)
    archive.unlink()
    print(f'Installed and verified {len(manifest["files"]):,} files at {dest}', flush=True)
    print('Next: source '+shlex.quote(str(dest/'activate.sh')))


if __name__ == '__main__':
    main()
