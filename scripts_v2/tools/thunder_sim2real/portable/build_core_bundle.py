"""Maintainer: derive the compact core archive from a completed full staging tree."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-output', required=True, type=Path)
    args = p.parse_args()
    out = args.full_output.resolve()
    source = out/'thunder-lab'
    stage = out/'core'/'thunder-lab'
    stage.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source/'bundle_manifest.json').read_text())
    excluded = ('data/49_clearance_and_placement/atlas/slices/',
                'data/49_clearance_and_placement/lookup/', 'data/49_clearance_and_placement/OmniReset/')
    manifest['files'] = {k:v for k,v in manifest['files'].items() if not k.startswith(excluded)}
    for name in manifest['files']:
        target = stage/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
    manifest.update(profile='core', total_bytes=sum(v['bytes'] for v in manifest['files'].values()),
        scope='Robot/lab physics assets, collision meshes and models, atlas coverage, calibration, software sources and selected-motion evidence. Full joint lookup/slices and reset tensors are optional in the full profile.')
    (stage/'bundle_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive = out/'thunder-lab-core.tar.gz.part000'
    with archive.open('wb') as stream:
        subprocess.run(['tar','--use-compress-program=pigz -1 -p 4','-cf','-','-C',str(stage.parent),'thunder-lab'], stdout=stream, check=True)
    release = dict(schema=1, tag=manifest['bundle'], repository='sri299792458/UWLab', profile='core',
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        manifest_sha256=sha(stage/'bundle_manifest.json'), extracted_bytes=manifest['total_bytes'],
        parts=[dict(name=archive.name, bytes=archive.stat().st_size, sha256=sha(archive))])
    (out/'release_manifest.core.json').write_text(json.dumps(release, indent=2)+'\n')
    print(json.dumps(release, indent=2))


if __name__ == '__main__':
    main()
