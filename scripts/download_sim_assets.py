#!/usr/bin/env python3
"""Download and verify the exact binary assets used by the Franka simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use a previously downloaded archive')
    parser.add_argument('--destination', type=Path, default=ROOT)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing asset files')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/simulation_assets.json').read_text())
    archive = args.archive
    if archive is None:
        from modelscope.hub.file_download import dataset_file_download
        archive = Path(dataset_file_download(
            manifest['repo_id'], manifest['path_in_repo'],
            revision=manifest['revision'], local_dir=str(ROOT / '.cache/modelscope'),
        ))
    digest = hashlib.sha256()
    with archive.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != manifest['sha256']:
        raise RuntimeError('Simulation asset SHA256 mismatch; refusing to extract.')
    destination = args.destination.resolve()
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not (member.isdir() or member.isfile()):
                raise RuntimeError(f'Unsafe archive member: {member.name}')
            if not args.verify_only and member.isfile() and target.exists() and not args.overwrite:
                raise FileExistsError(f'{target} exists; use --overwrite only to replace your local assets.')
        if not args.verify_only:
            destination.mkdir(parents=True, exist_ok=True)
            bundle.extractall(destination, members=members)
    print(f'Verified {len(members)} archive entries; SHA256={digest.hexdigest()}')
    if not args.verify_only:
        print(f'Installed simulation assets under {destination}')


if __name__ == '__main__':
    main()
