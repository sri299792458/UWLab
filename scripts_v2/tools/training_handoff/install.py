"""Install the complete, verified training data and portable stock-hand layer."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="Download cache; defaults beside the destination, not in home")
    parser.add_argument("--local-parts", type=Path)
    args = parser.parse_args()
    root = args.destination.expanduser().resolve()
    if not args.cache and not args.local_parts:
        release = json.loads((HERE / "release_manifest.json").read_text())
        args.cache = root.parent / ".uwlab-downloads" / release["tag"]
    command = [sys.executable, str(HERE.parent / "thunder_sim2real/portable/install_bundle.py"),
               "--profile", "full", "--manifest", str(HERE / "release_manifest.json"),
               "--destination", str(root)]
    for key in ("cache", "local_parts"):
        if getattr(args, key):
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    manifest = json.loads((HERE / "stock_asset_manifest.json").read_text())
    for name, entry in manifest["files"].items():
        path = HERE / "stock_hand_asset" / name
        assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], path
    subprocess.run(command, check=True)
    assert sha(root / manifest["base_usd_relative"]) == manifest["base_usd_sha256"]
    stock = root / manifest["installed_directory"]
    shutil.copytree(HERE / "stock_hand_asset", stock)
    record = {
        "status": "PASS", "release": json.loads((HERE / "release_manifest.json").read_text()),
        "stock_asset_manifest": manifest, "repository": str(REPO),
        "scope": "All downloaded file hashes verified; stock layer sublayer path is relative. Simulator validation is separate.",
    }
    (root / "training_installation.json").write_text(json.dumps(record, indent=2) + "\n")
    with (root / "activate.sh").open("a") as stream:
        stream.write("export UWLAB_TRAINING_BUNDLE=" + shlex.quote(str(root)) + "\n")
        stream.write("export UWLAB_STOCK_HAND_USD=" + shlex.quote(str(stock / "umi_geometry_stock_hand_dynamics.usda")) + "\n")
    print("Training inputs ready: " + str(root), flush=True)


if __name__ == "__main__":
    main()
