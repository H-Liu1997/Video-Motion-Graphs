import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def verify(path, asset):
    if not path.is_file() or path.stat().st_size != asset["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == asset["sha256"]


def main():
    parser = argparse.ArgumentParser(description="Download and verify VMG models and raw examples.")
    parser.add_argument("--group", choices=["all", "base", "preprocessing", "checkpoint", "examples"], default="all")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assets = json.loads((root / "asset_manifest.json").read_text())["assets"]
    missing = []
    for asset in assets:
        if args.group != "all" and asset["group"] != args.group:
            continue
        path = root / asset["path"]
        if verify(path, asset):
            print(f"Verified {asset['path']}", flush=True)
            continue
        if args.verify_only:
            missing.append(asset["path"])
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        print(f"Downloading {asset['path']} ({asset['bytes']:,} bytes)", flush=True)
        with urllib.request.urlopen(asset["url"], timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        if not verify(temporary, asset):
            raise RuntimeError(f"Checksum mismatch: {asset['path']}")
        temporary.replace(path)
    if missing:
        raise SystemExit("Missing or invalid assets: " + ", ".join(missing))


if __name__ == "__main__":
    main()
