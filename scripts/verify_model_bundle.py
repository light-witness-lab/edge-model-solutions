#!/usr/bin/env python3
"""Verify a checked-in model bundle against its manifest."""
import argparse, hashlib, json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.model_dir / "manifest.json").read_text(encoding="utf-8"))
    total = 0
    for spec in manifest["files"]:
        path = args.model_dir / spec["path"]
        if not path.is_file():
            raise SystemExit(f"missing: {spec['path']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        if size != spec["bytes"] or digest != spec["sha256"]:
            raise SystemExit(f"mismatch: {spec['path']}")
        total += size
        print(f"OK {size:>10} {digest}  {spec['path']}")
    if total != manifest["total_bytes"]:
        raise SystemExit(f"total mismatch: {total} != {manifest['total_bytes']}")
    print(f"verified {len(manifest['files'])} files, {total} bytes")

if __name__ == "__main__":
    main()
