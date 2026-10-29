"""Update exact shared files from a checked-out EDOS revision."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent


def _normalised_digest(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    resolved_commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if resolved_commit != args.commit:
        raise SystemExit(f"EDOS checkout is {resolved_commit}, expected {args.commit}")

    manifest_path = HERE / "alignment.yml"
    lock_path = HERE / "edos.lock"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    product_root = (REPOSITORY / manifest["product_root"]).resolve()
    hashes = {}
    for relative_path in manifest["shared_exact"]:
        source_path = source / relative_path
        target_path = product_root / relative_path
        if not source_path.is_file():
            raise SystemExit(f"EDOS shared file is missing: {relative_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        hashes[relative_path] = _normalised_digest(source_path)

    manifest["expected_sha256"] = hashes
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["commit"] = args.commit
    lock["aligned_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {len(hashes)} shared files from EDOS {args.commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
