"""Verify the product's exact shared files against its locked EDOS commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent


def _git_bytes(source: Path, commit: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(source), "show", f"{commit}:{relative_path}"],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"Unable to read {relative_path} at {commit}")
    return result.stdout


def _digest(content: bytes) -> str:
    return hashlib.sha256(_normalise(content)).hexdigest()


def _normalise(content: bytes) -> bytes:
    """Ignore platform line-ending conversion while preserving all code."""
    return content.replace(b"\r\n", b"\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to a checkout of giancarlopascali-hub/EDOS_opt",
    )
    args = parser.parse_args()
    source = args.source.resolve() if args.source else None

    lock = json.loads((HERE / "edos.lock").read_text(encoding="utf-8"))
    manifest = json.loads((HERE / "alignment.yml").read_text(encoding="utf-8"))
    product_root = (REPOSITORY / manifest.get("product_root", ".")).resolve()
    commit = lock["commit"]

    if source is not None and not (source / ".git").exists():
        print(f"EDOS source checkout not found: {source}", file=sys.stderr)
        return 2

    mismatches: list[str] = []
    for relative_path in manifest["shared_exact"]:
        target = product_root / relative_path
        if not target.is_file():
            mismatches.append(f"MISSING {relative_path}")
            continue
        actual = target.read_bytes()
        expected_digest = manifest.get("expected_sha256", {}).get(relative_path)
        if source is not None:
            try:
                expected = _git_bytes(source, commit, relative_path)
            except RuntimeError as exc:
                print(f"Invalid EDOS lock or source checkout: {exc}", file=sys.stderr)
                return 2
            source_digest = _digest(expected)
            if expected_digest and expected_digest != source_digest:
                print(f"Manifest hash is stale for {relative_path}", file=sys.stderr)
                return 2
            expected_digest = source_digest
        if not expected_digest:
            print(f"No expected hash recorded for {relative_path}", file=sys.stderr)
            return 2
        actual_digest = _digest(actual)
        if actual_digest != expected_digest:
            mismatches.append(
                f"DRIFT {relative_path} (expected {expected_digest[:12]}, got {actual_digest[:12]})"
            )

    for category in ("shared_with_extensions", "product_owned"):
        for relative_path in manifest.get(category, []):
            if not (product_root / relative_path).exists():
                mismatches.append(f"MISSING {category} entry {relative_path}")

    if mismatches:
        print("Alignment check failed:")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
        return 1

    print(f"Alignment verified against EDOS {commit}")
    print(f"  exact shared files: {len(manifest['shared_exact'])}")
    print(f"  extension files: {len(manifest.get('shared_with_extensions', []))}")
    print(f"  product-owned entries: {len(manifest.get('product_owned', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
