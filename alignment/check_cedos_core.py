"""Verify that the CEDOS numerical engines still match their source snapshot.

Line endings are normalized because Windows Git checkouts may use CRLF.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).with_name("cedos_core.sha256")


def main() -> int:
    failed = []
    checked = 0
    for record in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not record.strip() or record.lstrip().startswith("#"):
            continue
        expected, relative_path = record.split(maxsplit=1)
        path = ROOT / relative_path.strip()
        if not path.is_file():
            failed.append(f"missing: {relative_path}")
            continue
        content = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
        checked += 1
        if actual != expected:
            failed.append(f"changed: {relative_path}")
    if failed:
        print("CEDOS core alignment failed:")
        for issue in failed:
            print(f"  {issue}")
        return 1
    print(f"CEDOS core alignment passed ({checked} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
