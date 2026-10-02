"""Deterministic benchmark workspace.

Static text files are copied from fixtures/filesystem/. Binary files (PDF,
PNG, ZIP, CSV) are generated from fixed seeds so every build is
byte-identical. Modification times are pinned so date-based tasks give the
same answer on any machine and on any day.
"""

from __future__ import annotations

import hashlib
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tests.test_filesystem import make_pdf

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "filesystem"

OLD_MTIME = datetime(2020, 6, 1, 12, 0).timestamp()
RECENT_MTIME = datetime(2025, 6, 1, 12, 0).timestamp()

# Files whose mtime is OLD_MTIME; everything else gets RECENT_MTIME.
OLD_FILES = {"Downloads/notes.txt", "Downloads/nested/old_report.pdf", "Misc/random.txt"}

PDF_TEXT = {
    "Downloads/report.pdf": "Quarterly revenue grew 12 percent in Q3",
    "Downloads/nested/old_report.pdf": "Legacy infrastructure report from 2019",
}
BINARY_SIZES = {
    "Downloads/image.png": (b"\x89PNG\r\n\x1a\n", 1_200_000),
    "Downloads/archive.zip": (b"PK\x03\x04", 3_000_000),
}
CSV_PATH, CSV_SIZE = "Misc/data.csv", 600_000


@dataclass
class Workspace:
    root: Path
    manifest: dict[str, str]  # relative path -> sha256 at creation time

    def path(self, rel: str) -> Path:
        return self.root / rel

    def current_hashes(self) -> dict[str, str]:
        return hash_tree(self.root)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }


def _deterministic_bytes(seed: str, size: int) -> bytes:
    return random.Random(seed).randbytes(size)


def _csv(size: int) -> bytes:
    rows, i = ["id,value,label"], 0
    length = len(rows[0]) + 1
    while length < size:
        rows.append(f"{i},{(i * 7919) % 10007},row-{i}")
        length += len(rows[-1]) + 1
        i += 1
    return ("\n".join(rows) + "\n").encode()[:size]


def build_workspace(root: Path) -> Workspace:
    """Create a fresh workspace under `root` (which must not exist yet or be empty)."""
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FIXTURE_DIR, root, dirs_exist_ok=True)
    for rel, text in PDF_TEXT.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        make_pdf(target, text)
    for rel, (header, size) in BINARY_SIZES.items():
        (root / rel).write_bytes(header + _deterministic_bytes(rel, size - len(header)))
    (root / CSV_PATH).write_bytes(_csv(CSV_SIZE))

    for path in root.rglob("*"):
        if path.is_file():
            rel = str(path.relative_to(root))
            stamp = OLD_MTIME if rel in OLD_FILES else RECENT_MTIME
            os.utime(path, (stamp, stamp))
    return Workspace(root=root.resolve(), manifest=hash_tree(root))
