"""Where the vectors live.

Two files on disk:

    index/vectors.npy   an (N, D) float32 array, row i belongs to entry i
    index/meta.json     which file each row came from, and its hash

Deliberately boring. `.npy` is numpy's own binary format, so loading is a
single mmap-able read with no parsing; JSON keeps the human-readable part
readable, so you can open it and see exactly what the program knows.

This module is the seam we replace with Qdrant at v2. Everything else in
the project talks to these four functions, so that swap stays local.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"


@dataclass
class Index:
    backend: str  # which model produced these vectors
    paths: list[str]  # image path per row, relative to the library root
    hashes: list[str]  # content hash per row, used to detect changes
    vectors: np.ndarray  # (N, D) float32, unit length

    def __len__(self) -> int:
        return len(self.paths)


def content_hash(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of the file's bytes.

    Why hash instead of using the modified time: mtime changes when a file
    is copied or synced, so we would re-embed thousands of unchanged photos.
    Bytes are the thing we actually care about. At 40k images, embedding is
    the expensive step and hashing is nearly free -- this is the trade we
    want.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def save(index: Index, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / VECTORS_FILE, index.vectors)
    meta = {
        "backend": index.backend,
        "dim": int(index.vectors.shape[1]) if len(index) else 0,
        "count": len(index),
        "entries": [
            {"path": p, "hash": h} for p, h in zip(index.paths, index.hashes)
        ],
    }
    (index_dir / META_FILE).write_text(json.dumps(meta, indent=2))


def load(index_dir: Path) -> Index | None:
    """Return the stored index, or None if there isn't one yet."""
    vectors_path = index_dir / VECTORS_FILE
    meta_path = index_dir / META_FILE
    if not (vectors_path.exists() and meta_path.exists()):
        return None

    meta = json.loads(meta_path.read_text())
    entries = meta["entries"]
    return Index(
        backend=meta["backend"],
        paths=[e["path"] for e in entries],
        hashes=[e["hash"] for e in entries],
        vectors=np.load(vectors_path),
    )
