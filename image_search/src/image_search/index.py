"""Build or update the index for a folder of images.

    uv run python -m image_search.index data/library
    uv run python -m image_search.index data/library --backend resnet

Each backend gets its own folder (index/clip, index/resnet), so you can
keep several and compare them. Re-running only embeds what changed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .embed import BACKENDS, get_backend
from .store import Index, content_hash, load, save

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
}


def find_images(root: Path) -> list[Path]:
    """All image files under root, sorted so runs are reproducible."""
    return sorted(
        p
        for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
    )


def build(
    library: Path,
    index_dir: Path,
    backend_name: str = "clip",
    batch_size: int = 16,
) -> Index:
    """Embed everything in `library` that isn't already indexed."""
    files = find_images(library)
    print(f"found {len(files)} images in {library}")

    print("hashing...")
    hashes = {f: content_hash(f) for f in files}

    # Reuse vectors from the previous run of *this* backend. Vectors from
    # two different models live in different spaces, so mixing them would
    # be meaningless -- hence the guard, even though separate folders
    # make it unlikely.
    old = load(index_dir)
    reusable: dict[str, np.ndarray] = {}
    if old is not None and old.backend == backend_name:
        reusable = {h: old.vectors[i] for i, h in enumerate(old.hashes)}
    elif old is not None:
        print(f"backend changed ({old.backend} -> {backend_name}), rebuild")

    todo = [f for f in files if hashes[f] not in reusable]
    print(f"{len(files) - len(todo)} reused, {len(todo)} to embed")

    if todo:
        backend = get_backend(backend_name)
        print(
            f"loaded {backend.name} (dim={backend.dim}) "
            f"on {backend.device}"
        )
        start = time.perf_counter()
        fresh = backend.embed_paths(todo, batch_size=batch_size)
        elapsed = time.perf_counter() - start
        print(
            f"embedded {len(todo)} images in {elapsed:.1f}s "
            f"({len(todo) / elapsed:.1f} img/s)"
        )
        for file, vector in zip(todo, fresh):
            reusable[hashes[file]] = vector

    # Rebuild in `files` order. Deleted images simply drop out, because
    # we only look up hashes that are still on disk.
    vectors = np.stack(
        [reusable[hashes[f]] for f in files]
    ).astype(np.float32)
    index = Index(
        backend=backend_name,
        paths=[str(f.relative_to(library)) for f in files],
        hashes=[hashes[f] for f in files],
        vectors=vectors,
    )
    save(index, index_dir)
    print(
        f"saved {len(index)} vectors of dim {vectors.shape[1]} "
        f"-> {index_dir}"
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "library", type=Path, help="folder of images to index"
    )
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument(
        "--backend", default="clip", choices=sorted(BACKENDS)
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not args.library.is_dir():
        raise SystemExit(f"not a folder: {args.library}")
    build(
        args.library,
        args.index_dir / args.backend,
        args.backend,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
