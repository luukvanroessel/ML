"""Find the images most similar to a query image.

    uv run python -m image_search.search query.jpg -k 5
    uv run python -m image_search.search query.jpg --backend resnet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .embed import BACKENDS, get_backend
from .store import Index, load


def rank(index: Index, query: np.ndarray, k: int) -> list[tuple[str, float]]:
    """Return the k closest entries as (path, similarity) pairs.

    The whole search is one line of maths. Because every stored vector
    and the query are unit length, the dot product between them equals
    the cosine of the angle between them:

        cos(theta) = (a . b) / (|a| |b|)   and   |a| = |b| = 1

    So `index.vectors @ query` gives us, in a single matrix multiply, the
    similarity of the query against every image in the library at once.
    1.0 means "pointing the same way", 0.0 means unrelated.

    At 100 images this is instant. At 40k images it is a 40000x512 times
    512x1 multiply -- ~20M multiply-adds, a few milliseconds. Brute force
    stays honest for a surprisingly long time; a vector database (next
    step) earns its keep in the millions, or when you want the index to
    live somewhere other than this process's memory.
    """
    sims = index.vectors @ query

    # argsort ascending, take the tail, reverse -> k highest, best first.
    top = np.argsort(sims)[-k:][::-1]
    return [(index.paths[i], float(sims[i])) for i in top]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", type=Path, help="the sample image")
    parser.add_argument("-k", type=int, default=5, help="how many results")
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument(
        "--backend", default="clip", choices=sorted(BACKENDS)
    )
    args = parser.parse_args()

    index_dir = args.index_dir / args.backend
    index = load(index_dir)
    if index is None:
        raise SystemExit(
            f"no index in {index_dir} -- run image_search.index "
            f"--backend {args.backend} first"
        )
    if not args.query.is_file():
        raise SystemExit(f"no such file: {args.query}")

    # The query must go through the exact same model as the library, or
    # the two vectors are not in the same space and the numbers are junk.
    backend = get_backend(index.backend)
    query = backend.embed_paths([args.query])[0]

    print(f"\nquery: {args.query}")
    print(f"searching {len(index)} images ({backend.name})\n")
    results = rank(index, query, args.k)
    for position, (path, score) in enumerate(results, start=1):
        print(f"{position:>2}. {score:.3f}  {path}")


if __name__ == "__main__":
    main()
