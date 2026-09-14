"""Score every index in index/ against the breed labels, and compare.

    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py -k 10 --labels

The trick that makes this free: the library images are *already* embedded
and sitting in the index. So we can use each one as a query against all
the others -- "leave-one-out" -- without running the model even once.
The whole evaluation is one matrix multiply per backend.

Two numbers per backend:

  top-1   how often the single closest *other* image is the same breed.
          The strictest useful question: "is the nearest neighbour right?"

  p@k     of the k closest images, what fraction are the same breed.
          Closer to what you actually experience when you look at a page
          of results, where one bad hit among five is survivable.

Both use the filename as ground truth (Persian_106.jpg -> Persian), which
is the reason this dataset was chosen. Your friend's windmill photos will
have no labels, so treat this as the place to learn *how* to measure --
at that point you would hand-label a couple of hundred pairs instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from image_search.store import Index, load


def label_of(filename: str) -> str:
    """Persian_106.jpg -> Persian"""
    return Path(filename).stem.rsplit("_", 1)[0]


def score(index: Index, k: int) -> tuple[float, float, dict[str, float]]:
    """Leave-one-out top-1 accuracy, precision@k, and top-1 per label."""
    labels = np.array([label_of(p) for p in index.paths])

    # (N, N) similarity of every image against every other image. Unit
    # vectors, so this dot product is cosine similarity again.
    sims = index.vectors @ index.vectors.T

    # An image is always its own best match (similarity 1.0), which would
    # make every score a perfect 100%. Blanking the diagonal is what
    # "leave-one-out" means here.
    np.fill_diagonal(sims, -np.inf)

    # argsort ascending, take the last k, reverse -> k best per row.
    order = np.argsort(sims, axis=1)[:, -k:][:, ::-1]
    hits = labels[order] == labels[:, None]  # (N, k) booleans

    top1 = hits[:, 0].mean()
    precision_at_k = hits.mean()

    per_label = {
        label: float(hits[labels == label, 0].mean())
        for label in sorted(set(labels.tolist()))
    }
    return float(top1), float(precision_at_k), per_label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument(
        "--labels", action="store_true", help="show per-breed top-1"
    )
    args = parser.parse_args()

    found = sorted(
        d for d in args.index_dir.iterdir() if (d / "meta.json").exists()
    ) if args.index_dir.is_dir() else []
    if not found:
        raise SystemExit(
            f"no indexes in {args.index_dir} -- run image_search.index"
        )

    results = []
    for directory in found:
        index = load(directory)
        if index is None or len(index) == 0:
            continue
        top1, prec, per_label = score(index, args.k)
        results.append((directory.name, index, top1, prec, per_label))

    n = len(results[0][1])
    print(f"\nleave-one-out over {n} images, k={args.k}\n")
    print(f"{'backend':<10} {'dim':>5} {'top-1':>8} {f'p@{args.k}':>8}")
    print("-" * 34)
    for name, index, top1, prec, _ in results:
        dim = index.vectors.shape[1]
        print(f"{name:<10} {dim:>5} {top1:>8.3f} {prec:>8.3f}")

    if args.labels:
        for name, _, _, _, per_label in results:
            print(f"\n{name} top-1 per breed:")
            for label, value in sorted(per_label.items(), key=lambda x: x[1]):
                print(f"  {value:>5.2f}  {label}")


if __name__ == "__main__":
    main()
