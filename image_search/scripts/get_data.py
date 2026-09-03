"""Fetch a small starter photo library.

    uv run python scripts/get_data.py            # 100 images, 10 breeds
    uv run python scripts/get_data.py --per-breed 50 --breeds 37   # bigger

Uses Oxford-IIIT Pet: 7390 photos of 37 cat and dog breeds. Two reasons
for this dataset rather than something generic:

  * The breed is in every filename (Abyssinian_101.jpg), so we get free
    ground truth. At v2 that lets us *measure* quality -- "does the top-5
    contain the same breed?" -- instead of squinting at results.
  * Telling two spaniel breeds apart is a fine-grained problem, the same
    shape as telling two windmills apart. Easier problems (cat vs car)
    would make the model look better than it is.

The download is ~800 MB one-off. It lands in data/_raw/ and stays there,
so you can rebuild a bigger library later without downloading again.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path


def breed_of(filename: str) -> str:
    """Abyssinian_101.jpg -> Abyssinian"""
    return filename.rsplit("_", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/_raw"))
    parser.add_argument("--library", type=Path, default=Path("data/library"))
    parser.add_argument("--breeds", type=int, default=10)
    parser.add_argument("--per-breed", type=int, default=10)
    args = parser.parse_args()

    from torchvision.datasets import OxfordIIITPet

    print(f"downloading Oxford-IIIT Pet into {args.raw_dir} (~800 MB, once)...")
    OxfordIIITPet(root=str(args.raw_dir), split="trainval", download=True)

    images = sorted((args.raw_dir / "oxford-iiit-pet" / "images").glob("*.jpg"))
    print(f"{len(images)} images available")

    by_breed: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        by_breed[breed_of(image.name)].append(image)

    chosen = sorted(by_breed)[: args.breeds]
    args.library.mkdir(parents=True, exist_ok=True)
    copied = 0
    for breed in chosen:
        for image in by_breed[breed][: args.per_breed]:
            shutil.copy2(image, args.library / image.name)
            copied += 1

    print(f"copied {copied} images from {len(chosen)} breeds -> {args.library}")
    print("breeds:", ", ".join(chosen))


if __name__ == "__main__":
    main()
