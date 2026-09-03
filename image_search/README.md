# image_search

Give it a folder of pictures and a sample picture; it returns the most
similar pictures in the folder. No training involved.

## v0 — working

```bash
uv sync                                              # install
uv run python scripts/get_data.py                    # 100 cat photos
uv run python -m image_search.index data/library     # build the index
uv run python -m image_search.search data/queries/persian_query.jpg -k 5
```

Result on a Persian cat photo that is *not* in the library:

```
 1. 0.857  Persian_106.jpg
 2. 0.848  Persian_107.jpg
 3. 0.835  Persian_104.jpg
 4. 0.815  Persian_10.jpg
 5. 0.814  Persian_105.jpg
```

All five are the right breed, out of 10 breeds. Nothing was trained.

## How it works

```
                  ┌─ index (once, slow) ─────────────────┐
data/library/*.jpg → CLIP → 512 numbers per image → index/vectors.npy
                  └──────────────────────────────────────┘

                  ┌─ search (per query, fast) ───────────┐
query.jpg         → CLIP → 512 numbers → vectors @ query → top-k
                  └──────────────────────────────────────┘
```

The one idea worth holding on to: a neural network trained on images
learns, layer by layer, to turn pixels into a description of *content*.
Chop off its last layer and what remains is a vector where direction
means meaning. Similar pictures point the same way. Search is then just
measuring angles — which is a matrix multiply, not machine learning.

## Files

| File | Job |
|---|---|
| `src/image_search/embed.py` | image → vector. The only file that knows about neural nets. |
| `src/image_search/store.py` | reading/writing the index. The seam we swap for Qdrant at v2. |
| `src/image_search/index.py` | walk a folder, embed what changed, save. |
| `src/image_search/search.py` | embed the query, rank by cosine similarity. |
| `scripts/get_data.py` | fetch a starter library. |

## Why these tools

- **uv** — resolves and installs in seconds where pip takes minutes, and
  writes a `uv.lock` so this project rebuilds identically next year. Also
  manages the Python version itself.
- **CPU torch wheel** — pinned via `[tool.uv.sources]` in `pyproject.toml`.
  The default PyPI torch bundles ~2.5 GB of CUDA libraries a CPU-only
  machine cannot use; the `cpu` index gives a ~200 MB wheel. Delete those
  two blocks when a GPU shows up.
- **open_clip** rather than the one-line `sentence-transformers` wrapper —
  you can `print(backend.model)` and read the actual architecture, which
  is the point of this project.
- **`.npy` + `.json`** rather than a database — at 100 images a database
  would hide the mechanism behind an API. The whole search is one line of
  numpy you can read.
- **No Docker yet** — Docker's job is to make a *service* reproducible for
  someone else. Right now there is no service and no someone else, so it
  would only add a build step between you and the code. It arrives at v3,
  when the API and Qdrant need to start together on your friend's machine.

## Measured on this machine (CPU)

| | |
|---|---|
| Embedding speed | ~17 images/sec |
| So: 40,000 windmills | ~40 min, once |
| Index size | 512 floats × 4 bytes = 2 KB/image → 80 MB for 40k |
| Search over 40k | one 40000×512 matmul, single-digit ms |

Brute-force search stays viable much longer than people expect. Qdrant
earns its place for other reasons (persistence, incremental updates,
filtering, running as a separate service) more than raw speed at 40k.

## Things to try before v1

```bash
# Query with a breed that ISN'T in the library — what does it reach for?
cp data/_raw/oxford-iiit-pet/images/beagle_10.jpg data/queries/
uv run python -m image_search.search data/queries/beagle_10.jpg -k 5

# Query an image that IS in the library. Similarity should be ~1.000.

# Look at the model
uv run python -c "
from image_search.embed import CLIPBackend
b = CLIPBackend(); print(b.model.visual)"

# Add photos to data/library and re-run index — only new ones get embedded
```

## Next steps (pick an order)

- **ResNet backend** beside CLIP — compare both on the same query. Model side.
- **FastAPI + thumbnail page** — actually *see* the results instead of
  reading filenames. MLOps side.
- **Qdrant** instead of `.npy` — a real vector DB, HNSW, incremental adds.
- **recall@k** measured against the breed labels — turn "looks right" into
  a number, so later changes are provable rather than vibes.
- **Docker compose** — app + Qdrant starting together on someone else's
  machine.
- **Later:** text→image search, fine-tuning for fine-grained similarity.
