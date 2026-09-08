# image_search

Give it a folder of pictures and a sample picture; it returns the most
similar pictures in the folder. No training involved.

```bash
uv sync
uv run python scripts/get_data.py                     # 100 cat photos
uv run python -m image_search.index data/library      # build the index
uv run python -m image_search.search data/queries/persian_query.jpg -k 5
```

```
 1. 0.857  Persian_106.jpg
 2. 0.848  Persian_107.jpg
 3. 0.835  Persian_104.jpg
```

All correct breed, out of 10 breeds, on a photo not in the library.
Nothing was trained.

## How it works

```
                  ┌─ index (once, slow) ─────────────────┐
data/library/*.jpg → model → N numbers per image → index/<backend>/vectors.npy
                  └──────────────────────────────────────┘

                  ┌─ search (per query, fast) ───────────┐
query.jpg         → model → N numbers → vectors @ query → top-k
                  └──────────────────────────────────────┘
```

The one idea worth holding on to: a neural network trained on images
learns, layer by layer, to turn pixels into a description of *content*.
Chop off its last layer and what remains is a vector where direction
means meaning. Similar pictures point the same way. Search is then just
measuring angles — a matrix multiply, not machine learning.

Everything runs **locally**. The models are files of weights in
`~/.cache` (CLIP 578 MB, ResNet50 98 MB); no image ever leaves the
machine. Verify it yourself:

```bash
unshare -rn uv run python -m image_search.search <query> -k 3
```

That runs with no network interfaces at all, and still works.

## Backends

| `--backend` | model | dim | trained on | objective |
|---|---|---|---|---|
| `clip` | OpenCLIP ViT-B/32 | 512 | 2B image–caption pairs | match image to its caption |
| `resnet` | ResNet50 | 2048 | ImageNet-1k, 1.2M photos | pick 1 of 1000 labels |
| `resnet18` | ResNet18 | 512 | ImageNet-1k | pick 1 of 1000 labels |

Each gets its own folder under `index/`, so they coexist and can be
compared. `embed.py` is arranged so the two classes share all plumbing —
read `_TorchBackend` first, then notice the subclasses differ in only
three things: which weights, which preprocessing, which forward call.

```bash
uv run python -m image_search.index data/library --backend resnet
uv run python -m image_search.search <query> --backend resnet
```

## Measuring, instead of eyeballing

```bash
uv run python scripts/evaluate.py --labels
```

Because the library images are *already* embedded, each one can act as a
query against all the others — "leave-one-out" — with no model runs at
all. The whole evaluation is one `V @ V.T` matrix multiply. Ground truth
comes from the filenames (`Persian_106.jpg` → `Persian`), which is why
this dataset was chosen.

```
leave-one-out over 102 images, k=5

backend      dim    top-1      p@5
----------------------------------
clip         512    0.725    0.606
resnet      2048    0.794    0.647
resnet18     512    0.755    0.600
```

### Reading this honestly

**ResNet50 looks best — but the result does not support that claim.**
Three reasons, and this is the actual lesson of this step:

1. **Not significant.** The two models disagree on only 23 images (CLIP
   right 8 times, ResNet right 15). Exact binomial test: **p = 0.21**.
   With 102 images, a 7-point gap is noise. Any honest comparison needs
   a bigger library.
2. **The benchmark is rigged toward ResNet.** ImageNet-1k contains
   `Persian cat` and `Egyptian cat` as literal classes — ResNet was
   *trained* to separate exactly these. It scores 1.00 on both. That
   advantage will not exist for windmills, which ImageNet has never
   heard of.
3. **Bigger isn't better.** ResNet50's 2048 dims cost 4× the storage of
   CLIP's 512 for a ~0 (statistically) gain, and resnet18 matches
   resnet50's p@5 with a quarter of the parameters.

The per-breed split (`--labels`) is more informative than the average:
ResNet is perfect on **Bombay** (a solid-black cat — pure appearance
cue) but only 0.50 on **Bengal**, where CLIP gets 0.75. That is the
training objective showing through: ResNet groups by *"same kind of
object, same texture"*, CLIP by *"same kind of picture"*.

### The scores are not comparable across models

Same two queries, both backends:

| query | clip | resnet50 |
|---|---|---|
| Persian cat (right answer exists) | **0.857** | **0.787** |
| beagle (no dog in the library at all) | 0.533 | 0.224 |

Both models rank sensibly, but they live on different scales, and the
*gap* between a hit and a miss differs wildly — CLIP separates by 0.32,
ResNet by 0.56. So a threshold like "only show results above 0.7" tuned
on one backend is meaningless on the other. If you ever add a
"no good match found" behaviour, that cutoff has to be calibrated per
model, from data, not guessed.

Also worth noticing *what* each reached for when no dog was available:
CLIP picked **Bengal** cats (spotted, tan, dog-ish colouring — it went
for the overall look of the picture), ResNet picked **British Shorthair
and Maine Coon** (large, solid, front-facing animal shapes).

## Files

| File | Job |
|---|---|
| `src/image_search/embed.py` | image → vector. Both backends. The only file that knows about neural nets. |
| `src/image_search/store.py` | reading/writing the index. The seam Qdrant replaces next. |
| `src/image_search/index.py` | walk a folder, embed what changed, save. |
| `src/image_search/search.py` | embed the query, rank by cosine similarity. |
| `scripts/get_data.py` | fetch a starter library. |
| `scripts/evaluate.py` | leave-one-out top-1 and p@k per backend. |

## Why these tools

- **uv** — resolves and installs in seconds where pip takes minutes, and
  writes a `uv.lock` so this rebuilds identically next year.
- **CPU torch wheel** — pinned via `[tool.uv.sources]`. The default PyPI
  torch bundles ~2.5 GB of CUDA a CPU-only machine cannot use; the `cpu`
  index gives a ~200 MB wheel. Delete those two blocks when a GPU shows up.
- **open_clip** rather than the one-line `sentence-transformers` wrapper —
  you can `print(backend.model.visual)` and read the real architecture.
- **`weights.transforms()`** rather than hand-written resize/normalise —
  feeding a model differently-scaled pixels than it trained on degrades
  vectors silently, with no error.
- **`.npy` + `.json`** rather than a database — at 100 images a database
  would hide the mechanism behind an API.
- **ruff, line-length 79** in `pyproject.toml` — your editor already
  enforced 79; declaring it means IDE, ruff and any future CI agree.
- **No Docker yet** — Docker makes a *service* reproducible for *someone
  else*. There is neither yet, so it would only add a build step between
  you and the code.

## Measured on this machine (CPU)

| backend | speed | 40k windmills | index size at 40k |
|---|---|---|---|
| clip | 19 img/s | ~35 min | 80 MB |
| resnet50 | 12 img/s | ~55 min | 320 MB |
| resnet18 | 37 img/s | ~18 min | 80 MB |

Search over 40k is one 40000×512 matmul — single-digit milliseconds.
Brute force stays viable far longer than people expect; Qdrant earns its
place for persistence, incremental updates, filtering and running as a
separate service more than for raw speed at this size.

## One MLOps catch to remember

`uv.lock` pins your *code* dependencies, but the model weights live
outside it, in `~/.cache`. That's why `embed.py` names
`pretrained="laion2b_s34b_b79k"` explicitly rather than taking a default —
the string is the identity of the weights. On a fresh machine with no
network, the first index fails. Those weights need shipping or fetching
too. That problem lands at the Docker step.

## Things to try

```bash
# A breed that isn't in the library at all — what does each model reach for?
cp data/_raw/oxford-iiit-pet/images/beagle_10.jpg data/queries/
uv run python -m image_search.search data/queries/beagle_10.jpg -k 5
uv run python -m image_search.search data/queries/beagle_10.jpg -k 5 --backend resnet

# Query an image that IS in the library — similarity should be ~1.000

# Look at the models
uv run python -c "
from image_search.embed import CLIPBackend, ResNetBackend
print(CLIPBackend().model.visual)"

# Grow the library to 370 images (all 37 breeds) and re-run evaluate.
# Does the clip-vs-resnet gap survive a bigger sample?
uv run python scripts/get_data.py --breeds 37 --per-breed 10
uv run python -m image_search.index data/library --backend clip
uv run python -m image_search.index data/library --backend resnet
uv run python scripts/evaluate.py
```

## Next steps (pick an order)

- **FastAPI + thumbnail page** — actually *see* results instead of reading
  filenames. Where "which model is better" stops being abstract.
- **Qdrant** instead of `.npy` — a real vector DB, HNSW, incremental adds.
- **Docker compose** — app + Qdrant starting together on someone else's
  machine, and the weights problem above.
- **Later:** text→image search (CLIP only — that's what the shared
  image/text space buys), fine-tuning for fine-grained similarity.
