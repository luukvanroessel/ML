"""Turn images into vectors.

This is the only place in the project that knows anything about neural
networks. Everything downstream just sees an (N, D) array of floats.

Keeping that boundary sharp is the point: at v1 we add a ResNet backend
next to the CLIP one, and nothing else in the codebase has to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import torch
from PIL import Image


class Backend(Protocol):
    """What every embedding model must offer the rest of the program."""

    name: str  # human-readable model name, printed while indexing
    dim: int  # length of one vector
    device: str  # "cpu" or "cuda"

    def embed_paths(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        """Return an (len(paths), dim) float32 array of unit-length vectors."""
        ...


class CLIPBackend:
    """OpenCLIP ViT-B/32.

    CLIP was trained on ~2 billion (image, caption) pairs with one job:
    make an image's vector point in the same direction as its caption's
    vector, and away from every other caption in the batch. Nobody ever
    told it "cat" or "windmill" as a label. To win that game it had to
    learn a vector space where *meaning* is direction.

    That is why we can use it without any training of our own: similarity
    in that space already lines up with "these pictures show the same
    kind of thing".

    Three stages run inside embed_paths:

    1. Preprocess. `self.preprocess` (built by open_clip to match exactly
       what the model saw during training) resizes the short side to 224,
       centre-crops to 224x224, converts to a tensor, and normalises each
       colour channel by CLIP's training mean/std. Feeding differently
       scaled pixels than training used quietly wrecks the vectors, which
       is why we never hand-roll this step.

    2. Encode. A Vision Transformer cuts the 224x224 image into a grid of
       32x32 patches (7x7 = 49 of them), turns each patch into a token,
       and runs 12 layers of self-attention so every patch can look at
       every other patch. Early layers end up responding to edges and
       colour, later layers to objects and composition. A final linear
       projection maps the result into the 512-dim space shared with text.

    3. Normalise. We divide each vector by its own length, so all vectors
       sit on the unit sphere. After that, a dot product *is* the cosine
       similarity -- which turns the whole search step into one matrix
       multiply later on. This is a genuine trick, not a formality.
    """

    name = "clip-vit-b-32"

    def __init__(self, device: str | None = None) -> None:
        import open_clip  # imported late: it pulls in torch, which is slow

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
        )
        self.model.eval()  # turn off dropout etc. -- we are not training
        self.model.to(self.device)
        self.dim = self.model.visual.output_dim

    def embed_paths(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for batch in _batched(paths, batch_size):
            tensors = [self.preprocess(_load(p)) for p in batch]
            stacked = torch.stack(tensors).to(self.device)

            # no_grad: we only run the network forwards, so skip building
            # the graph PyTorch would need to compute gradients. Faster,
            # and much lighter on memory.
            with torch.no_grad():
                feats = self.model.encode_image(stacked)

            feats = feats / feats.norm(dim=-1, keepdim=True)  # stage 3
            vectors.append(feats.cpu().numpy().astype(np.float32))

        if not vectors:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate(vectors, axis=0)


def get_backend(name: str = "clip") -> Backend:
    """Look up a backend by short name. v1 adds "resnet" here."""
    if name == "clip":
        return CLIPBackend()
    raise ValueError(f"unknown backend {name!r} (available: clip)")


def _load(path: Path) -> Image.Image:
    # convert("RGB") because a library will contain greyscale photos and
    # PNGs with alpha channels, and the model expects exactly 3 channels.
    return Image.open(path).convert("RGB")


def _batched(items: list[Path], size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
