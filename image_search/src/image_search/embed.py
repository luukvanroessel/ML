"""Turn images into vectors.

This is the only place in the project that knows anything about neural
networks. Everything downstream just sees an (N, D) array of floats.

Two backends live here. Read `_TorchBackend` first: it holds everything
they share. Then read the two subclasses and notice how little differs --
which weights, which preprocessing, which forward call. That is the
whole comparison.
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

    def embed_paths(
        self, paths: list[Path], batch_size: int = 16
    ) -> np.ndarray:
        """Return an (len(paths), dim) float32 array of unit-length vectors."""
        ...


class _TorchBackend:
    """Shared plumbing for any PyTorch vision model.

    Subclasses set `self.model`, `self.preprocess`, `self.name`, `self.dim`
    and implement `_forward`. Everything else -- batching, no_grad, moving
    to the device, L2 normalising -- is identical no matter what the model
    is, so it lives once, here.
    """

    name: str
    dim: int
    device: str

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Run a preprocessed (B, 3, H, W) batch through the model."""
        raise NotImplementedError

    def embed_paths(
        self, paths: list[Path], batch_size: int = 16
    ) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for batch in _batched(paths, batch_size):
            tensors = [self.preprocess(_load(p)) for p in batch]
            stacked = torch.stack(tensors).to(self.device)

            # no_grad: we only run the network forwards, so skip building
            # the graph PyTorch would need to compute gradients. Faster,
            # and much lighter on memory.
            with torch.no_grad():
                feats = self._forward(stacked)

            # Divide each vector by its own length, putting all of them on
            # the unit sphere. After this a dot product *is* the cosine
            # similarity, which turns search into one matrix multiply.
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vectors.append(feats.cpu().numpy().astype(np.float32))

        if not vectors:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate(vectors, axis=0)

    def _setup(self, device: str | None) -> None:
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.eval()  # turn off dropout etc. -- we are not training
        self.model.to(self.device)


class CLIPBackend(_TorchBackend):
    """OpenCLIP ViT-B/32. 512-dim vectors.

    Trained on ~2 billion (image, caption) pairs with one job: make an
    image's vector point the same way as its caption's vector, and away
    from every other caption in the batch. Nobody ever handed it a label
    like "cat". To win that game it had to build a space where *meaning*
    is direction -- because captions describe scenes, moods, styles and
    relations, not just object categories.

    Preprocessing (built by open_clip to match training exactly): resize
    short side to 224, centre-crop 224x224, normalise each colour channel
    by CLIP's training mean/std.

    Architecture: a Vision Transformer cuts the image into a 7x7 grid of
    32x32 patches, turns each into a token, and runs 12 layers of
    self-attention so every patch can see every other patch. A final
    linear projection lands it in the 512-dim space shared with text.
    """

    def __init__(self, device: str | None = None) -> None:
        import open_clip  # imported late: pulls in torch, which is slow

        self.name = "clip-vit-b-32"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
        )
        self.dim = self.model.visual.output_dim
        self._setup(device)

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        # CLIP has two towers (image and text). We only want the image one.
        return self.model.encode_image(batch)


class ResNetBackend(_TorchBackend):
    """torchvision ResNet with its classifier removed. 2048-dim (resnet50).

    The interesting contrast with CLIP is the *training objective*. This
    network was trained on ImageNet-1k: 1.2 million photos, each tagged
    with exactly one of 1000 hand-chosen labels, and scored purely on
    whether it picked the right label. Everything it learned, it learned
    because it helped answer "which of these 1000 nouns is this?".

    So its features are excellent at object category, and comparatively
    blind to anything the label set never rewarded -- scene, style, mood,
    relations between things. Watch for this when you compare: ResNet
    tends to group by "same kind of object, similar texture", CLIP by
    "same kind of picture".

    How the classifier comes off: ResNet ends with a global average pool
    (giving 2048 numbers) followed by `fc`, a single linear layer mapping
    2048 -> 1000 class scores. Replacing `fc` with `Identity` (a
    pass-through) means `model(x)` now returns those 2048 numbers instead
    of class scores. That is the entire "chop off the last layer" trick,
    and it is one line.

    Note the dimension: 2048 vs CLIP's 512. Bigger is not better here --
    it costs 4x the storage per image and buys features aimed at a
    narrower question.
    """

    def __init__(
        self, arch: str = "resnet50", device: str | None = None
    ) -> None:
        from torchvision import models

        # Each weights enum carries the exact preprocessing used in
        # training. Using `weights.transforms()` rather than hand-rolling
        # resize/normalise numbers is how you avoid silently feeding the
        # model differently-scaled pixels than it was trained on.
        weights = {
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        }[arch]

        self.name = f"{arch}-imagenet"
        self.model = getattr(models, arch)(weights=weights)
        self.preprocess = weights.transforms()

        self.dim = self.model.fc.in_features  # read it BEFORE replacing fc
        self.model.fc = torch.nn.Identity()

        self._setup(device)

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)


# Short name -> how to build it. `index --backend <name>` uses these keys,
# and the key is also the folder the index lands in.
BACKENDS = {
    "clip": CLIPBackend,
    "resnet": ResNetBackend,
    "resnet18": lambda: ResNetBackend(arch="resnet18"),
}


def get_backend(name: str = "clip") -> Backend:
    if name not in BACKENDS:
        available = ", ".join(BACKENDS)
        raise ValueError(
            f"unknown backend {name!r} (available: {available})"
        )
    return BACKENDS[name]()


def _load(path: Path) -> Image.Image:
    # convert("RGB") because a library will contain greyscale photos and
    # PNGs with alpha channels, and the models expect exactly 3 channels.
    return Image.open(path).convert("RGB")


def _batched(items: list[Path], size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
