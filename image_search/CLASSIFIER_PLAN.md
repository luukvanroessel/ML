# Cat breed classifier — plan

Goal: learn PyTorch by training our own breed classifier, then compare it
with the embedders already in use (CLIP, ResNet).

## Data
- Oxford-IIIT Pet, cats only: 12 breeds, ~200 images each (`data/_raw/`).
- Official split: `trainval.txt` (1188 cats) / `test.txt` (1183 cats).
- Score on **test** only: the 102 library images come from trainval.
- CPU only → small images (64×64) for the from-scratch models.

## Steps
| # | What | Learn | Expect |
|---|------|-------|--------|
| 1 | `Dataset` + `DataLoader` on the official split | tensors, transforms, batching | – |
| 2 | Tiny CNN from scratch (3 conv layers) | `nn.Module`, training loop, loss, optimizer | 20–35% (chance 8%) |
| 3 | Improve: augmentation, batchnorm, dropout | overfitting, regularisation | higher, but hits a ceiling |
| 4 | Transfer learning: pretrained ResNet18, train the head, then unfreeze more layers | why pretrained features win on small data | 80–90% |
| 5 | Compare: k-NN / linear probe on CLIP and ResNet embeddings, same test split; plug our model's embeddings into `evaluate.py` | fair comparison of a classifier vs an embedder | CLIP hard to beat |

## Log
| Step | Model | Test acc | Notes |
|------|-------|----------|-------|
