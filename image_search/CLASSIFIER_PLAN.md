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
Best val acc while developing; test acc only in step 5.

| Step | Model | Val acc | Test acc | Notes |
|------|-------|---------|----------|-------|
| 2 | TinyCNN, 64px, 15 ep | 0.30 | | heavy overfit: train 0.88, val loss rising after ~6 ep |
| 3a | TinyCNN + augmentation, 30 ep | 0.36 | | gap closes (train 0.54); clear gain |
| 3b | TinyCNN2 (+ batchnorm, dropout) + aug, 30 ep | 0.38 | | within noise of 3a (val = 237 imgs, single run) |
| 4a | ResNet18 pretrained, head only, 224px, 5 ep | 0.84 | | 6k trainable params; pretraining = the big jump |
| 4b | 4a + fine-tune layer4, lr 1e-4, 5 ep | 0.90 | | train 1.00; gap grows but val still improves |
