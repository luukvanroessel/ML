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
| 4b | 4a + fine-tune layer4, lr 1e-4, 5 ep | 0.90 | 0.873 | train 1.00; gap grows but val still improves |

### Step 5: embedders vs ours on test (1183 cats)
k-NN / probe: trained on the same 951 images. Retrieval: leave-one-out within test.

| Model | Dim | k-NN | Linear probe | Retrieval top-1 | p@5 |
|-------|-----|------|--------------|-----------------|-----|
| CLIP ViT-B/32 | 512 | 0.804 | 0.873 | 0.804 | 0.760 |
| ResNet50 (ImageNet) | 2048 | 0.820 | 0.861 | 0.821 | 0.767 |
| ResNet18 (ImageNet) | 512 | 0.791 | 0.850 | 0.806 | 0.744 |
| **Ours (4b)** | 512 | **0.869** | 0.872 | **0.880** | **0.860** |

Takeaways:
- Fine-tuning mostly *reshaped* the embedding space (same breed → close together): big gains on k-NN and retrieval, while the linear probe ties with CLIP. The breed information was already in CLIP's features, but less neatly clustered.
- Weakest breeds: Ragdoll (0.75, confused with Birman) and Maine Coon (0.80, confused with Ragdoll), i.e. long-haired breeds that look alike.
- Lesson learned: after a kernel restart, the saved model was the 4a one. "Frozen" layers still update their BatchNorm statistics in `train()` mode; that alone cost ~2 points of retrieval.

## Next: windmills
The real task is **instance retrieval** ("is this *the same* windmill?"), not classification: every photo is a windmill, the details decide.

Sample to ask for:
- ~300–500 photos of ~50–100 windmills, ≥2–3 photos each (different angles, seasons, years), plus some single-photo windmills as distractors
- a windmill ID per photo, e.g. in the filename (`de_adriaan_03.jpg`): free ground truth, like the breed here
- original files: EXIF/GPS present? duplicates or crops?
- permission to use; leave out photos with recognisable people

Metric: **recall@k** (is a photo of the same windmill in the top k?), not p@k: with only 2–3 photos per windmill, p@5 can never reach 1.

Options, cheap to expensive:
1. Baseline: CLIP and ResNet on the sample
2. Stronger off-the-shelf embedder (e.g. DINOv2); check the security review for new model weights first
3. Re-ranking: embeddings pick a top 20, local feature matching verifies the same object
4. Fine-tune with metric learning (contrastive/triplet loss) on the windmill IDs
5. GPS metadata to narrow down candidates
