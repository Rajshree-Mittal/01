# Graph-Based PIDA + Contrastive Learning

**Imbalanced IoT attack detection pipeline**

`tabular data` · `SimCLR` · `KNN augmentation` · `imbalanced classes`

---

## Overview

A multi-stage pipeline for IoT intrusion detection — built around the insight that standard balancing techniques don't play well with rare attack classes. PIDA handles minority class augmentation using local graph structure; SimCLR then learns representations without relying on those labels. The two stages are fused before final classification.

---

## Pipeline stages

### 01 — Data aggregation

Multiple CSV sources merged into a single dataset. Increases class diversity before any balancing happens.

### 02 — 03 — Label engineering + preprocessing

Binary labels (normal / attack) and multiclass attack types. Drops IP addresses, timestamps, and IDs. Applies `StandardScaler` — needed for the KNN step to be meaningful.

### 04 — Binary balancing

Lightweight oversampling of the minority class with additive Gaussian noise. Simpler than SMOTE — no triangulation, just perturbation.

> *noise-based*

### 05 — PIDA augmentation

Core contribution. For each minority sample, builds a local KNN neighborhood and interpolates toward neighbors:

```
x_new = x_i + α(x_nbr − x_i)
```

Noise is added for diversity. Targets the rarest attack classes specifically.

> *graph-inspired · density-aware*

### 06 — Contrastive learning (SimCLR)

Two augmented views of each sample are passed through a shared encoder. Training minimizes distance between same-sample views, maximizes distance between different ones. No labels used.

> *self-supervised*

### 07 — Feature fusion

Original scaled features concatenated with learned SimCLR embeddings. Richer input space for downstream classifiers.

### 08 — 09 — Multiclass + final binary rebalancing

A second pass of noise-based augmentation ensures equal representation across attack types, then a final binary rebalance to normalize the normal-vs-attack ratio.

---

## PIDA — closer look

### Why not just SMOTE?

SMOTE builds synthetic samples by interpolating between random minority pairs without considering local density. PIDA uses a KNN graph to restrict interpolation to the local neighborhood, which matters for high-dimensional IoT traffic data where rare classes may occupy thin manifolds.

### Closest prior work

> *GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks*
> Zhao et al., 2021 — neighborhood-based synthesis, focus on rare classes, graph-structured augmentation

---

## Design decisions

**Why contrastive learning**

Attack labels in IoT datasets are often noisy or coarse. SimCLR learns structure without relying on them — the embeddings capture variance that supervised loss functions miss.

**Why multi-stage balancing**

Binary balancing first prevents the contrastive step from being dominated by attack traffic. Multiclass rebalancing happens after fusion, once embeddings are stable.

---

## Contributions

1. Graph-inspired PIDA augmentation adapted for tabular IoT data — no GNN required
2. SimCLR integrated into a tabular preprocessing pipeline, not just vision tasks
3. Multi-stage imbalance strategy that separates binary and multiclass balancing concerns

---

## Known limitations

- Synthetic sample quality isn't validated against a held-out distribution — PIDA may overfit to local noise
- Noise-based augmentation can blur class boundaries if α is too large or neighborhoods are heterogeneous
- No actual graph construction — "graph-inspired" means neighborhood structure only, not edge features or message passing
