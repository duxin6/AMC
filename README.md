# AMC: Adaptive Multi-modal Fusion for Cross-modal Music Generation

Official PyTorch implementation of the AMC framework for cross-modal music generation via adaptive multi-modal fusion and transformer-based temporal modeling.

---

## Introduction

AMC is a cross-modal music generation framework designed to jointly model:

* visual features
* audio/music features
* textual semantic information

The framework establishes a shared semantic embedding space across multiple modalities and enables semantically aligned music generation through adaptive fusion and temporal transformer learning.

The current implementation includes:

* Projection-based modality encoders
* Efficient block-wise temporal attention
* Adaptive Multi-modal Fusion (AMF)
* Cross-modal Transformer (CMT)
* Temporal Significance Filter (TSF)
* Contrastive semantic alignment training

---

## Framework Architecture

```text
Visual Features ───► Visual Encoder ───┐
                                       │
Audio Features ────► Audio Encoder ────┼──► AMF Fusion ─► CMT Blocks ─► TSF ─► Global Music Embedding
                                       │
Text Features ─────► Text Encoder ─────┘
```

---

# Implemented Modules

## 1. ProjectionLayer

Defined in `modules.py`.

Each modality is projected into a shared hidden embedding space using:

* Linear layers
* Batch normalization
* ReLU activation
* Dropout regularization

```python
self.net = nn.Sequential(
    nn.Linear(in_dim, out_dim),
    nn.BatchNorm1d(out_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(out_dim, out_dim),
    nn.BatchNorm1d(out_dim),
    nn.ReLU(),
    nn.Dropout(dropout)
)
```

Supported modalities:

* Visual features
* Audio features
* Text features

---

## 2. BlockwiseAttention

The framework adopts an efficient block-wise attention mechanism to reduce the computational complexity of long sequence modeling.

Instead of global full attention:

```text
O(T²)
```

the complexity becomes:

```text
O(TB)
```

where:

* `T` denotes sequence length
* `B` denotes block size

Implementation:

```python
x_blocked = x.reshape(b * num_blocks, self.block_size, d)
attn_out, _ = self.attn(x_blocked, x_blocked, x_blocked)
```

This design enables scalable modeling for long musical sequences.

---

## 3. Adaptive Multi-modal Fusion (AMF)

The AMF module dynamically estimates the reliability of different modalities and suppresses noisy information.

Implementation:

```python
s = self.phi(modalities)
w = F.softmax(s / self.temp, dim=1)
rectified = modalities * w
```

Key advantages:

* Dynamic modality weighting
* Noise suppression
* Semantic reliability estimation
* Cross-modal information filtering

---

## 4. Cross-modal Transformer (CMT)

The framework employs stacked Cross-modal Transformer blocks for global dependency modeling.

Each block contains:

* Temporal attention
* Channel-wise attention
* Feed-forward MLP
* Residual connections
* Layer normalization

Implementation:

```python
self.temp_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
self.chan_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
```

The model currently uses:

```python
self.cmt = nn.Sequential(*[CMTBlock(hid_dim) for _ in range(4)])
```

which stacks 4 transformer fusion blocks.

---

## 5. Temporal Significance Filter (TSF)

The TSF module aggregates temporal information using learnable importance weights.

Implementation:

```python
weights = F.softmax(self.alpha_net(x), dim=1)
g = torch.sum(weights * x, dim=1)
```

The module enables the model to focus on:

* important rhythmic events
* harmonic transitions
* salient temporal structures

---

# Model Pipeline

The complete forward process is implemented in `model.py`.

```python
v = self.v_enc(v_feat)
a = self.block_attn(self.a_enc(a_feat))
t = self.t_enc(t_feat)

v_r, a_r, t_r = self.amf(v, a, t)

x = self.cmt(v_r + a_r + t_r)

g_music = self.tsf(x)
```

Pipeline summary:

1. Modality encoding
2. Block-wise temporal attention
3. Adaptive fusion
4. Cross-modal transformer interaction
5. Temporal semantic aggregation

---

# Training Objective

Training is implemented in `train.py`.

The framework jointly optimizes:

* reconstruction loss
* cross-modal contrastive alignment loss

## Contrastive Loss

```python
logits = torch.matmul(g_m, g_t.T) / temp
loss = F.cross_entropy(logits, labels)
```

This objective aligns:

* global music embeddings
* text semantic embeddings

within a shared semantic space.

---

# Installation

## Requirements

* Python >= 3.9
* PyTorch >= 2.0
* CUDA >= 11.8

Install dependencies:

```bash
pip install torch torchvision torchaudio
```

---

# Project Structure

```text
AMC/
├── model.py
├── modules.py
├── train.py
└── README.md
```

---

# Usage

## Training

Run the training script:

```bash
python train.py
```

The current demo implementation randomly generates:

* visual features
* audio features
* text embeddings

for validating the training pipeline.

---

# Input Dimensions

Default dimensions used in the implementation:

| Modality       | Dimension |
| -------------- | --------- |
| Visual Feature | 1024      |
| Audio Feature  | 512       |
| Text Feature   | 768       |
| Hidden Feature | 512       |

---

# Example Batch Shapes

```python
v = torch.randn(8, 64, 1024)
a = torch.randn(8, 64, 512)
t = torch.randn(8, 64, 768)
```

where:

* batch size = 8
* sequence length = 64

---

# Future Work

Potential future extensions include:

* diffusion-based music generation
* AudioLDM integration
* video-conditioned soundtrack synthesis
* symbolic music generation
* large-scale multimodal pretraining
* real-world dataset support

---



