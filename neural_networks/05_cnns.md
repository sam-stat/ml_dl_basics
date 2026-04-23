# 5. Convolutional Neural Networks (CNNs)

---

## The Core Idea

Natural images have two properties that CNNs exploit:
1. **Local structure** — edges, textures, and shapes are defined by nearby pixels, not pixels far apart
2. **Translation invariance** — a cat detector should fire whether the cat is top-left or bottom-right

A standard feedforward network applied to images would need $H \times W \times C$ input weights per neuron — impractical for high-resolution images (e.g. $224 \times 224 \times 3 = 150{,}528$ inputs). CNNs solve this with **local receptive fields** and **shared weights**.

---

## The Convolution Operation

A **filter** (kernel) $K \in \mathbb{R}^{k \times k}$ slides over the input, computing a dot product at each spatial location:

$$S(i, j) = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I(i+m,\; j+n) \cdot K(m, n)$$

- $I$ — input image (or feature map from previous layer)
- $S$ — output **feature map** (one scalar per spatial location)
- The same filter weights $K$ are applied at every position — this is **weight sharing**

For an input of size $H \times W$ with stride $s$ and padding $p$, the output feature map has size:

$$H_{\text{out}} = \left\lfloor\frac{H + 2p - k}{s}\right\rfloor + 1$$

**Multiple filters** — if we apply $C_{\text{out}}$ filters, we get $C_{\text{out}}$ feature maps stacked to form a $H_{\text{out}} \times W_{\text{out}} \times C_{\text{out}}$ output.

### Example — Vertical Edge Filter

$$K = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$

High positive response where there is a left-to-right intensity transition (vertical edge).

---

## Pooling

Pooling reduces spatial dimensions and introduces local translation invariance.

**Max pooling** (most common): take the maximum value in each $k \times k$ window:

$$P(i, j) = \max_{0 \leq m,n < k} S(si+m,\; sj+n)$$

**Average pooling**: take the mean instead of the max. Used in some architectures at the final global level (Global Average Pooling).

Pooling:
- Reduces feature map size → fewer parameters in later layers
- Preserves the strongest activation regardless of exact position

---

## Full CNN Architecture

```
Input image (H × W × C)
  │
  ├── Conv Layer (k filters, k×k kernel) + ReLU   →  feature maps
  ├── Max Pool (2×2, stride 2)                     →  halve spatial dims
  │
  ├── Conv Layer (2k filters) + ReLU               →  richer features
  ├── Max Pool (2×2, stride 2)
  │
  ├── Flatten  →  1D vector
  │
  ├── Fully Connected (FC) + ReLU
  └── Softmax  →  class probabilities
```

**What each stage learns:**
- Early conv layers: edges, corners, blobs
- Mid conv layers: textures, shapes, parts
- Deep conv layers: high-level concepts (faces, wheels, text)
- FC layers: combine features for the final decision

---

## 1×1 Convolutions

A $1 \times 1$ convolution applies a linear combination across the channel dimension only — no spatial mixing. Used to:
- Change channel depth cheaply (e.g. reduce from 256 to 64 channels)
- Add non-linearity between channels (bottleneck in ResNet, Inception)

---

## Key Architectural Features

### Residual Connections (ResNet)

Skip connections add the input directly to the output of a conv block:

$$a^{(\ell+2)} = \sigma\!\bigl(F(a^{(\ell)}, W) + a^{(\ell)}\bigr)$$

If $F$ learns zero, the block becomes an identity — the network can always fall back to a shallower version. This solves the **degradation problem** in very deep networks and enables training nets with 100+ layers.

### Batch Normalisation in CNNs

Applied after each convolution, before activation. Normalises across the batch dimension **and** spatial positions (per channel):

$$\hat{z}_{n,c,h,w} = \frac{z_{n,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \varepsilon}}$$

This is why modern CNNs (ResNet, EfficientNet) train stably without careful LR tuning.

---

## Comparison: CNN vs FC for Images

| | Fully Connected | CNN |
|--|----------------|-----|
| Parameters (224×224×3 input) | $\sim 150K$ per neuron | Shared: $k \times k \times C$ per filter |
| Translation invariance | No | Yes (via weight sharing) |
| Spatial structure exploitation | No | Yes (local receptive fields) |
| Scalability to high-res | Poor | Good |
