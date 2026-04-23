# Neural Networks — Notes Index

> A structured reference for neural networks from fundamentals to modern architectures.
> All diagrams are SVG files in `diagrams/` — rendered natively on GitHub.

---

## Contents

| # | Topic | File |
|---|-------|------|
| 1 | Basic Neural Network Design | [01_basic_neural_networks.md](01_basic_neural_networks.md) |
| 2 | Training Issues & Solutions | [02_training_issues.md](02_training_issues.md) |
| 3 | Word Embeddings | [03_word_embeddings.md](03_word_embeddings.md) |
| 4 | RNNs & LSTMs | [04_rnns_lstms.md](04_rnns_lstms.md) |
| 5 | CNNs | [05_cnns.md](05_cnns.md) |
| 6 | Transformers & Attention | [06_transformers.md](06_transformers.md) |
| 7 | Architecture Guide | [07_architecture_guide.md](07_architecture_guide.md) |
| 8 | Agents & LLMs | [08_agents_llms.md](08_agents_llms.md) |

---

## Notation Convention Used Across All Notes

| Symbol | Meaning |
|--------|---------|
| $x \in \mathbb{R}^n$ | Input vector of dimension $n$ |
| $W^{(\ell)} \in \mathbb{R}^{m \times n}$ | Weight matrix at layer $\ell$ |
| $b^{(\ell)} \in \mathbb{R}^m$ | Bias vector at layer $\ell$ |
| $z^{(\ell)}$ | Pre-activation (linear transform output) at layer $\ell$ |
| $a^{(\ell)}$ | Post-activation output at layer $\ell$;  $a^{(0)} = x$ |
| $\sigma$ | Activation function (sigmoid, tanh, ReLU, …) |
| $\hat{y}$ | Model prediction |
| $\mathcal{L}$ | Loss function |
| $\eta$ | Learning rate |
| $\nabla_W \mathcal{L}$ | Gradient of loss w.r.t. weight $W$ |
| $\odot$ | Element-wise (Hadamard) product |
