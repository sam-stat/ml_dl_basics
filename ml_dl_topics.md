## 1. Deep Learning / Neural Networks

### Foundations
- Loss functions (squared error, cross-entropy / logistic loss)
- Forward pass and backward pass (backpropagation)
- Stochastic Gradient Descent (SGD)
- Activation functions: Sigmoid, Tanh, ReLU, Leaky ReLU, GELU
- Vanishing and exploding gradients
- Weight initialization (Xavier, He)
- Dropout
- Batch Normalization, Layer Normalization
- Optimizers: Momentum, RMSProp, Adam, AdamW
- Learning rate schedules (step decay, cosine, warm-up)

### Word Embeddings
- Word2Vec (CBOW, Skip-Gram)
- Negative sampling
- Contrastive learning
- GloVe

### Sequence Models
- RNN
- LSTM
- GRU
- ELMo
- FastText
- Seq2Seq

### Convolutional Networks
- CNN
- Pooling (max, average)
- Padding, stride
- Transfer learning (ResNet, EfficientNet)

### Transformers
- Self-attention
- Multi-head attention
- Positional encoding
- Transformer block (attention + FFN + residual + layer norm)
- Encoder, Decoder, Encoder-Decoder (BERT, GPT, T5)
- Tokenization (BPE, WordPiece, SentencePiece)

### LLM Applications
- Vector search (FAISS, Pinecone)
- RAG (Retrieval-Augmented Generation)
- Agents vs. workflows
- Prompt engineering (few-shot, chain-of-thought)
- Fine-tuning, LoRA
- LLM evaluation (perplexity, BLEU, ROUGE, LLM-as-judge)

---

## 2. Classical Machine Learning

### Dimensionality Reduction
- PCA
- SVD
- NMF
- t-SNE
- UMAP

### Boosting and Ensembles
- AdaBoost
- XGBoost
- CatBoost
- LightGBM
- Random Forest
- Bagging

### Classification
- K-Nearest Neighbors (K-NN)
- Naive Bayes
- Logistic Regression
- Support Vector Machines (SVM)

### Clustering
- K-Means
- K-Medoids
- Hierarchical clustering (agglomerative, divisive)
- DBSCAN
- Gaussian Mixture Models (GMM) with EM

### Model Selection and Evaluation
- AIC, BIC
- Probability calibration (Platt scaling, isotonic regression)
- Cross-validation (k-fold, stratified)
- Metrics: precision, recall, F1, ROC-AUC, PR-AUC, RMSE, MAE, R²

### Monitoring and Drift
- PSI (Population Stability Index)
- KS (Kolmogorov-Smirnov) statistic

### Feature Engineering
- Categorical encodings (target, frequency, hashing, leave-one-out)
- Scaling (standardization, min-max, robust)
- Missing value imputation

### Interpretability
- SHAP
- LIME
- Permutation importance

---

## 3. Reinforcement Learning

### Multi-Armed Bandits
- ε-greedy
- UCB (Upper Confidence Bound)
- Thompson Sampling

### Value-Based RL
- Q-Learning
- SARSA
- Deep Q-Networks (DQN)

### Policy-Based and Actor-Critic
- REINFORCE (policy gradient)
- Actor-Critic
- PPO, A2C

---

## 4. Optimization

### Problem Types
- Linear Programming (LP)
- Integer / Mixed-Integer Programming (IP / MIP)
- Non-Linear Programming (NLP)
- Convex optimization

### Exact Methods
- Branch and Bound
- Cutting planes
- Lagrangian relaxation, duality

### Heuristic and Evolutionary Methods
- DEAP (genetic algorithms in Python)
- Simulated annealing
- Tabu search
