# Agents & LLMs

---

## Large Language Models (LLMs)

A **Large Language Model** is a Transformer decoder trained on massive text corpora to predict the next token. The same architecture as [Transformers & Attention](06_transformers.md), but scaled to billions of parameters.

### Training

**Pre-training:** autoregressive language modelling on internet-scale text ($\sim 10^{12}$ tokens):

$$\mathcal{L} = -\sum_{t} \log P(w_t \mid w_1, \ldots, w_{t-1};\; \theta)$$

**Instruction fine-tuning (SFT):** further train on curated (instruction, response) pairs — teaches the model to follow instructions.

**RLHF (Reinforcement Learning from Human Feedback):** align the model to human preferences using a reward model trained on human comparisons.

### Key Properties

| Property | Explanation |
|----------|-------------|
| **In-context learning** | Can solve new tasks from a few examples in the prompt — no weight update needed |
| **Chain-of-thought** | Prompting with step-by-step reasoning improves multi-step problems |
| **Hallucination** | LLMs generate plausible-sounding but incorrect facts — they have no external memory |
| **Knowledge cutoff** | Training data has a fixed date — no knowledge of events after that |

---

## RAG — Retrieval Augmented Generation

RAG (Lewis et al. 2020) solves the hallucination and knowledge cutoff problems by **retrieving** relevant documents at query time and injecting them into the prompt.

### Core Idea

$$\text{LLM input} = \underbrace{[\text{retrieved context}]}_{\text{fresh, specific knowledge}} + \underbrace{[\text{user query}]}_{}$$

The LLM generates an answer **grounded** in the retrieved context rather than relying solely on training memory.

### Pipeline

```
User Query
    ↓
Embed query  →  q ∈ ℝᵈ   (e.g. using sentence-BERT)
    ↓
Vector search  →  top-k documents by cosine similarity
    (index built offline from your knowledge base)
    ↓
Augmented prompt:
    "Context:\n{doc₁}\n{doc₂}\n...\nQuestion: {query}\nAnswer:"
    ↓
LLM generates grounded answer
```

### Indexing Phase (Offline)

1. Chunk documents into $\sim$256 token passages
2. Embed each chunk: $d_i = \text{Encoder}(\text{chunk}_i) \in \mathbb{R}^d$
3. Store in a vector index (FAISS, Pinecone)

### Retrieval (Online)

$$\text{score}(q, d_i) = \frac{q \cdot d_i}{\|q\|\,\|d_i\|} \quad \text{(cosine similarity)}$$

Return the $k$ documents with highest score.

### RAG vs Fine-tuning

| | RAG | Fine-tuning |
|--|-----|------------|
| Update knowledge | At query time (dynamic) | At training time (static) |
| Cost | Low — just index documents | High — GPU, annotated data |
| Hallucination | Lower — grounded in retrieved docs | Higher |
| Best for | Factual Q&A, private knowledge bases | Style, tone, task-specific behaviour |

---

## Workflows vs Agents

### Workflow (Predefined Steps)

A workflow is a **fixed sequence of steps** designed by the developer. The LLM executes each step in order; the overall control flow is deterministic.

```
User input
    → Step 1: classify intent
    → Step 2: route to appropriate handler
    → Step 3: generate response
    → Step 4: post-process / format
    → Output
```

**Characteristics:**
- Control flow determined by developer, not model
- Same input → same sequence of steps
- Reliable, auditable, easy to debug
- Best for: well-defined, repeatable tasks (customer support routing, document processing)

### Agent (Model-Driven Actions)

An agent is a system where the **LLM itself decides** which tools to call and in what order, in a loop, until the goal is met.

```
User input
    ↓
LLM reasons → decides next action → calls tool
    ↓
Observe result
    ↓
LLM reasons again → next action or done
    ↑_____________↑
    (loop)
```

**Characteristics:**
- Control flow determined by model at runtime
- Flexible: can handle novel situations
- Harder to audit; can fail in unexpected ways
- Best for: complex, open-ended tasks (coding assistant, research agent, data analysis)

### Comparison

| Feature | Workflow | Agent |
|---------|---------|-------|
| Steps fixed in advance | Yes | No |
| Model decides next action | No | Yes |
| Tools | Optional | Required |
| Reliability | High | Variable |
| Flexibility | Low | High |
| Debugging | Easy | Hard |

### Practical Rule

> Start with a workflow. Move to an agent only when the task requires dynamic decision-making that cannot be captured in a fixed sequence.

---

## Tool Use

LLMs can be given **tools** (functions) that they can call during generation. The model outputs a structured tool call; the system executes it and returns the result.

Common tools:
- Web search
- Code execution (Python REPL)
- Database queries
- API calls

**Tool call format (e.g. OpenAI function calling):**
```json
{
  "name": "search_web",
  "arguments": {"query": "current gold price"}
}
```

The model sees the result and continues generating.

---

## Vector Search (FAISS / Pinecone)

Used in RAG retrieval and semantic search.

**FAISS** (Facebook AI Similarity Search):
- Library for exact and approximate nearest-neighbour search
- Supports CPU and GPU; good for local or batch use
- Index types: `Flat` (exact), `IVF` (inverted index, approximate), `HNSW` (graph-based, fast)

**Pinecone / Weaviate / Qdrant:**
- Managed vector databases
- Handle indexing, replication, scaling automatically
- Production use cases with $10^8+$ vectors

**Similarity metric:**
$$\text{cosine similarity}(a, b) = \frac{a \cdot b}{\|a\|\,\|b\|}, \qquad \text{dot product} = a \cdot b$$

Cosine similarity is preferred when embedding magnitudes are not meaningful; dot product when they are (e.g. ColBERT).
