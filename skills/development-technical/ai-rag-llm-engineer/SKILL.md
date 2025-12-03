---
name: ai-rag-llm-engineer
description: Design, build, and optimize AI systems using Large Language Models, Retrieval-Augmented Generation, and semantic search. Covers transformer architecture, embeddings, vector databases, RAG pipelines, prompt engineering, fine-tuning, and LLM agents. Use when building search systems, chatbots, knowledge bases, or any LLM-powered application.
version: 1.0.0
author: AI Overmind
tags: [llm, rag, transformers, embeddings, vector-search, semantic-search, prompt-engineering, fine-tuning, agents]
---

# AI/RAG/LLM Systems Engineer

Comprehensive guidance for designing and building AI systems powered by Large Language Models. Based on authoritative sources: AI-Powered Search (Grainger, Turnbull, Irwin), Super Study Guide: Transformers and LLMs (Amidi), Hands-On Large Language Models (Alammar, Grootendorst), and current research.

**Supporting Documents:**
- [transformers.md](transformers.md) - Transformer architecture deep dive
- [rag-systems.md](rag-systems.md) - RAG pipeline design and optimization
- [embeddings-vectors.md](embeddings-vectors.md) - Embeddings and vector search
- [fine-tuning.md](fine-tuning.md) - Fine-tuning and alignment techniques
- [agents.md](agents.md) - LLM agents and tool use
- [evaluation.md](evaluation.md) - Evaluation metrics and benchmarks

## Core Concepts

### The LLM Landscape

Large Language Models are deep neural networks trained on massive text corpora that can recognize, translate, summarize, predict, and generate text. Modern LLMs are based on the **Transformer architecture** (Vaswani et al., 2017), which applies "attention" to learn contextual representations of language.

**Key Model Categories:**

| Type | Architecture | Strengths | Examples |
|------|-------------|-----------|----------|
| **Autoencoding** | Encoder-only | Understanding, classification | BERT, RoBERTa |
| **Autoregressive** | Decoder-only | Text generation | GPT-4, LLaMA, Claude |
| **Seq2Seq** | Encoder-Decoder | Translation, summarization | T5, BART |

### The AI-Powered Search Spectrum

Modern search operates on a **personalization spectrum** from traditional keyword search to fully personalized recommendations:

```
Keyword Search ←→ Personalized Search ←→ Semantic Search ←→ Recommendations
     ↓                    ↓                    ↓                   ↓
  Term matching    User signals +        Knowledge graphs    Collaborative
                   keyword search        + keyword search      filtering
```

**Optimal search relevancy** lies at the intersection of:
- **Personalized Search**: Traditional keyword search + collaborative recommendations
- **Semantic Search**: Traditional keyword search + knowledge graphs
- **Multimodal Recommendations**: Collaborative recommendations + knowledge graphs

## Transformer Architecture

### Self-Attention Mechanism

The core innovation enabling LLMs is **self-attention**, which allows each token to attend to all other tokens in a sequence, capturing long-range dependencies.

**Key-Query-Value Computation:**
```
For each token position:
1. Compute Query (Q), Key (K), Value (V) vectors from embeddings
2. Calculate attention scores: Attention(Q,K,V) = softmax(QK^T / √d_k) × V
3. Weighted sum produces context-aware representation
```

**Properties of self-attention:**
- ✅ Captures long-range dependencies (unlike RNNs)
- ✅ Parallelizable (unlike sequential processing)
- ✅ Position-agnostic (requires positional encoding)
- ❌ Quadratic complexity O(n²) with sequence length

### Multi-Head Attention

Multiple attention heads allow the model to focus on different aspects simultaneously:
- Head 1: Syntactic relationships
- Head 2: Semantic similarity
- Head 3: Coreference resolution
- etc.

### Positional Encoding

Since self-attention is position-agnostic, positional information is added:
- **Sinusoidal**: Fixed mathematical function (original Transformer)
- **Learned**: Trainable embeddings per position
- **Rotary (RoPE)**: Rotation-based encoding (LLaMA, modern models)
- **ALiBi**: Attention with Linear Biases (no explicit encoding)

> 💡 **Deep Dive Available:** For detailed transformer internals including layer normalization, feed-forward networks, and architectural variants, see **[transformers.md](transformers.md)**

## Embeddings and Vector Search

### Word Embeddings

Embeddings are dense vector representations that capture semantic meaning:

```
Traditional (sparse):  [0, 0, 1, 0, 0, ..., 0]  # One-hot, high-dimensional
Embedding (dense):     [0.23, -0.45, 0.12, ...]  # Learned, lower-dimensional
```

**Key insight**: Similar concepts have similar vectors. Vector operations reveal relationships:
```
king - man + woman ≈ queen
```

### Embedding Models

| Model | Dimensions | Use Case |
|-------|------------|----------|
| **Word2Vec** | 100-300 | Word-level similarity |
| **Sentence-BERT** | 384-768 | Sentence similarity |
| **OpenAI text-embedding-3** | 256-3072 | General purpose, Matryoshka |
| **Cohere embed-v3** | 1024 | Multilingual |
| **BGE/E5** | 384-1024 | Open-source, fine-tunable |
| **ColBERT** | 128 | Late interaction, high quality |
| **Jina-embeddings-v3** | 1024 | Long context, late chunking |

### Vector Similarity

**Cosine Similarity** (most common):
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```
- Range: [-1, 1] (1 = identical direction)
- Normalized, ignores magnitude

**Dot Product**:
```
similarity(A, B) = A · B
```
- Faster computation
- Magnitude-sensitive

**Euclidean Distance**:
```
distance(A, B) = √(Σ(Aᵢ - Bᵢ)²)
```
- Lower = more similar

### Vector Databases and ANN Search

For production systems, exact nearest neighbor search is too slow. Use **Approximate Nearest Neighbor (ANN)** algorithms:

**HNSW (Hierarchical Navigable Small World)**:
- Graph-based index with hierarchical layers
- Excellent recall/speed tradeoff
- Memory-intensive
- Used by: Pinecone, Weaviate, Qdrant

**IVF (Inverted File Index)**:
- Clusters vectors, searches relevant clusters
- Good for very large datasets
- Trade-off: cluster count vs accuracy
- Used by: Faiss, Milvus

**Product Quantization (PQ)**:
- Compresses vectors for memory efficiency
- Some accuracy loss
- Often combined with IVF (IVF-PQ)

> 💡 **Deep Dive Available:** For vector database selection, indexing strategies, and optimization, see **[embeddings-vectors.md](embeddings-vectors.md)**

## Retrieval-Augmented Generation (RAG)

RAG combines retrieval with generation to ground LLM outputs in factual, up-to-date information.

### Why RAG?

LLMs have inherent limitations:
- **Knowledge cutoff**: Training data has a cutoff date
- **Hallucination**: May generate plausible but false information
- **No source attribution**: Can't cite where information came from
- **Static knowledge**: Can't access real-time or proprietary data

RAG addresses these by retrieving relevant context before generation.

### RAG Pipeline Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Query     │───▶│  Retriever  │───▶│  Augment    │───▶│  Generator  │
│             │    │             │    │   Prompt    │    │    (LLM)    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   Vector    │
                   │  Database   │
                   └─────────────┘
```

### Document Processing Pipeline

**1. Document Ingestion**
```
Raw Documents → Text Extraction → Cleaning → Chunking → Embedding → Indexing
```

**2. Chunking Strategies**

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Fixed-size** | Split by character/token count | Simple, predictable |
| **Sentence-based** | Split at sentence boundaries | Preserves grammar |
| **Paragraph-based** | Split at paragraph breaks | Preserves context |
| **Semantic** | Split by topic/meaning changes | Best quality, slower |
| **Recursive** | Hierarchical splitting | Long documents |

**Chunking Best Practices:**
- **Chunk size**: 256-512 tokens typical; balance context vs specificity
- **Overlap**: 10-20% overlap prevents information loss at boundaries
- **Metadata**: Preserve source, section headers, page numbers

**3. Advanced Chunking Techniques**

| Technique | Description | Best For |
|-----------|-------------|----------|
| **Meta-Chunking** | Preserve logical coherence (causal, transitional) | Complex documents |
| **Late Chunking** | Embed full doc, chunk after transformer | Context preservation |
| **Contextual Retrieval** | Prepend LLM-generated context to chunks | High-value collections |

### Advanced RAG Architectures

| Architecture | Description | Use Case |
|--------------|-------------|----------|
| **GraphRAG** | Knowledge graph + vector retrieval | Multi-hop reasoning |
| **CRAG** | Self-correcting retrieval with web fallback | Reducing hallucinations |
| **Self-RAG** | LLM decides when/what to retrieve | Adaptive retrieval |
| **Agentic RAG** | Autonomous agents manage retrieval | Complex queries |

### Retrieval Strategies

**Basic Retrieval:**
1. Embed query using same model as documents
2. Find top-K nearest neighbors
3. Return documents above similarity threshold

**Advanced Retrieval:**

| Technique | Description |
|-----------|-------------|
| **Hybrid Search** | Combine dense (vector) + sparse (BM25) retrieval |
| **Query Expansion** | Generate related queries, retrieve for all |
| **HyDE** | Generate hypothetical answer, use as query |
| **Multi-Query** | Break complex query into sub-queries |
| **Parent Document** | Retrieve child chunks, return parent context |
| **Self-Query** | LLM extracts metadata filters from query |

**Re-ranking:**
After initial retrieval, re-rank results using:
- Cross-encoder models (more accurate, slower)
- LLM-based relevance scoring
- Diversity-aware re-ranking (MMR)

### Prompt Augmentation

Structure the prompt to effectively use retrieved context:

```
System: You are a helpful assistant. Answer based on the provided context.
If the context doesn't contain the answer, say "I don't know."

Context:
{retrieved_documents}

User: {query}
```

**Best Practices:**
- Place context before the question
- Instruct model to cite sources
- Handle "no relevant context" cases
- Limit context to fit in context window

> 💡 **Deep Dive Available:** For advanced RAG patterns, evaluation, and production considerations, see **[rag-systems.md](rag-systems.md)**

## Prompt Engineering

### Prompt Structure

Effective prompts have clear structure:

```
[System Message]     - Role, constraints, output format
[Context/Examples]   - Few-shot examples, retrieved documents
[User Input]         - The actual query/task
[Output Format]      - Expected structure of response
```

### Prompting Techniques

**Zero-Shot**: Direct instruction without examples
```
Classify the sentiment of this review as positive, negative, or neutral:
"The product arrived damaged but customer service was helpful."
```

**Few-Shot**: Provide examples to guide behavior
```
Classify sentiment:
"Great product!" → positive
"Terrible experience" → negative
"It was okay" → neutral
"The product arrived damaged but customer service was helpful." →
```

**Chain-of-Thought (CoT)**: Encourage step-by-step reasoning
```
Let's solve this step by step:
1. First, identify...
2. Then, consider...
3. Finally, conclude...
```

**ReAct (Reasoning + Acting)**: Interleave reasoning with actions
```
Thought: I need to find the current population of Tokyo
Action: search("Tokyo population 2024")
Observation: Tokyo's population is approximately 14 million
Thought: Now I can answer the question
Answer: Tokyo has approximately 14 million people
```

### Prompt Engineering Best Practices

1. **Be specific**: Vague prompts yield vague results
2. **Provide context**: Include relevant background
3. **Specify format**: JSON, markdown, bullet points
4. **Use delimiters**: Clearly separate sections
5. **Include constraints**: Length limits, style requirements
6. **Iterate**: Test and refine prompts systematically

## Fine-Tuning and Alignment

### When to Fine-Tune

| Approach | When to Use |
|----------|-------------|
| **Prompt Engineering** | Quick iteration, no training data |
| **RAG** | Need current/proprietary knowledge |
| **Fine-Tuning** | Specific style, format, or domain expertise |
| **RLHF/DPO** | Align with human preferences |

### Fine-Tuning Methods

**Full Fine-Tuning:**
- Update all model parameters
- Requires significant compute
- Risk of catastrophic forgetting

**Parameter-Efficient Fine-Tuning (PEFT):**

| Method | Description | Parameters Updated |
|--------|-------------|-------------------|
| **LoRA** | Low-rank adaptation matrices | ~0.1-1% |
| **QLoRA** | LoRA + quantization | ~0.1-1% |
| **Prefix Tuning** | Learnable prefix tokens | ~0.1% |
| **Adapters** | Small bottleneck layers | ~1-5% |

### Alignment Techniques

**Supervised Fine-Tuning (SFT):**
- Train on high-quality instruction-response pairs
- Foundation for instruction-following

**RLHF (Reinforcement Learning from Human Feedback):**
1. Collect human preference data
2. Train reward model
3. Optimize policy with PPO

**DPO (Direct Preference Optimization):**
- Simpler alternative to RLHF
- Directly optimizes on preference pairs
- No separate reward model needed

> 💡 **Deep Dive Available:** For fine-tuning recipes, data preparation, and alignment details, see **[fine-tuning.md](fine-tuning.md)**

## LLM Agents

### Agent Architecture

LLM agents extend LLMs with:
- **Memory**: Short-term (context) and long-term (vector store)
- **Tools**: External capabilities (search, code execution, APIs)
- **Planning**: Task decomposition and execution

```
┌─────────────────────────────────────────────────────────┐
│                      LLM Agent                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │ Memory  │  │Planning │  │  Tools  │  │  LLM    │   │
│  │         │  │         │  │         │  │ (Brain) │   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Tool Use / Function Calling

Modern LLMs can call external functions:

```json
{
  "name": "search_database",
  "description": "Search the product database",
  "parameters": {
    "query": {"type": "string", "description": "Search query"},
    "limit": {"type": "integer", "description": "Max results"}
  }
}
```

The LLM decides when to call tools and how to use results.

### Agent Patterns

**ReAct Pattern:**
```
Thought → Action → Observation → Thought → ... → Answer
```

**Plan-and-Execute:**
```
1. Create plan with steps
2. Execute each step
3. Revise plan if needed
4. Return final result
```

**Multi-Agent Systems:**
- Coordinator agent delegates to specialists
- Agents can critique each other's work
- Enables complex workflows

> 💡 **Deep Dive Available:** For agent frameworks, memory systems, and production patterns, see **[agents.md](agents.md)**

## Evaluation

### LLM Evaluation Metrics

**Perplexity**: How confident is the model?
- Lower = more confident (but not necessarily correct)

**Task-Specific Metrics:**

| Task | Metrics |
|------|---------|
| **Translation** | BLEU, METEOR, chrF |
| **Summarization** | ROUGE-1, ROUGE-2, ROUGE-L |
| **QA** | Exact Match (EM), F1 |
| **Generation** | Human evaluation, LLM-as-judge |

### RAG Evaluation

**Retrieval Quality:**
- **Recall@K**: % of relevant docs in top K
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain

**Generation Quality:**
- **Faithfulness**: Does answer match retrieved context?
- **Relevance**: Does answer address the question?
- **Groundedness**: Is answer supported by sources?

### Hallucination Detection

Hallucinations are outputs that don't align with facts or context:
- **Factual hallucination**: Contradicts known facts
- **Faithfulness hallucination**: Contradicts provided context
- **Prompt misalignment**: Doesn't follow instructions

**Mitigation Strategies:**
1. Use RAG to ground responses
2. Instruct model to say "I don't know"
3. Implement fact-checking pipelines
4. Use lower temperature for factual tasks

> 💡 **Deep Dive Available:** For evaluation frameworks, benchmarks, and testing strategies, see **[evaluation.md](evaluation.md)**

## Design Checklist

### Requirements Analysis
- [ ] Define use case (search, QA, generation, agents)
- [ ] Identify data sources and update frequency
- [ ] Determine latency requirements
- [ ] Assess accuracy/hallucination tolerance
- [ ] Estimate query volume and scaling needs

### Architecture Decisions
- [ ] Choose base model (size, capabilities, cost)
- [ ] Select embedding model for your domain
- [ ] Design chunking strategy for your content
- [ ] Choose vector database based on scale
- [ ] Plan retrieval strategy (hybrid, re-ranking)
- [ ] Design prompt templates
- [ ] Decide on fine-tuning vs RAG vs both

### Production Readiness
- [ ] Implement caching for embeddings and responses
- [ ] Set up monitoring for latency and quality
- [ ] Create evaluation dataset
- [ ] Plan for model updates and reindexing
- [ ] Implement rate limiting and cost controls
- [ ] Design fallback strategies

## Common Anti-Patterns

### ❌ Ignoring Chunking Strategy
**Problem:** Using arbitrary chunk sizes without considering content structure
**Solution:** Analyze your documents; use semantic chunking for complex content

### ❌ Over-Retrieving Context
**Problem:** Stuffing too much context, overwhelming the model
**Solution:** Retrieve focused, relevant chunks; use re-ranking

### ❌ No Evaluation Pipeline
**Problem:** Deploying without measuring quality
**Solution:** Create golden dataset; implement automated evaluation

### ❌ Ignoring Hallucination Risk
**Problem:** Trusting LLM outputs without verification
**Solution:** Ground with RAG; implement fact-checking; set appropriate expectations

### ❌ One-Size-Fits-All Prompts
**Problem:** Using same prompt for all query types
**Solution:** Classify queries; use specialized prompts per type

### ❌ Embedding Model Mismatch
**Problem:** Using general embeddings for specialized domain
**Solution:** Fine-tune embeddings or use domain-specific models

## Technology Selection Guide

### Embedding Models
| Need | Recommendation |
|------|----------------|
| General purpose | OpenAI text-embedding-3-small |
| Cost-sensitive | BGE-small, E5-small |
| Multilingual | Cohere embed-multilingual-v3 |
| Fine-tunable | sentence-transformers |

### Vector Databases
| Need | Recommendation |
|------|----------------|
| Managed, scalable | Pinecone, Weaviate Cloud |
| Self-hosted, feature-rich | Weaviate, Qdrant |
| Embedded/lightweight | ChromaDB, LanceDB |
| Existing Postgres | pgvector |

### LLM Providers
| Need | Recommendation |
|------|----------------|
| Best quality | Claude, GPT-4 |
| Cost-effective | GPT-4o-mini, Claude Haiku |
| Self-hosted | LLaMA, Mistral |
| Fine-tuning | OpenAI, Together AI |

## References

### Books
- **"AI-Powered Search"** by Trey Grainger, Doug Turnbull, Max Irwin (Manning, 2023)
- **"Super Study Guide: Transformers and Large Language Models"** by Afshine & Shervine Amidi (2024)
- **"Hands-On Large Language Models"** by Jay Alammar, Maarten Grootendorst (O'Reilly, 2024)
- **"Quick Start Guide to Large Language Models"** by Sinan Ozdemir (Addison-Wesley, 2024)
- **"Foundations of Large Language Models"** by Tong Xiao, Jingbo Zhu (2024)

### Foundational Papers
- Vaswani et al. (2017). "Attention Is All You Need" - Original Transformer
- Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
- Brown et al. (2020). "Language Models are Few-Shot Learners" - GPT-3
- Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Khattab & Zaharia (2020). "ColBERT: Efficient and Effective Passage Search via Late Interaction"
- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
- Yao et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models"
- Rafailov et al. (2023). "Direct Preference Optimization"

### Advanced RAG Research
- Zhao et al. (2024). "Meta-Chunking: Learning Text Segmentation and Semantic Completion"
- Knollmeyer et al. (2025). "Document GraphRAG: Knowledge Graph Enhanced RAG"
- Günther et al. (2025). "Late Chunking: Contextual Chunk Embeddings"
- Singh et al. (2025). "Agentic RAG: A Survey on Agentic RAG"
- Li et al. (2025). "Towards Agentic RAG with Deep Reasoning"

### Preference Optimization
- Hong et al. (2024). "ORPO: Monolithic Preference Optimization without Reference Model"
- Meng et al. (2024). "SimPO: Simple Preference Optimization with Reference-Free Reward"

### Embeddings and Retrieval
- Jha et al. (2024). "Jina-ColBERT-v2: Multilingual Late Interaction Retriever"
- Kusupati et al. (2024). "Matryoshka Representation Learning"

### Evaluation and Benchmarks
- Hong et al. (2024). "The Hallucinations Leaderboard"
- Giuffrè et al. (2025). "Expert-Validated Retrieval-Augmented and Fine-Tuned GPT-4"

## Key Takeaway

> **RAG and LLM systems require careful orchestration of multiple components: embeddings, retrieval, generation, and evaluation. Start simple, measure everything, and iterate. The best system is one that reliably serves your users' needs while managing costs and hallucination risks.**

The goal is not to use the most advanced techniques, but to build systems that are reliable, maintainable, and aligned with your specific requirements.
