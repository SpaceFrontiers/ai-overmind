# Embeddings and Vector Search

This document covers embedding models, vector similarity, and vector database selection for AI systems.

## Understanding Embeddings

### What Are Embeddings?

Embeddings are dense vector representations that capture semantic meaning in a continuous vector space. Similar concepts have similar vectors.

**Traditional representations (sparse):**
```
"cat" → [0, 0, 0, 1, 0, 0, ..., 0]  # One-hot: vocab_size dimensions
"dog" → [0, 0, 1, 0, 0, 0, ..., 0]  # No similarity captured
```

**Embeddings (dense):**
```
"cat" → [0.23, -0.45, 0.12, 0.89, ...]  # 384-3072 dimensions
"dog" → [0.21, -0.42, 0.15, 0.85, ...]  # Similar vectors!
```

### The Distributional Hypothesis

"Words that occur in similar contexts tend to have similar meanings."

This principle underlies all embedding methods:
- Word2Vec learns from word co-occurrence
- BERT learns from masked language modeling
- Sentence embeddings learn from semantic similarity tasks

### Embedding Properties

**Semantic similarity:**
```
similarity("king", "queen") > similarity("king", "banana")
```

**Analogical reasoning:**
```
embedding("king") - embedding("man") + embedding("woman") ≈ embedding("queen")
```

**Clustering:**
Similar concepts cluster together in embedding space.

## Embedding Model Types

### Word Embeddings

**Word2Vec (2013):**
- Skip-gram: Predict context from word
- CBOW: Predict word from context
- Fixed embeddings per word (no context)

**GloVe (2014):**
- Global word co-occurrence statistics
- Matrix factorization approach
- Good for analogy tasks

**FastText (2016):**
- Subword embeddings (character n-grams)
- Handles out-of-vocabulary words
- Good for morphologically rich languages

**Limitations of word embeddings:**
- No context: "bank" has same embedding in "river bank" and "bank account"
- Fixed vocabulary
- Single vector per word

### Contextual Embeddings

**ELMo (2018):**
- Bidirectional LSTM
- Different embeddings based on context
- Concatenate layers for final representation

**BERT Embeddings (2018):**
- Transformer-based
- Deeply bidirectional
- [CLS] token or mean pooling for sentences

**Limitations:**
- Not optimized for similarity tasks
- Anisotropic embedding space (clustered)

### Sentence Embeddings

**Sentence-BERT (SBERT, 2019):**
- Fine-tuned BERT for similarity
- Siamese/triplet network training
- Much faster than cross-encoder

**Training objectives:**
- Contrastive learning
- Multiple negatives ranking
- Knowledge distillation

### Modern Embedding Models

| Model | Provider | Dimensions | Context | Notes |
|-------|----------|------------|---------|-------|
| text-embedding-3-small | OpenAI | 1536 | 8191 | Cost-effective |
| text-embedding-3-large | OpenAI | 3072 | 8191 | Higher quality |
| embed-english-v3.0 | Cohere | 1024 | 512 | Fast, good quality |
| embed-multilingual-v3.0 | Cohere | 1024 | 512 | 100+ languages |
| voyage-large-2 | Voyage AI | 1536 | 16000 | Long context |
| BGE-large-en-v1.5 | BAAI | 1024 | 512 | Open source |
| E5-large-v2 | Microsoft | 1024 | 512 | Open source |
| GTE-large | Alibaba | 1024 | 512 | Open source |
| nomic-embed-text | Nomic | 768 | 8192 | Open, long context |
| mxbai-embed-large | Mixedbread | 1024 | 512 | Open source |
| jina-embeddings-v3 | Jina AI | 1024 | 8192 | Late chunking support |

### Late Interaction Models (ColBERT)

ColBERT introduces **late interaction** - a middle ground between bi-encoders and cross-encoders:

**Architecture comparison:**
```
Bi-encoder:     Query → [Encoder] → single vector
                Doc   → [Encoder] → single vector
                Similarity = dot(query_vec, doc_vec)

Cross-encoder:  [Query, Doc] → [Encoder] → relevance score
                (Must process each pair - slow)

ColBERT:        Query → [Encoder] → multiple token vectors
                Doc   → [Encoder] → multiple token vectors  
                Similarity = MaxSim(query_tokens, doc_tokens)
```

**MaxSim operation:**
```python
def maxsim(query_embeddings, doc_embeddings):
    """
    For each query token, find max similarity with any doc token.
    Sum these max similarities.
    """
    # query_embeddings: (num_query_tokens, dim)
    # doc_embeddings: (num_doc_tokens, dim)
    
    similarities = query_embeddings @ doc_embeddings.T  # (q_tokens, d_tokens)
    max_per_query = similarities.max(dim=1).values      # (q_tokens,)
    return max_per_query.sum()
```

**Benefits of ColBERT:**
- Document embeddings can be precomputed (like bi-encoder)
- Fine-grained token-level matching (like cross-encoder)
- 2 orders of magnitude faster than cross-encoder
- Better quality than single-vector bi-encoder

**ColBERT models:**
| Model | Dimensions | Context | Notes |
|-------|------------|---------|-------|
| ColBERTv2 | 128 | 512 | Original, English |
| Jina-ColBERT-v2 | 128 | 8192 | Long context, multilingual |
| ColBERT-XM | 128 | 512 | Multilingual |

**When to use ColBERT:**
- Need better quality than bi-encoder
- Can't afford cross-encoder latency
- Have storage for multi-vector representations
- Complex queries with multiple aspects

### Embedding Model Selection

**Considerations:**
1. **Quality**: MTEB benchmark scores
2. **Dimensions**: Higher = more expressive, more storage
3. **Context length**: How much text can be embedded at once
4. **Speed**: Inference latency
5. **Cost**: API pricing or compute for self-hosted
6. **Language**: Multilingual support if needed

**Decision guide:**

| Use Case | Recommendation |
|----------|----------------|
| General English | text-embedding-3-small or BGE-large |
| Multilingual | Cohere multilingual or E5-mistral |
| Long documents | voyage-large-2 or nomic-embed |
| Cost-sensitive | BGE-small or E5-small |
| Maximum quality | text-embedding-3-large or voyage-large-2 |
| Self-hosted | BGE, E5, or GTE family |

## Vector Similarity Metrics

### Cosine Similarity

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

**Properties:**
- Range: [-1, 1]
- 1 = identical direction
- 0 = orthogonal
- -1 = opposite direction
- Magnitude-invariant (normalized)

**Best for:**
- Text similarity
- When magnitude shouldn't matter
- Most common choice

### Dot Product (Inner Product)

```python
def dot_product(a, b):
    return np.dot(a, b)
```

**Properties:**
- Range: (-∞, +∞)
- Magnitude-sensitive
- Faster computation (no normalization)

**Best for:**
- Pre-normalized vectors
- When magnitude carries meaning
- Maximum inner product search (MIPS)

### Euclidean Distance (L2)

```python
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)
```

**Properties:**
- Range: [0, +∞)
- Lower = more similar
- Magnitude-sensitive

**Best for:**
- Clustering
- When absolute position matters

### Manhattan Distance (L1)

```python
def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))
```

**Properties:**
- Sum of absolute differences
- More robust to outliers than L2

### Choosing a Metric

| Metric | When to Use |
|--------|-------------|
| Cosine | Default for text, normalized embeddings |
| Dot Product | Pre-normalized, speed-critical |
| Euclidean | Clustering, spatial relationships |
| Manhattan | Sparse vectors, outlier robustness |

**Note:** For normalized vectors, cosine similarity and dot product give equivalent rankings.

## Approximate Nearest Neighbor (ANN) Algorithms

Exact nearest neighbor search is O(n) - too slow for large datasets. ANN algorithms trade accuracy for speed.

### HNSW (Hierarchical Navigable Small World)

**How it works:**
1. Build hierarchical graph with multiple layers
2. Top layers: sparse, long-range connections
3. Bottom layers: dense, local connections
4. Search: Start at top, greedily descend

```
Layer 2:  A ─────────────────── D
          │                     │
Layer 1:  A ─── B ─────── C ─── D
          │     │         │     │
Layer 0:  A ─ B ─ C ─ D ─ E ─ F ─ G
```

**Parameters:**
- `M`: Max connections per node (16-64)
- `ef_construction`: Build-time search width (100-500)
- `ef_search`: Query-time search width (50-200)

**Trade-offs:**
- ✅ Excellent recall/speed
- ✅ No training required
- ❌ High memory usage
- ❌ Slow index construction

**Used by:** Pinecone, Weaviate, Qdrant, Milvus, pgvector

### IVF (Inverted File Index)

**How it works:**
1. Cluster vectors using k-means
2. Assign each vector to nearest centroid
3. Search: Find nearest centroids, search within those clusters

```
Centroids:  C1    C2    C3    C4
            │     │     │     │
Vectors:   [v1]  [v4]  [v7]  [v10]
           [v2]  [v5]  [v8]  [v11]
           [v3]  [v6]  [v9]  [v12]
```

**Parameters:**
- `nlist`: Number of clusters (sqrt(n) to 4*sqrt(n))
- `nprobe`: Clusters to search (1-100)

**Trade-offs:**
- ✅ Lower memory than HNSW
- ✅ Good for very large datasets
- ❌ Requires training (k-means)
- ❌ Less accurate than HNSW

### Product Quantization (PQ)

**How it works:**
1. Split vector into subvectors
2. Quantize each subvector to nearest centroid
3. Store only centroid IDs (compression)

```
Original: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
Split:    [0.1, 0.2] [0.3, 0.4] [0.5, 0.6] [0.7, 0.8]
Quantize: [ID: 42]   [ID: 17]   [ID: 89]   [ID: 3]
```

**Parameters:**
- `m`: Number of subvectors (8-64)
- `nbits`: Bits per subvector (8 typical)

**Trade-offs:**
- ✅ Massive compression (32x or more)
- ✅ Enables billion-scale search
- ❌ Accuracy loss
- ❌ Requires training

### IVF-PQ (Combined)

Combines IVF clustering with PQ compression:
1. IVF for coarse search
2. PQ for compressed storage within clusters

**Best for:** Very large datasets (billions of vectors)

### Scalar Quantization (SQ)

**How it works:**
- Quantize each dimension to int8 or int4
- Simpler than PQ, less compression

**Trade-offs:**
- ✅ Simple, fast
- ✅ Good accuracy retention
- ❌ Less compression than PQ

### Algorithm Comparison

| Algorithm | Memory | Build Time | Query Time | Recall |
|-----------|--------|------------|------------|--------|
| Flat (exact) | 1x | O(1) | O(n) | 100% |
| HNSW | 1.5-2x | Slow | Fast | 95-99% |
| IVF | 1x | Medium | Medium | 90-95% |
| IVF-PQ | 0.1x | Slow | Fast | 85-95% |
| SQ | 0.25-0.5x | Fast | Fast | 95-99% |

## Vector Database Selection

### Managed Services

**Pinecone:**
- Fully managed, serverless option
- Excellent developer experience
- Hybrid search (sparse + dense)
- Metadata filtering
- Pricing: Per-pod or serverless

**Weaviate Cloud:**
- Managed Weaviate
- GraphQL API
- Built-in vectorization
- Hybrid search
- Generative search features

**Qdrant Cloud:**
- Managed Qdrant
- Rust-based, fast
- Rich filtering
- Payload storage

**Zilliz Cloud:**
- Managed Milvus
- Enterprise features
- GPU acceleration

### Self-Hosted

**Weaviate:**
- Feature-rich
- Multiple vectorizers
- GraphQL + REST API
- Docker/Kubernetes

**Qdrant:**
- High performance (Rust)
- Simple API
- Good filtering
- Docker/Kubernetes

**Milvus:**
- Highly scalable
- GPU support
- Complex queries
- Kubernetes-native

**Chroma:**
- Simple, lightweight
- Good for prototyping
- Python-native
- Embedded or client-server

**LanceDB:**
- Embedded (no server)
- Columnar storage
- Good for local/edge
- Python/JS SDKs

### Database Extensions

**pgvector (PostgreSQL):**
- Add vectors to existing Postgres
- SQL interface
- ACID transactions
- Good for hybrid workloads

**Elasticsearch:**
- Add vectors to existing ES
- Combine with text search
- Familiar tooling

### Selection Guide

| Need | Recommendation |
|------|----------------|
| Quick start, managed | Pinecone Serverless |
| Feature-rich, managed | Weaviate Cloud |
| Self-hosted, performance | Qdrant |
| Self-hosted, scale | Milvus |
| Prototyping | Chroma |
| Embedded/edge | LanceDB |
| Existing Postgres | pgvector |
| Existing Elasticsearch | ES vector search |

### Capacity Planning

**Memory estimation:**
```
Memory = num_vectors × dimensions × bytes_per_float × overhead_factor

Example (1M vectors, 1536 dims, float32, HNSW):
Memory = 1,000,000 × 1536 × 4 × 1.5 ≈ 9.2 GB
```

**Storage estimation:**
```
Storage = num_vectors × (dimensions × bytes + metadata_size)
```

**QPS estimation:**
Depends on:
- Hardware (CPU/GPU)
- Index type
- Recall requirements
- Query complexity

Typical ranges:
- HNSW on CPU: 100-1000 QPS
- HNSW on GPU: 1000-10000 QPS
- IVF-PQ: 1000-5000 QPS

### Embedding Cost Estimation

**Storage cost calculator:**
```python
def estimate_embedding_storage(
    num_documents: int,
    avg_chunks_per_doc: int,
    dimensions: int,
    precision: str = "float32"
) -> dict:
    """
    Estimate storage requirements for embeddings.
    """
    bytes_per_float = {"float32": 4, "float16": 2, "int8": 1, "binary": 0.125}
    
    total_chunks = num_documents * avg_chunks_per_doc
    bytes_per_vector = dimensions * bytes_per_float[precision]
    
    # Raw embedding storage
    raw_storage_gb = (total_chunks * bytes_per_vector) / (1024**3)
    
    # Add overhead for index (HNSW ~1.5x, IVF ~1.1x)
    hnsw_storage_gb = raw_storage_gb * 1.5
    
    # Metadata estimate (100 bytes per chunk typical)
    metadata_gb = (total_chunks * 100) / (1024**3)
    
    return {
        "total_chunks": total_chunks,
        "raw_storage_gb": raw_storage_gb,
        "with_hnsw_index_gb": hnsw_storage_gb,
        "metadata_gb": metadata_gb,
        "total_estimated_gb": hnsw_storage_gb + metadata_gb
    }

# Example: 100K documents, 10 chunks each, 1536 dims
estimate_embedding_storage(100_000, 10, 1536)
# {'total_chunks': 1000000, 'raw_storage_gb': 5.72, 
#  'with_hnsw_index_gb': 8.58, 'total_estimated_gb': 8.67}
```

**API cost estimation:**
```python
def estimate_embedding_api_cost(
    num_chunks: int,
    avg_tokens_per_chunk: int,
    model: str = "text-embedding-3-small"
) -> dict:
    """
    Estimate API costs for embedding generation.
    """
    # Prices per 1M tokens (as of 2024)
    prices = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "text-embedding-ada-002": 0.10,
        "cohere-embed-v3": 0.10,
        "voyage-large-2": 0.12
    }
    
    total_tokens = num_chunks * avg_tokens_per_chunk
    cost = (total_tokens / 1_000_000) * prices.get(model, 0.10)
    
    return {
        "total_tokens": total_tokens,
        "model": model,
        "estimated_cost_usd": round(cost, 2)
    }

# Example: 1M chunks, 256 tokens each
estimate_embedding_api_cost(1_000_000, 256, "text-embedding-3-small")
# {'total_tokens': 256000000, 'estimated_cost_usd': 5.12}
```

### Embedding Compression Techniques

**Quantization options:**

| Format | Bytes/dim | Compression | Quality Loss |
|--------|-----------|-------------|--------------|
| float32 | 4 | 1x (baseline) | None |
| float16 | 2 | 2x | Negligible |
| float8 | 1 | 4x | <0.3% |
| int8 | 1 | 4x | 1-2% |
| binary | 0.125 | 32x | 5-10% |

**Matryoshka Representation Learning (MRL):**

Train embeddings that work at multiple dimensions - like Russian nesting dolls:

```python
from sentence_transformers import SentenceTransformer, losses

# Models supporting Matryoshka embeddings
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")

# Full dimension embedding (768)
full_embedding = model.encode("Hello world")

# Truncate to smaller dimensions - still works!
embedding_512 = full_embedding[:512]  # 33% smaller
embedding_256 = full_embedding[:256]  # 66% smaller
embedding_128 = full_embedding[:128]  # 83% smaller
```

**Benefits of Matryoshka embeddings:**
- Single model, multiple dimension options
- Trade quality for storage/speed at query time
- No retraining needed for different dimensions
- Supported by: OpenAI text-embedding-3-*, nomic-embed, jina-embeddings-v3

**Dimension reduction comparison:**

| Method | Compression | Quality | Flexibility |
|--------|-------------|---------|-------------|
| **Matryoshka** | 2-8x | High | Query-time choice |
| **PCA** | 2-4x | Medium | Fixed after training |
| **Quantization** | 4-32x | High-Medium | Fixed format |
| **PCA + Quantization** | 8-16x | Medium | Combined benefits |

**Recommendation:** Use Matryoshka-enabled models (OpenAI, Nomic, Jina) and combine with float8 quantization for optimal storage/quality tradeoff.

## Embedding Best Practices

### Query vs Document Embeddings

Some models use different prefixes:

```python
# E5 model
query_embedding = model.encode("query: What is machine learning?")
doc_embedding = model.encode("passage: Machine learning is a subset of AI...")

# BGE model
query_embedding = model.encode("Represent this sentence for searching: What is ML?")
doc_embedding = model.encode("Machine learning is...")
```

### Handling Long Documents

**Chunking:**
```python
def embed_long_document(text, model, chunk_size=512, overlap=50):
    chunks = chunk_text(text, chunk_size, overlap)
    embeddings = model.encode(chunks)
    return chunks, embeddings
```

**Late chunking (experimental):**
1. Embed full document with long-context model
2. Extract chunk embeddings from full embedding
3. Preserves cross-chunk context

### Embedding Normalization

```python
def normalize(embedding):
    return embedding / np.linalg.norm(embedding)

# For cosine similarity, normalize once at index time
normalized_embedding = normalize(embedding)
```

### Batching for Efficiency

```python
def batch_embed(texts, model, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
```

### Caching Embeddings

```python
import hashlib
import redis

def get_embedding(text, model, cache):
    # Create cache key
    key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
    
    # Check cache
    cached = cache.get(key)
    if cached:
        return np.frombuffer(cached, dtype=np.float32)
    
    # Compute and cache
    embedding = model.encode(text)
    cache.set(key, embedding.tobytes(), ex=86400)  # 24h TTL
    return embedding
```

## Fine-Tuning Embeddings

### When to Fine-Tune

- Domain-specific vocabulary (legal, medical, technical)
- Specific similarity notion (not general semantic similarity)
- Significant quality gap with off-the-shelf models

### Training Data

**Positive pairs:**
```json
{"query": "What is the return policy?", "positive": "Our return policy allows..."}
{"query": "How do I reset password?", "positive": "To reset your password..."}
```

**Hard negatives:**
```json
{
  "query": "What is the return policy?",
  "positive": "Our return policy allows...",
  "negative": "Our shipping policy states..."  // Related but wrong
}
```

### Training Approaches

**Contrastive learning:**
```python
from sentence_transformers import SentenceTransformer, losses

model = SentenceTransformer('BAAI/bge-base-en-v1.5')
train_loss = losses.MultipleNegativesRankingLoss(model)

# Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100
)
```

**Matryoshka embeddings:**
Train embeddings that work at multiple dimensions:
```python
from sentence_transformers import losses

loss = losses.MatryoshkaLoss(
    model,
    losses.MultipleNegativesRankingLoss(model),
    matryoshka_dims=[768, 512, 256, 128, 64]
)
```

### Evaluation

**Metrics:**
- Recall@K on held-out queries
- MRR (Mean Reciprocal Rank)
- NDCG

```python
def evaluate_embeddings(model, eval_data):
    queries = [d['query'] for d in eval_data]
    positives = [d['positive'] for d in eval_data]
    
    query_embs = model.encode(queries)
    doc_embs = model.encode(positives)
    
    # Compute recall@k
    similarities = cosine_similarity(query_embs, doc_embs)
    recall_at_10 = compute_recall(similarities, k=10)
    
    return recall_at_10
```

## Multimodal Embeddings

### CLIP-style Models

Embed images and text in same space:

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Embed image
image_inputs = processor(images=image, return_tensors="pt")
image_embedding = model.get_image_features(**image_inputs)

# Embed text
text_inputs = processor(text="a photo of a cat", return_tensors="pt")
text_embedding = model.get_text_features(**text_inputs)

# Compare
similarity = cosine_similarity(image_embedding, text_embedding)
```

### Use Cases

- Image search with text queries
- Text search with image queries
- Multimodal RAG
- Content moderation

## References

### Papers
- Mikolov et al. (2013). "Efficient Estimation of Word Representations" (Word2Vec)
- Pennington et al. (2014). "GloVe: Global Vectors for Word Representation"
- Reimers & Gurevych (2019). "Sentence-BERT"
- Malkov & Yashunin (2018). "Efficient and Robust Approximate Nearest Neighbor"
- Jégou et al. (2011). "Product Quantization for Nearest Neighbor Search"

### Benchmarks
- MTEB (Massive Text Embedding Benchmark)
- BEIR (Benchmarking IR)
- ANN Benchmarks (ann-benchmarks.com)

### Resources
- Sentence Transformers documentation
- Pinecone learning center
- Weaviate documentation
