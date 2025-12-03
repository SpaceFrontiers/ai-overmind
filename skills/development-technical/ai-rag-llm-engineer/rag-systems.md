# RAG Systems Design and Optimization

This document provides detailed guidance on building production-ready Retrieval-Augmented Generation systems.

## Why RAG?

### LLM Limitations RAG Addresses

| Limitation | Description | How RAG Helps |
|------------|-------------|---------------|
| **Knowledge Cutoff** | Training data has fixed date | Retrieve current information |
| **Hallucination** | Generates plausible falsehoods | Ground in retrieved facts |
| **No Attribution** | Can't cite sources | Return source documents |
| **Static Knowledge** | Can't access new data | Query live databases |
| **Domain Gaps** | May lack specialized knowledge | Retrieve domain documents |
| **Privacy** | Can't access private data | Query internal systems |

### RAG vs Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Knowledge updates** | Instant (update index) | Requires retraining |
| **Cost** | Lower (no training) | Higher (GPU time) |
| **Transparency** | High (show sources) | Low (black box) |
| **Hallucination** | Reduced (grounded) | Can still hallucinate |
| **Latency** | Higher (retrieval step) | Lower (direct generation) |
| **Best for** | Facts, current info | Style, format, behavior |

**Recommendation:** Start with RAG. Add fine-tuning only if RAG doesn't meet quality needs.

## RAG Architecture Patterns

### Basic RAG

```
Query → Embed → Vector Search → Top-K Docs → Augment Prompt → LLM → Response
```

Simple and effective for many use cases.

### Advanced RAG Patterns

**1. Hybrid Search (Dense + Sparse)**
```
Query → [Dense Embedding] → Vector Search ─┐
      → [Sparse (BM25)]   → Keyword Search ─┼→ Fusion → Re-rank → LLM
```

Combines semantic understanding with keyword precision.

**2. Query Transformation**
```
Original Query → Query Rewriter → Multiple Queries → Retrieve Each → Merge Results
```

Techniques:
- **Query expansion**: Add synonyms, related terms
- **Query decomposition**: Break complex query into sub-queries
- **HyDE**: Generate hypothetical answer, use as query

**3. Iterative Retrieval**
```
Query → Retrieve → Generate Partial → Retrieve More → Generate Final
```

For complex questions requiring multiple retrieval steps.

**4. Self-RAG**
```
Query → Retrieve → LLM Evaluates Relevance → Filter → Generate → LLM Evaluates Output
```

LLM critiques its own retrieval and generation.

### Multi-Index Architectures

**Separate Indices by Content Type:**
```
┌─────────────────────────────────────────────────┐
│                   Query Router                   │
└─────────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ FAQ Index│  │ Docs Index│  │ API Index│
   └──────────┘  └──────────┘  └──────────┘
```

**Benefits:**
- Optimized chunking per content type
- Different embedding models if needed
- Targeted retrieval

### GraphRAG (Knowledge Graph Enhanced RAG)

GraphRAG incorporates knowledge graphs into the RAG pipeline for improved multi-hop reasoning.

**Architecture:**
```
Documents → Entity Extraction → Relationship Extraction → Knowledge Graph
                                                              ↓
Query → Entity Recognition → Graph Traversal → Relevant Subgraph → LLM
```

**Key Components:**
1. **Entity extraction**: Identify entities (people, places, concepts) from documents
2. **Relationship extraction**: Discover connections between entities
3. **Graph construction**: Build knowledge graph with entities as nodes, relationships as edges
4. **Graph-based retrieval**: Use graph traversal (e.g., Personalized PageRank) for retrieval

**When to use GraphRAG:**
- Multi-hop questions requiring reasoning across multiple documents
- Questions about relationships between entities
- Complex reasoning tasks
- When document structure matters

**Implementation approaches:**
- **Microsoft GraphRAG**: Hierarchical community summaries for global questions
- **Document GraphRAG**: Leverage document structure as graph
- **Hybrid**: Combine vector search with graph traversal

```python
# Simplified GraphRAG retrieval
def graphrag_retrieve(query, kg, vector_index, llm):
    # Extract entities from query
    entities = llm.extract_entities(query)
    
    # Graph traversal from seed entities
    subgraph = kg.traverse(entities, max_hops=2)
    
    # Also retrieve via vector search
    vector_results = vector_index.search(query)
    
    # Combine and rank
    combined = merge_results(subgraph.to_text(), vector_results)
    return combined
```

### Corrective RAG (CRAG)

CRAG adds self-correction mechanisms to evaluate and refine retrieval quality.

**Pipeline:**
```
Query → Retrieve → Evaluate Relevance → [Correct/Ambiguous/Incorrect]
                                              ↓
                        Correct: Use retrieved docs
                        Ambiguous: Web search + retrieved docs  
                        Incorrect: Web search only
                                              ↓
                                    Generate Answer
```

**Key insight:** Not all retrieved documents are helpful. CRAG uses an evaluator to:
1. Score document relevance
2. Decide whether to use, augment, or replace retrieved content
3. Trigger web search for knowledge refinement when needed

```python
def crag_pipeline(query, retriever, evaluator, llm):
    docs = retriever.retrieve(query)
    
    # Evaluate each document
    scores = [evaluator.score(query, doc) for doc in docs]
    
    if all(s > 0.7 for s in scores):
        # High confidence - use retrieved docs
        context = docs
    elif any(s > 0.5 for s in scores):
        # Ambiguous - augment with web search
        web_results = web_search(query)
        context = docs + web_results
    else:
        # Low confidence - use web search only
        context = web_search(query)
    
    return llm.generate(query, context)
```

### Self-RAG

Self-RAG trains the LLM to self-reflect on retrieval and generation quality using special tokens.

**Reflection tokens:**
- `[Retrieve]`: Should I retrieve? (yes/no/continue)
- `[IsRel]`: Is retrieved doc relevant? (relevant/irrelevant)
- `[IsSup]`: Is response supported by context? (fully/partially/no)
- `[IsUse]`: Is response useful? (5/4/3/2/1)

**Process:**
1. Decide whether retrieval is needed
2. For each retrieved doc, generate response + reflection tokens
3. Score each path based on reflection token probabilities
4. Select highest-scoring response

### Agentic RAG

Agentic RAG embeds autonomous agents into the RAG pipeline for dynamic, adaptive retrieval.

**Key capabilities:**
- **Reflection**: Evaluate retrieval quality and self-correct
- **Planning**: Decompose complex queries into retrieval steps
- **Tool use**: Access multiple retrieval sources dynamically
- **Multi-agent collaboration**: Specialized agents for different tasks

**Agentic RAG patterns:**

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Router Agent** | Routes queries to appropriate retrieval sources | Multi-source systems |
| **Iterative Agent** | Retrieves, evaluates, retrieves more if needed | Complex questions |
| **Planning Agent** | Creates retrieval plan, executes steps | Multi-hop reasoning |
| **Critic Agent** | Evaluates and refines retrieved context | High-stakes applications |

```python
class AgenticRAG:
    def __init__(self, retrievers, llm):
        self.retrievers = retrievers  # Multiple retrieval sources
        self.llm = llm
    
    def answer(self, query, max_iterations=3):
        context = []
        
        for i in range(max_iterations):
            # Plan: What information do I need?
            plan = self.llm.plan(query, context)
            
            if plan.has_enough_info:
                break
            
            # Act: Retrieve from appropriate source
            source = self.llm.select_source(plan.info_need, self.retrievers)
            new_docs = source.retrieve(plan.sub_query)
            
            # Reflect: Is this useful?
            useful_docs = self.llm.filter_relevant(query, new_docs)
            context.extend(useful_docs)
        
        return self.llm.generate(query, context)
```

## Document Processing Pipeline

### 1. Document Ingestion

**Supported formats:**
- PDF (with OCR for scanned)
- Word documents
- HTML/Markdown
- Plain text
- Structured data (CSV, JSON)

**Extraction considerations:**
- Preserve structure (headers, lists, tables)
- Extract metadata (title, author, date)
- Handle images (OCR, captions)
- Maintain formatting context

### 2. Text Cleaning

```python
def clean_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters (carefully)
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', text)
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    return text.strip()
```

**Cleaning decisions:**
- Keep or remove headers/footers
- Handle code blocks specially
- Preserve or flatten structure
- Language detection and handling

### 3. Chunking Strategies

#### Fixed-Size Chunking
```python
def fixed_chunk(text, chunk_size=512, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
```

**Pros:** Simple, predictable
**Cons:** May split mid-sentence, loses context

#### Sentence-Based Chunking
```python
def sentence_chunk(text, max_sentences=5):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunks.append(' '.join(sentences[i:i + max_sentences]))
    return chunks
```

**Pros:** Preserves sentence boundaries
**Cons:** Variable chunk sizes

#### Semantic Chunking
```python
def semantic_chunk(text, similarity_threshold=0.7):
    sentences = sent_tokenize(text)
    embeddings = embed(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        if similarity > similarity_threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
    
    chunks.append(' '.join(current_chunk))
    return chunks
```

**Pros:** Preserves semantic coherence
**Cons:** Slower, requires embedding model

#### Recursive Chunking
```python
def recursive_chunk(text, chunk_size=512, separators=['\n\n', '\n', '. ', ' ']):
    if len(text) <= chunk_size:
        return [text]
    
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ''
            for part in parts:
                if len(current) + len(part) <= chunk_size:
                    current += sep + part
                else:
                    if current:
                        chunks.append(current)
                    current = part
            if current:
                chunks.append(current)
            return chunks
    
    # Fallback to fixed-size
    return fixed_chunk(text, chunk_size)
```

**Pros:** Respects document structure
**Cons:** More complex logic

#### Meta-Chunking (Advanced)

Based on recent research, chunks should preserve **logical coherence**:

```python
def meta_chunk(text, model):
    """
    Use LLM to identify logical boundaries based on:
    - Causal relationships
    - Transitional connections  
    - Progressive arguments
    """
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Ask LLM if sentences should be in same chunk
        prompt = f"""Should these sentences be in the same chunk?
        Previous: {current_chunk[-1]}
        Next: {sentences[i]}
        Answer: yes/no"""
        
        if model.predict(prompt) == 'yes':
            current_chunk.append(sentences[i])
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
    
    return chunks
```

#### Late Chunking (State-of-the-Art)

Traditional chunking embeds each chunk independently, losing cross-chunk context. **Late chunking** solves this by:

1. Embedding the **entire document** through a long-context model
2. Applying chunking **after** the transformer, before mean pooling
3. Each chunk embedding captures full document context

```python
def late_chunking(text, model, chunk_boundaries):
    """
    Late chunking preserves cross-chunk context.
    Requires long-context embedding model (e.g., jina-embeddings-v3).
    """
    # 1. Get token embeddings for entire document
    token_embeddings = model.encode_tokens(text)  # Shape: (seq_len, dim)
    
    # 2. Apply chunking to token embeddings
    chunk_embeddings = []
    for start, end in chunk_boundaries:
        # Mean pool tokens within chunk boundary
        chunk_emb = token_embeddings[start:end].mean(dim=0)
        chunk_embeddings.append(chunk_emb)
    
    return chunk_embeddings
```

**Benefits:**
- Chunks retain awareness of surrounding context
- Pronouns and references resolved correctly
- Better retrieval for context-dependent passages

**Trade-offs:**
- Requires long-context embedding model
- Higher computational cost at indexing time
- Document must fit in model's context window

#### Contextual Retrieval (Anthropic)

Add document-level context to each chunk before embedding:

```python
def contextual_chunking(document, chunks, llm):
    """
    Prepend contextual summary to each chunk.
    Improves retrieval by 49% (Anthropic research).
    """
    contextualized_chunks = []
    
    for chunk in chunks:
        # Generate context for this chunk
        prompt = f"""<document>
{document}
</document>

Here is the chunk we want to situate:
<chunk>
{chunk}
</chunk>

Give a short succinct context to situate this chunk within 
the overall document for retrieval purposes."""
        
        context = llm.generate(prompt)
        
        # Prepend context to chunk
        contextualized = f"{context}\n\n{chunk}"
        contextualized_chunks.append(contextualized)
    
    return contextualized_chunks
```

**Comparison:**

| Method | Context Preservation | Compute Cost | Quality |
|--------|---------------------|--------------|---------|
| **Fixed chunking** | None | Low | Baseline |
| **Semantic chunking** | Within chunk | Medium | Good |
| **Late chunking** | Full document | High (index) | Better |
| **Contextual retrieval** | Explicit summary | High (LLM calls) | Best |

**Recommendation:** 
- Start with semantic chunking
- Use late chunking if you have long-context embedding model
- Use contextual retrieval for high-value, static document collections

### Chunking Best Practices

| Parameter | Recommendation | Rationale |
|-----------|----------------|-----------|
| **Chunk size** | 256-512 tokens | Balance context vs specificity |
| **Overlap** | 10-20% | Prevent information loss at boundaries |
| **Min chunk** | 50 tokens | Avoid too-small chunks |
| **Max chunk** | 1000 tokens | Stay within embedding model limits |

**Content-specific guidance:**
- **Technical docs**: Larger chunks (512-1000), preserve code blocks
- **FAQs**: One Q&A per chunk
- **Legal docs**: Paragraph-based, preserve section references
- **Chat logs**: Conversation-turn based

### 4. Metadata Enrichment

Add metadata to improve retrieval:

```python
chunk = {
    "text": "...",
    "metadata": {
        "source": "user_manual_v2.pdf",
        "page": 42,
        "section": "Troubleshooting",
        "doc_type": "manual",
        "created_at": "2024-01-15",
        "language": "en"
    }
}
```

**Useful metadata:**
- Source document
- Page/section numbers
- Document type
- Creation/update date
- Author
- Tags/categories
- Language

### 5. Embedding Generation

```python
def embed_chunks(chunks, model):
    embeddings = []
    for chunk in chunks:
        # Optionally add instruction prefix
        text = f"passage: {chunk['text']}"
        embedding = model.encode(text)
        embeddings.append({
            **chunk,
            "embedding": embedding
        })
    return embeddings
```

**Embedding model selection:**

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| text-embedding-3-small | 1536 | Fast | Good |
| text-embedding-3-large | 3072 | Medium | Better |
| BGE-large | 1024 | Medium | Good |
| E5-large-v2 | 1024 | Medium | Good |
| Cohere embed-v3 | 1024 | Fast | Good |

### 6. Indexing

```python
# Example with Pinecone
import pinecone

index = pinecone.Index("my-index")

# Upsert in batches
batch_size = 100
for i in range(0, len(embeddings), batch_size):
    batch = embeddings[i:i + batch_size]
    vectors = [
        {
            "id": f"chunk_{j}",
            "values": item["embedding"],
            "metadata": item["metadata"]
        }
        for j, item in enumerate(batch, start=i)
    ]
    index.upsert(vectors=vectors)
```

## Retrieval Strategies

### Basic Vector Search

```python
def retrieve(query, index, model, top_k=5):
    query_embedding = model.encode(f"query: {query}")
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results
```

### Hybrid Search

Combine dense (vector) and sparse (BM25) retrieval:

```python
def hybrid_search(query, vector_index, bm25_index, alpha=0.5, top_k=5):
    # Dense retrieval
    dense_results = vector_search(query, vector_index, top_k=top_k*2)
    
    # Sparse retrieval
    sparse_results = bm25_search(query, bm25_index, top_k=top_k*2)
    
    # Reciprocal Rank Fusion
    scores = {}
    for rank, doc in enumerate(dense_results):
        scores[doc.id] = scores.get(doc.id, 0) + alpha / (rank + 60)
    for rank, doc in enumerate(sparse_results):
        scores[doc.id] = scores.get(doc.id, 0) + (1-alpha) / (rank + 60)
    
    # Sort by combined score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

### Query Transformation

**Query Expansion:**
```python
def expand_query(query, llm):
    prompt = f"""Generate 3 alternative phrasings of this query:
    Query: {query}
    Alternatives:"""
    alternatives = llm.generate(prompt)
    return [query] + alternatives
```

**HyDE (Hypothetical Document Embeddings):**
```python
def hyde_search(query, llm, index, model):
    # Generate hypothetical answer
    prompt = f"Write a passage that answers: {query}"
    hypothetical = llm.generate(prompt)
    
    # Use hypothetical as query
    embedding = model.encode(hypothetical)
    return index.query(vector=embedding)
```

**Multi-Query Retrieval:**
```python
def multi_query_retrieve(query, llm, index, model):
    # Generate sub-queries
    prompt = f"""Break this question into 3 simpler questions:
    Question: {query}"""
    sub_queries = llm.generate(prompt)
    
    # Retrieve for each
    all_results = []
    for sq in sub_queries:
        results = retrieve(sq, index, model)
        all_results.extend(results)
    
    # Deduplicate and rank
    return dedupe_and_rank(all_results)
```

### Re-Ranking

After initial retrieval, re-rank for better precision:

**Cross-Encoder Re-ranking:**
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, documents, top_k=5):
    pairs = [[query, doc.text] for doc in documents]
    scores = reranker.predict(pairs)
    
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]
```

**Cohere Rerank:**
```python
import cohere

co = cohere.Client(api_key)

def rerank(query, documents, top_k=5):
    response = co.rerank(
        query=query,
        documents=[doc.text for doc in documents],
        top_n=top_k,
        model='rerank-english-v2.0'
    )
    return [documents[r.index] for r in response.results]
```

**LLM-based Re-ranking:**
```python
def llm_rerank(query, documents, llm):
    prompt = f"""Rank these documents by relevance to the query.
    Query: {query}
    
    Documents:
    {format_documents(documents)}
    
    Return document numbers in order of relevance:"""
    
    ranking = llm.generate(prompt)
    return reorder_by_ranking(documents, ranking)
```

### Filtering

Use metadata to narrow search:

```python
def filtered_search(query, index, model, filters):
    embedding = model.encode(query)
    results = index.query(
        vector=embedding,
        filter={
            "doc_type": {"$eq": "manual"},
            "language": {"$eq": "en"},
            "date": {"$gte": "2024-01-01"}
        },
        top_k=10
    )
    return results
```

**Self-Query (LLM extracts filters):**
```python
def self_query(query, llm, index, model):
    # LLM extracts structured filters
    prompt = f"""Extract search filters from this query:
    Query: {query}
    
    Available filters: doc_type, language, date, author
    Return as JSON:"""
    
    filters = json.loads(llm.generate(prompt))
    clean_query = extract_semantic_query(query)
    
    return filtered_search(clean_query, index, model, filters)
```

## Prompt Augmentation

### Basic Augmentation

```python
def augment_prompt(query, documents):
    context = "\n\n".join([
        f"[{i+1}] {doc.text}"
        for i, doc in enumerate(documents)
    ])
    
    return f"""Answer the question based on the context below.
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {query}

Answer:"""
```

### Advanced Prompt Patterns

**With Source Attribution:**
```python
prompt = f"""Answer the question using the provided sources.
Cite sources using [1], [2], etc.

Sources:
{format_sources(documents)}

Question: {query}

Answer (with citations):"""
```

**With Confidence:**
```python
prompt = f"""Answer the question based on the context.
Rate your confidence (high/medium/low) based on how well the context supports your answer.

Context:
{context}

Question: {query}

Answer:
Confidence:
Reasoning:"""
```

**Chain-of-Thought RAG:**
```python
prompt = f"""Use the context to answer the question step by step.

Context:
{context}

Question: {query}

Let's think through this:
1. First, identify relevant information in the context...
2. Then, synthesize the answer...
3. Finally, verify against the context...

Answer:"""
```

### Context Window Management

When retrieved content exceeds context window:

**Truncation strategies:**
1. **Simple truncation**: Take first N tokens
2. **Relevance-based**: Keep highest-scoring chunks
3. **Summarization**: Summarize each chunk
4. **Map-reduce**: Process chunks separately, combine

```python
def fit_context(documents, max_tokens=4000):
    total_tokens = 0
    selected = []
    
    for doc in documents:  # Assume sorted by relevance
        doc_tokens = count_tokens(doc.text)
        if total_tokens + doc_tokens <= max_tokens:
            selected.append(doc)
            total_tokens += doc_tokens
        else:
            # Truncate last document to fit
            remaining = max_tokens - total_tokens
            if remaining > 100:  # Minimum useful size
                truncated = truncate_to_tokens(doc.text, remaining)
                selected.append(truncated)
            break
    
    return selected
```

## Production Considerations

### Caching

**Query cache:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_retrieve(query_hash):
    return retrieve(query)
```

**Embedding cache:**
```python
import redis

def get_or_compute_embedding(text, model, cache):
    key = f"emb:{hash(text)}"
    cached = cache.get(key)
    if cached:
        return np.frombuffer(cached, dtype=np.float32)
    
    embedding = model.encode(text)
    cache.set(key, embedding.tobytes(), ex=86400)  # 24h TTL
    return embedding
```

### Monitoring

**Key metrics:**
- Retrieval latency (p50, p95, p99)
- Generation latency
- Retrieval recall (if ground truth available)
- User feedback (thumbs up/down)
- Cache hit rate
- Token usage and cost

```python
import time
from prometheus_client import Histogram, Counter

retrieval_latency = Histogram('rag_retrieval_seconds', 'Retrieval latency')
generation_latency = Histogram('rag_generation_seconds', 'Generation latency')
feedback_counter = Counter('rag_feedback', 'User feedback', ['type'])

def rag_pipeline(query):
    with retrieval_latency.time():
        docs = retrieve(query)
    
    with generation_latency.time():
        response = generate(query, docs)
    
    return response
```

### Error Handling

```python
def robust_rag(query):
    try:
        docs = retrieve(query)
        if not docs:
            return fallback_response(query)
        
        response = generate(query, docs)
        return response
        
    except RetrievalError:
        logger.error("Retrieval failed")
        return generate_without_context(query)
        
    except GenerationError:
        logger.error("Generation failed")
        return "I'm having trouble generating a response. Please try again."
```

### Index Updates

**Incremental updates:**
```python
def update_index(new_documents, index):
    # Process new documents
    chunks = chunk_documents(new_documents)
    embeddings = embed_chunks(chunks)
    
    # Upsert to index
    index.upsert(embeddings)
    
    # Optionally remove outdated
    outdated_ids = find_outdated(new_documents)
    index.delete(ids=outdated_ids)
```

**Full reindex (for major changes):**
```python
def full_reindex(documents, index_name):
    # Create new index
    new_index = create_index(f"{index_name}_new")
    
    # Populate
    populate_index(documents, new_index)
    
    # Atomic swap
    swap_index_alias(index_name, new_index)
    
    # Delete old
    delete_index(f"{index_name}_old")
```

## Evaluation

### Retrieval Evaluation

**Recall@K:**
```python
def recall_at_k(retrieved, relevant, k):
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)
```

**MRR (Mean Reciprocal Rank):**
```python
def mrr(retrieved, relevant):
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1 / (i + 1)
    return 0
```

**NDCG:**
```python
def ndcg(retrieved, relevance_scores, k):
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))
    ideal = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0
```

### End-to-End Evaluation

**Faithfulness (is answer grounded in context?):**
```python
def evaluate_faithfulness(answer, context, llm):
    prompt = f"""Is this answer fully supported by the context?
    
    Context: {context}
    Answer: {answer}
    
    Respond: fully_supported / partially_supported / not_supported"""
    
    return llm.generate(prompt)
```

**Relevance (does answer address the question?):**
```python
def evaluate_relevance(question, answer, llm):
    prompt = f"""Does this answer address the question?
    
    Question: {question}
    Answer: {answer}
    
    Score from 1-5:"""
    
    return int(llm.generate(prompt))
```

### Creating Evaluation Datasets

```python
# Golden dataset structure
evaluation_set = [
    {
        "query": "What is the return policy?",
        "relevant_docs": ["doc_123", "doc_456"],
        "expected_answer": "30-day return policy for unused items",
        "metadata": {"category": "policy", "difficulty": "easy"}
    },
    # ... more examples
]
```

## Common Issues and Solutions

### Low Retrieval Quality

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Missing relevant docs | Chunk too large | Smaller chunks |
| Too many irrelevant docs | Chunk too small | Larger chunks, re-ranking |
| Semantic mismatch | Wrong embedding model | Domain-specific model |
| Keyword queries fail | Pure vector search | Add hybrid search |

### Hallucination Despite RAG

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Makes up facts | Context not used | Stronger grounding prompt |
| Contradicts context | Context too long | Reduce context, highlight key parts |
| Mixes sources | Unclear attribution | Require citations |
| Confident when wrong | No uncertainty | Add confidence scoring |

### Performance Issues

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Slow retrieval | Large index | Add filtering, use ANN |
| Slow embedding | Large model | Smaller model, batching |
| High latency | Sequential steps | Parallelize retrieval |
| Memory issues | Large context | Streaming, chunked processing |

## References

### Papers
- Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Gao et al. (2023). "Retrieval-Augmented Generation for Large Language Models: A Survey"
- Zhao et al. (2024). "Meta-Chunking: Learning Text Segmentation and Semantic Completion"
- Giuffrè et al. (2025). "Expert-Validated Retrieval-Augmented and Fine-Tuned GPT-4"

### Resources
- LangChain RAG documentation
- LlamaIndex guides
- Pinecone learning center
- Weaviate recipes
