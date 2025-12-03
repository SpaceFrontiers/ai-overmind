# Transformer Architecture Deep Dive

This document provides implementation-level details about the Transformer architecture that powers modern LLMs.

## Historical Context

### Evolution of Sequence Models

**RNNs (Recurrent Neural Networks):**
- Process sequences token-by-token
- Hidden state carries information forward
- Limitation: Vanishing gradients, slow training

**LSTMs (Long Short-Term Memory):**
- Gates control information flow
- Better at long-range dependencies
- Still sequential, limited parallelization

**Attention Mechanism (Bahdanau, 2014):**
- Allow decoder to "look back" at encoder states
- Weighted sum based on relevance
- Breakthrough for machine translation

**Transformer (Vaswani, 2017):**
- "Attention is All You Need"
- Removed recurrence entirely
- Fully parallelizable
- Foundation for all modern LLMs

## Transformer Architecture

### High-Level Structure

```
Input Tokens
     ↓
[Input Embedding + Positional Encoding]
     ↓
┌─────────────────────────────────────┐
│         ENCODER (N layers)          │
│  ┌─────────────────────────────┐   │
│  │   Multi-Head Self-Attention  │   │
│  │   Add & Norm                 │   │
│  │   Feed-Forward Network       │   │
│  │   Add & Norm                 │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│         DECODER (N layers)          │
│  ┌─────────────────────────────┐   │
│  │   Masked Self-Attention      │   │
│  │   Add & Norm                 │   │
│  │   Cross-Attention            │   │
│  │   Add & Norm                 │   │
│  │   Feed-Forward Network       │   │
│  │   Add & Norm                 │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
     ↓
[Linear + Softmax]
     ↓
Output Probabilities
```

### Input Processing

**Tokenization:**
Converts text to token IDs using vocabulary:
```
"Hello world" → [15496, 995]
```

Common tokenization algorithms:
- **BPE (Byte Pair Encoding)**: GPT models
- **WordPiece**: BERT
- **SentencePiece**: T5, LLaMA
- **Unigram**: Alternative to BPE

**Token Embeddings:**
Lookup table mapping token IDs to dense vectors:
```
token_id → embedding_matrix[token_id] → vector ∈ ℝ^d_model
```

**Positional Encoding:**
Since self-attention is position-agnostic, position information must be added.

*Sinusoidal (Original Transformer):*
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Properties:
- Deterministic, no learned parameters
- Can extrapolate to longer sequences
- Relative positions have consistent representation

*Learned Positional Embeddings:*
```
position_id → position_embedding_matrix[position_id]
```

Properties:
- More flexible
- Limited to training sequence length
- Used by BERT, GPT-2

*Rotary Position Embedding (RoPE):*
- Encodes position through rotation in embedding space
- Better extrapolation to longer sequences
- Used by LLaMA, Mistral, modern models

*ALiBi (Attention with Linear Biases):*
- No explicit positional encoding
- Adds linear bias to attention scores based on distance
- Excellent length extrapolation

## Self-Attention Mechanism

### Scaled Dot-Product Attention

The core operation of transformers:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Step-by-step:**

1. **Project inputs to Q, K, V:**
   ```
   Q = X × W_Q    # Queries: what am I looking for?
   K = X × W_K    # Keys: what do I contain?
   V = X × W_V    # Values: what information do I provide?
   ```

2. **Compute attention scores:**
   ```
   scores = Q × K^T    # Shape: (seq_len, seq_len)
   ```

3. **Scale by √d_k:**
   ```
   scaled_scores = scores / √d_k
   ```
   Prevents softmax from becoming too peaked with large dimensions.

4. **Apply softmax:**
   ```
   attention_weights = softmax(scaled_scores, dim=-1)
   ```
   Each row sums to 1, representing attention distribution.

5. **Weighted sum of values:**
   ```
   output = attention_weights × V
   ```

### Multi-Head Attention

Instead of single attention, use multiple "heads" in parallel:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

where head_i = Attention(Q × W_Q^i, K × W_K^i, V × W_V^i)
```

**Benefits:**
- Different heads can focus on different aspects
- Head 1: Syntactic relationships
- Head 2: Semantic similarity
- Head 3: Positional patterns
- etc.

**Typical configurations:**
| Model | d_model | n_heads | d_head |
|-------|---------|---------|--------|
| BERT-base | 768 | 12 | 64 |
| GPT-2 | 768 | 12 | 64 |
| GPT-3 | 12288 | 96 | 128 |
| LLaMA-7B | 4096 | 32 | 128 |

### Attention Variants

**Masked Self-Attention (Causal):**
Used in decoders to prevent looking at future tokens:
```
mask = [[1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]]
        
scores = scores.masked_fill(mask == 0, -inf)
```

**Cross-Attention:**
Queries from decoder, Keys/Values from encoder:
```
Q = decoder_hidden × W_Q
K = encoder_output × W_K
V = encoder_output × W_V
```

**Multi-Query Attention (MQA):**
- Single K, V shared across all heads
- Significantly faster inference
- Used by PaLM, Falcon

**Grouped-Query Attention (GQA):**
- Groups of heads share K, V
- Balance between MHA and MQA
- Used by LLaMA 2, Mistral

## Feed-Forward Network

After attention, each position passes through a feed-forward network:

```
FFN(x) = GELU(x × W_1 + b_1) × W_2 + b_2
```

**Typical expansion:**
- Input: d_model (e.g., 768)
- Hidden: 4 × d_model (e.g., 3072)
- Output: d_model (e.g., 768)

**Activation functions:**
- **ReLU**: Original transformer
- **GELU**: BERT, GPT-2
- **SwiGLU**: LLaMA, modern models (better performance)

## Layer Normalization

Normalizes activations for stable training:

```
LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β
```

**Placement options:**
- **Post-LN**: Original transformer (after sublayer)
- **Pre-LN**: More stable training (before sublayer)
- **RMSNorm**: Simplified, no mean centering (LLaMA)

## Residual Connections

Skip connections around each sublayer:

```
output = LayerNorm(x + Sublayer(x))
```

**Benefits:**
- Enables training very deep networks
- Gradient flow through skip connections
- Each layer learns residual refinement

## Model Variants

### Encoder-Only (BERT-style)

**Architecture:**
- Stack of encoder layers
- Bidirectional attention (sees all tokens)
- [CLS] token for classification

**Pre-training objectives:**
- **MLM (Masked Language Modeling)**: Predict masked tokens
- **NSP (Next Sentence Prediction)**: Binary classification

**Use cases:**
- Text classification
- Named entity recognition
- Sentence embeddings

**Models:** BERT, RoBERTa, ALBERT, DistilBERT

### Decoder-Only (GPT-style)

**Architecture:**
- Stack of decoder layers (no cross-attention)
- Causal (masked) attention only
- Autoregressive generation

**Pre-training objective:**
- **CLM (Causal Language Modeling)**: Predict next token

**Use cases:**
- Text generation
- Instruction following
- General-purpose LLMs

**Models:** GPT-2/3/4, LLaMA, Claude, Mistral

### Encoder-Decoder (T5-style)

**Architecture:**
- Encoder processes input
- Decoder generates output with cross-attention

**Pre-training objective:**
- **Span corruption**: Predict masked spans

**Use cases:**
- Translation
- Summarization
- Question answering

**Models:** T5, BART, mT5, FLAN-T5

## Efficiency Optimizations

### Attention Complexity

Standard attention: O(n²) in sequence length

**Efficient attention variants:**

| Method | Complexity | Description |
|--------|------------|-------------|
| **Sparse Attention** | O(n√n) | Attend to fixed patterns |
| **Linear Attention** | O(n) | Kernel approximation |
| **Flash Attention** | O(n²) time, O(n) memory | IO-aware implementation |
| **Sliding Window** | O(n × w) | Local attention window |

### Flash Attention

Memory-efficient attention through:
- Tiling: Process attention in blocks
- Recomputation: Trade compute for memory
- Kernel fusion: Minimize memory transfers

**Impact:**
- 2-4x speedup
- Enables longer sequences
- Standard in modern training

### KV Cache

During autoregressive generation, cache Key and Value projections:

```
# Without cache: recompute K, V for all tokens each step
# With cache: only compute K, V for new token, append to cache
```

**Memory requirement:**
```
KV cache size = 2 × n_layers × n_heads × d_head × seq_len × batch_size × dtype_size
```

For LLaMA-7B with 2048 context:
```
2 × 32 × 32 × 128 × 2048 × 1 × 2 bytes ≈ 1 GB per sequence
```

### Quantization

Reduce precision to save memory and speed up inference:

| Precision | Bits | Memory | Speed |
|-----------|------|--------|-------|
| FP32 | 32 | 1x | 1x |
| FP16/BF16 | 16 | 0.5x | ~2x |
| INT8 | 8 | 0.25x | ~2-4x |
| INT4 | 4 | 0.125x | ~4-8x |

**Methods:**
- **Post-training quantization**: Quantize after training
- **QLoRA**: 4-bit base + LoRA adapters
- **GPTQ**: Optimal weight quantization
- **AWQ**: Activation-aware quantization

## Decoding Strategies

### Greedy Decoding
```
next_token = argmax(logits)
```
Fast but can be repetitive.

### Temperature Sampling
```
probs = softmax(logits / temperature)
next_token = sample(probs)
```
- Temperature < 1: More deterministic
- Temperature > 1: More random

### Top-K Sampling
```
top_k_logits = top_k(logits, k)
probs = softmax(top_k_logits)
next_token = sample(probs)
```
Only consider top K tokens.

### Top-P (Nucleus) Sampling
```
sorted_probs = sort(softmax(logits))
cumsum = cumulative_sum(sorted_probs)
nucleus = tokens where cumsum <= p
next_token = sample(nucleus)
```
Dynamic vocabulary based on probability mass.

### Beam Search
Maintain top-B candidates at each step:
```
for each step:
    expand all candidates
    keep top B by cumulative probability
return best complete sequence
```
Better for deterministic tasks (translation).

## Implementation Considerations

### Training Stability

**Gradient clipping:**
```
clip_grad_norm_(parameters, max_norm=1.0)
```

**Learning rate warmup:**
```
lr = base_lr × min(step / warmup_steps, 1.0)
```

**Weight initialization:**
- Embeddings: N(0, 0.02)
- Linear layers: Xavier/Glorot
- Output projection: Scaled by 1/√(2 × n_layers)

### Memory Optimization

**Gradient checkpointing:**
- Don't store all activations
- Recompute during backward pass
- Trade compute for memory

**Mixed precision training:**
- Forward/backward in FP16
- Master weights in FP32
- Loss scaling to prevent underflow

### Inference Optimization

**Batching:**
- Continuous batching for variable-length sequences
- Speculative decoding with draft model

**Model parallelism:**
- Tensor parallelism: Split layers across GPUs
- Pipeline parallelism: Split layers sequentially
- Sequence parallelism: Split sequence dimension

## References

### Foundational Papers
- Vaswani et al. (2017). "Attention Is All You Need"
- Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
- Radford et al. (2019). "Language Models are Unsupervised Multitask Learners" (GPT-2)
- Brown et al. (2020). "Language Models are Few-Shot Learners" (GPT-3)

### Architecture Improvements
- Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Press et al. (2021). "Train Short, Test Long: Attention with Linear Biases"
- Shazeer (2019). "Fast Transformer Decoding: One Write-Head is All You Need" (MQA)
- Ainslie et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models"

### Efficiency
- Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention"
- Dettmers et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers"
- Frantar et al. (2022). "GPTQ: Accurate Post-Training Quantization for GPT"
