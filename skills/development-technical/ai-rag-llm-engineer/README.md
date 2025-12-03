# AI/RAG/LLM Systems Engineer Skill

A comprehensive skill for designing, building, and optimizing AI systems powered by Large Language Models, Retrieval-Augmented Generation, and semantic search.

## Overview

This skill provides guidance for:
- Understanding transformer architecture and LLM internals
- Building RAG pipelines for knowledge-grounded applications
- Working with embeddings and vector databases
- Fine-tuning and aligning LLMs
- Designing LLM agents with tool use
- Evaluating AI system quality

## Files

| File | Description |
|------|-------------|
| [SKILL.md](SKILL.md) | Main skill document with core concepts and decision frameworks |
| [transformers.md](transformers.md) | Deep dive into transformer architecture |
| [rag-systems.md](rag-systems.md) | RAG pipeline design and optimization |
| [embeddings-vectors.md](embeddings-vectors.md) | Embeddings and vector search |
| [fine-tuning.md](fine-tuning.md) | Fine-tuning and alignment techniques |
| [agents.md](agents.md) | LLM agents and tool use |
| [evaluation.md](evaluation.md) | Evaluation metrics and benchmarks |

## When to Use This Skill

Use this skill when:
- Building search systems with semantic understanding
- Creating chatbots or conversational AI
- Developing knowledge bases with natural language interfaces
- Implementing document Q&A systems
- Building AI agents that can use tools
- Fine-tuning models for specific domains
- Evaluating LLM-powered applications

## Key Topics Covered

### Transformers & LLMs
- Self-attention mechanism
- Encoder/decoder architectures
- BERT, GPT, T5 variants
- Positional encoding
- Efficient attention (Flash Attention)

### RAG Systems
- Document processing pipelines
- Chunking strategies (fixed, semantic, meta-chunking)
- Retrieval strategies (hybrid, multi-query, HyDE)
- Re-ranking techniques
- Prompt augmentation patterns

### Embeddings & Vector Search
- Embedding model selection
- Similarity metrics (cosine, dot product, euclidean)
- ANN algorithms (HNSW, IVF, PQ)
- Vector database selection

### Fine-Tuning
- Full fine-tuning vs PEFT (LoRA, QLoRA)
- Supervised fine-tuning (SFT)
- RLHF and DPO alignment
- Data preparation best practices

### Agents
- ReAct pattern
- Chain-of-thought reasoning
- Tool use / function calling
- Memory systems
- Multi-agent architectures

### Evaluation
- LLM metrics (perplexity, BLEU, ROUGE)
- RAG metrics (faithfulness, relevance)
- Hallucination detection
- LLM-as-judge approaches
- Benchmarks (MMLU, HotpotQA, etc.)

## References

### Primary Sources
- **AI-Powered Search** by Trey Grainger, Doug Turnbull, Max Irwin (Manning, 2023)
- **Super Study Guide: Transformers and LLMs** by Afshine & Shervine Amidi (2024)
- **Hands-On Large Language Models** by Jay Alammar, Maarten Grootendorst (O'Reilly, 2024)

### Key Papers
- Vaswani et al. (2017). "Attention Is All You Need"
- Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
- Yao et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models"

## Usage

This skill is designed to be used as a reference when building AI/LLM systems. Start with the main [SKILL.md](SKILL.md) for an overview, then dive into specific supporting documents based on your needs.

For quick decisions:
- **Which embedding model?** → See [embeddings-vectors.md](embeddings-vectors.md)
- **How to chunk documents?** → See [rag-systems.md](rag-systems.md)
- **Should I fine-tune?** → See [fine-tuning.md](fine-tuning.md)
- **How to evaluate?** → See [evaluation.md](evaluation.md)
