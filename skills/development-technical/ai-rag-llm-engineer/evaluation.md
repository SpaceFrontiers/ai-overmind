# LLM Evaluation Metrics and Benchmarks

This document covers evaluation methodologies for LLMs, RAG systems, and AI applications.

## Why Evaluation Matters

Without proper evaluation:
- Can't measure improvement
- Can't compare approaches
- Can't detect regressions
- Can't identify failure modes
- Can't build trust with users

## Evaluation Dimensions

### Quality Dimensions

| Dimension | Description | Metrics |
|-----------|-------------|---------|
| **Accuracy** | Correctness of output | Exact match, F1, accuracy |
| **Relevance** | Addresses the question | Human rating, LLM judge |
| **Faithfulness** | Grounded in sources | Entailment, citation accuracy |
| **Coherence** | Logical flow | Perplexity, human rating |
| **Fluency** | Natural language quality | Grammar check, human rating |
| **Helpfulness** | Useful to user | Task completion, user feedback |
| **Safety** | Avoids harmful content | Safety classifiers, red teaming |

### Operational Dimensions

| Dimension | Description | Metrics |
|-----------|-------------|---------|
| **Latency** | Response time | p50, p95, p99 |
| **Throughput** | Requests per second | QPS |
| **Cost** | Resource consumption | $/1K tokens, $/query |
| **Reliability** | Uptime and consistency | Error rate, variance |

## LLM Evaluation Metrics

### Perplexity

Measures model confidence in predictions:

```python
import torch
import math

def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    
    perplexity = math.exp(loss.item())
    return perplexity
```

**Interpretation:**
- Lower = more confident
- Typical range: 10-100 for good models
- Not a measure of correctness

**Limitations:**
- Doesn't measure factual accuracy
- Can be low for confident but wrong outputs
- Depends on tokenization

### BLEU (Machine Translation)

Measures n-gram overlap with reference:

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

def calculate_bleu(reference, hypothesis):
    """
    reference: list of reference translations (each is list of tokens)
    hypothesis: candidate translation (list of tokens)
    """
    # Sentence-level BLEU
    score = sentence_bleu([reference], hypothesis)
    return score

# Example
reference = ["the", "cat", "sat", "on", "the", "mat"]
hypothesis = ["the", "cat", "is", "on", "the", "mat"]
bleu = calculate_bleu(reference, hypothesis)  # ~0.75
```

**BLEU-N variants:**
- BLEU-1: Unigram precision
- BLEU-2: Up to bigrams
- BLEU-4: Up to 4-grams (most common)

**Limitations:**
- Doesn't capture meaning
- Penalizes valid paraphrases
- Brevity penalty can be harsh

### ROUGE (Summarization)

Measures recall of reference n-grams:

```python
from rouge_score import rouge_scorer

def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }
```

**Variants:**
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence
- **ROUGE-Lsum**: For multi-sentence (sum of LCS)

### METEOR

More sophisticated than BLEU, includes:
- Stemming
- Synonyms
- Paraphrase matching

```python
from nltk.translate.meteor_score import meteor_score

def calculate_meteor(reference, hypothesis):
    # reference and hypothesis are strings
    return meteor_score([reference.split()], hypothesis.split())
```

### BERTScore

Semantic similarity using BERT embeddings:

```python
from bert_score import score

def calculate_bertscore(references, hypotheses):
    P, R, F1 = score(hypotheses, references, lang="en", verbose=True)
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }
```

**Advantages:**
- Captures semantic similarity
- Handles paraphrases
- Correlates well with human judgment

### Exact Match and F1 (QA)

```python
def exact_match(prediction, ground_truth):
    """Strict string match after normalization."""
    return normalize(prediction) == normalize(ground_truth)

def f1_score(prediction, ground_truth):
    """Token-level F1."""
    pred_tokens = set(normalize(prediction).split())
    truth_tokens = set(normalize(ground_truth).split())
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    common = pred_tokens & truth_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

def normalize(text):
    """Lowercase, remove punctuation, extra whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text
```

## RAG Evaluation

### Retrieval Metrics

**Recall@K:**
```python
def recall_at_k(retrieved_ids, relevant_ids, k):
    """What fraction of relevant docs are in top K?"""
    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    if not relevant_set:
        return 0.0
    
    return len(retrieved_k & relevant_set) / len(relevant_set)
```

**Precision@K:**
```python
def precision_at_k(retrieved_ids, relevant_ids, k):
    """What fraction of top K are relevant?"""
    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    if not retrieved_k:
        return 0.0
    
    return len(retrieved_k & relevant_set) / len(retrieved_k)
```

**Mean Reciprocal Rank (MRR):**
```python
def mrr(retrieved_ids, relevant_ids):
    """Reciprocal of rank of first relevant doc."""
    relevant_set = set(relevant_ids)
    
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    
    return 0.0
```

**NDCG (Normalized Discounted Cumulative Gain):**
```python
import numpy as np

def ndcg_at_k(relevance_scores, k):
    """
    relevance_scores: list of relevance scores in retrieval order
    """
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))
    
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores[:k]))
    
    return dcg / idcg if idcg > 0 else 0.0
```

### Generation Quality (RAG-specific)

**Faithfulness:**
Does the answer only use information from retrieved context?

```python
def evaluate_faithfulness(answer, context, llm):
    prompt = f"""Evaluate if the answer is fully supported by the context.
    
    Context: {context}
    
    Answer: {answer}
    
    For each claim in the answer, check if it's supported by the context.
    
    Verdict (fully_supported / partially_supported / not_supported):
    Explanation:"""
    
    response = llm.generate(prompt)
    return parse_faithfulness_response(response)
```

**Answer Relevance:**
Does the answer address the question?

```python
def evaluate_relevance(question, answer, llm):
    prompt = f"""Rate how well this answer addresses the question.
    
    Question: {question}
    Answer: {answer}
    
    Score (1-5):
    1 = Completely irrelevant
    2 = Slightly relevant
    3 = Partially addresses question
    4 = Mostly addresses question
    5 = Fully addresses question
    
    Score:
    Reasoning:"""
    
    response = llm.generate(prompt)
    return parse_relevance_response(response)
```

**Context Relevance:**
Is the retrieved context relevant to the question?

```python
def evaluate_context_relevance(question, context, llm):
    prompt = f"""Rate the relevance of this context for answering the question.
    
    Question: {question}
    Context: {context}
    
    Score (1-5):
    Reasoning:"""
    
    response = llm.generate(prompt)
    return parse_context_relevance(response)
```

### RAG Evaluation Frameworks

**RAGAS:**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy

# Prepare dataset
dataset = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

# Evaluate
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_relevancy]
)
print(result)
```

**TruLens:**
```python
from trulens_eval import Feedback, TruLlama

# Define feedback functions
f_groundedness = Feedback(groundedness_provider.groundedness_measure)
f_relevance = Feedback(relevance_provider.relevance)

# Wrap RAG app
tru_rag = TruLlama(
    rag_app,
    feedbacks=[f_groundedness, f_relevance]
)

# Run and evaluate
with tru_rag:
    response = rag_app.query("What is...")
```

## Hallucination Detection

### Types of Hallucination

| Type | Description | Example |
|------|-------------|---------|
| **Factual** | Contradicts known facts | "Paris is the capital of Germany" |
| **Faithfulness** | Contradicts provided context | Context says X, answer says Y |
| **Fabrication** | Invents non-existent entities | "According to Dr. Smith's 2023 study..." (doesn't exist) |
| **Inconsistency** | Self-contradictory | "The product costs $50... The $30 product..." |

### Detection Methods

**Entailment-based:**
```python
from transformers import pipeline

nli = pipeline("text-classification", model="roberta-large-mnli")

def check_entailment(premise, hypothesis):
    """Check if premise entails hypothesis."""
    result = nli(f"{premise} [SEP] {hypothesis}")
    return result[0]["label"]  # ENTAILMENT, CONTRADICTION, NEUTRAL
```

**Self-consistency:**
```python
def check_consistency(question, model, n_samples=5):
    """Generate multiple answers and check consistency."""
    answers = []
    for _ in range(n_samples):
        answer = model.generate(question, temperature=0.7)
        answers.append(answer)
    
    # Check if answers are consistent
    consistency_score = calculate_answer_similarity(answers)
    return consistency_score
```

**Knowledge verification:**
```python
def verify_claims(answer, knowledge_base):
    """Extract and verify factual claims."""
    claims = extract_claims(answer)
    
    verified = []
    for claim in claims:
        evidence = knowledge_base.search(claim)
        is_supported = check_support(claim, evidence)
        verified.append({
            "claim": claim,
            "supported": is_supported,
            "evidence": evidence
        })
    
    return verified
```

### Hallucination Benchmarks

| Benchmark | Focus | Size |
|-----------|-------|------|
| **TruthfulQA** | Factual accuracy | 817 questions |
| **HaluEval** | Various hallucination types | 35K samples |
| **FaithDial** | Dialogue faithfulness | 50K turns |
| **SummaC** | Summarization consistency | 1.6K samples |

## LLM-as-Judge

Using LLMs to evaluate other LLMs:

### Pairwise Comparison

```python
def pairwise_judge(question, answer_a, answer_b, judge_model):
    prompt = f"""Compare these two answers to the question.
    
    Question: {question}
    
    Answer A: {answer_a}
    
    Answer B: {answer_b}
    
    Which answer is better? Consider:
    - Accuracy
    - Completeness
    - Clarity
    - Relevance
    
    Winner (A/B/Tie):
    Reasoning:"""
    
    response = judge_model.generate(prompt)
    return parse_winner(response)
```

### Rubric-Based Scoring

```python
def rubric_judge(question, answer, judge_model):
    prompt = f"""Evaluate this answer using the rubric below.
    
    Question: {question}
    Answer: {answer}
    
    Rubric:
    - Accuracy (1-5): Is the information correct?
    - Completeness (1-5): Does it fully answer the question?
    - Clarity (1-5): Is it easy to understand?
    - Conciseness (1-5): Is it appropriately brief?
    
    Scores:
    Accuracy:
    Completeness:
    Clarity:
    Conciseness:
    
    Overall (1-5):
    Justification:"""
    
    response = judge_model.generate(prompt)
    return parse_rubric_scores(response)
```

### Best Practices for LLM-as-Judge

1. **Use strong judge models**: GPT-4, Claude 3.5 Sonnet
2. **Randomize order**: Avoid position bias in pairwise comparisons
3. **Provide clear criteria**: Specific rubrics reduce variance
4. **Multiple judges**: Average across multiple evaluations
5. **Calibrate with humans**: Validate judge correlates with human judgment

## Benchmarks

### General LLM Benchmarks

| Benchmark | Focus | Tasks |
|-----------|-------|-------|
| **MMLU** | Knowledge | 57 subjects, multiple choice |
| **HellaSwag** | Commonsense | Sentence completion |
| **ARC** | Science reasoning | Grade-school science |
| **WinoGrande** | Coreference | Pronoun resolution |
| **GSM8K** | Math | Grade-school math word problems |
| **HumanEval** | Coding | Python function completion |
| **TruthfulQA** | Truthfulness | Avoiding false claims |

### RAG Benchmarks

| Benchmark | Focus | Size |
|-----------|-------|------|
| **Natural Questions** | Wikipedia QA | 307K questions |
| **TriviaQA** | Trivia | 95K questions |
| **HotpotQA** | Multi-hop reasoning | 113K questions |
| **MS MARCO** | Web search | 1M queries |
| **BEIR** | Zero-shot retrieval | 18 datasets |

### Running Benchmarks

```python
from lm_eval import evaluator, tasks

# Run MMLU
results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-2-7b-hf",
    tasks=["mmlu"],
    num_fewshot=5
)

print(results["results"]["mmlu"]["acc"])
```

## Building Evaluation Datasets

### Golden Dataset Creation

```python
# Structure
evaluation_dataset = [
    {
        "id": "q001",
        "question": "What is the return policy for electronics?",
        "context": "Our electronics return policy allows...",  # For RAG
        "ground_truth": "30 days for unopened items, 15 days for opened",
        "metadata": {
            "category": "policy",
            "difficulty": "easy",
            "source": "FAQ"
        }
    },
    # ... more examples
]
```

### Data Collection Strategies

1. **From production logs**: Real user questions
2. **Expert annotation**: Domain experts create Q&A pairs
3. **Synthetic generation**: LLM generates questions from documents
4. **Adversarial**: Specifically designed to find failures

### Annotation Guidelines

```markdown
## Annotation Guidelines for QA Dataset

### Question Quality
- Questions should be clear and unambiguous
- Questions should be answerable from the provided context
- Include a mix of factual, reasoning, and comparison questions

### Answer Quality
- Answers should be complete but concise
- Include only information from the context
- Use consistent formatting

### Difficulty Levels
- Easy: Single fact lookup
- Medium: Requires synthesis of 2-3 facts
- Hard: Requires reasoning or inference
```

## Continuous Evaluation

### Online Evaluation

```python
class OnlineEvaluator:
    def __init__(self, metrics):
        self.metrics = metrics
        self.results = []
    
    def log_interaction(self, query, response, feedback=None):
        evaluation = {
            "timestamp": datetime.now(),
            "query": query,
            "response": response,
            "feedback": feedback,
            "metrics": {}
        }
        
        for metric in self.metrics:
            evaluation["metrics"][metric.name] = metric.compute(query, response)
        
        self.results.append(evaluation)
        self._check_alerts(evaluation)
    
    def _check_alerts(self, evaluation):
        for metric_name, value in evaluation["metrics"].items():
            if value < self.thresholds.get(metric_name, 0):
                self._send_alert(metric_name, value)
```

### A/B Testing

```python
class ABTestEvaluator:
    def __init__(self, variant_a, variant_b, metrics):
        self.variants = {"A": variant_a, "B": variant_b}
        self.metrics = metrics
        self.results = {"A": [], "B": []}
    
    def run_test(self, queries, assignment_ratio=0.5):
        for query in queries:
            variant = "A" if random.random() < assignment_ratio else "B"
            response = self.variants[variant].generate(query)
            
            scores = {m.name: m.compute(query, response) for m in self.metrics}
            self.results[variant].append(scores)
        
        return self.analyze_results()
    
    def analyze_results(self):
        from scipy import stats
        
        analysis = {}
        for metric in self.metrics:
            a_scores = [r[metric.name] for r in self.results["A"]]
            b_scores = [r[metric.name] for r in self.results["B"]]
            
            t_stat, p_value = stats.ttest_ind(a_scores, b_scores)
            
            analysis[metric.name] = {
                "mean_A": np.mean(a_scores),
                "mean_B": np.mean(b_scores),
                "p_value": p_value,
                "significant": p_value < 0.05
            }
        
        return analysis
```

### Regression Testing

```python
class RegressionTester:
    def __init__(self, golden_dataset, metrics, threshold=0.95):
        self.golden = golden_dataset
        self.metrics = metrics
        self.threshold = threshold
        self.baseline_scores = None
    
    def set_baseline(self, model):
        self.baseline_scores = self._evaluate(model)
    
    def test_model(self, model):
        current_scores = self._evaluate(model)
        
        regressions = []
        for metric_name, current in current_scores.items():
            baseline = self.baseline_scores[metric_name]
            ratio = current / baseline if baseline > 0 else 0
            
            if ratio < self.threshold:
                regressions.append({
                    "metric": metric_name,
                    "baseline": baseline,
                    "current": current,
                    "ratio": ratio
                })
        
        return {
            "passed": len(regressions) == 0,
            "regressions": regressions,
            "scores": current_scores
        }
    
    def _evaluate(self, model):
        scores = {m.name: [] for m in self.metrics}
        
        for example in self.golden:
            response = model.generate(example["question"])
            for metric in self.metrics:
                score = metric.compute(
                    example["question"],
                    response,
                    example.get("ground_truth")
                )
                scores[metric.name].append(score)
        
        return {name: np.mean(values) for name, values in scores.items()}
```

## References

### Papers
- Lin (2004). "ROUGE: A Package for Automatic Evaluation of Summaries"
- Papineni et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation"
- Zhang et al. (2019). "BERTScore: Evaluating Text Generation with BERT"
- Es et al. (2023). "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
- Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"

### Resources
- Hugging Face Evaluate library
- EleutherAI LM Evaluation Harness
- RAGAS documentation
- TruLens documentation
