# Fine-Tuning and Alignment

This document covers techniques for adapting LLMs to specific tasks through fine-tuning and alignment.

## When to Fine-Tune

### Decision Framework

| Approach | When to Use | Effort | Cost |
|----------|-------------|--------|------|
| **Prompt Engineering** | Quick iteration, no training data | Low | Low |
| **RAG** | Need current/proprietary knowledge | Medium | Medium |
| **Fine-Tuning** | Specific style, format, domain | High | High |
| **RLHF/DPO** | Align with human preferences | Very High | Very High |

### Fine-Tuning Use Cases

**Good candidates:**
- Consistent output format (JSON, specific structure)
- Domain-specific terminology and style
- Task-specific behavior patterns
- Reducing prompt length (internalize instructions)
- Improving reliability on specific tasks

**Poor candidates:**
- Adding new factual knowledge (use RAG)
- One-off tasks (use prompting)
- Rapidly changing requirements
- Limited training data (<100 examples)

## Fine-Tuning Methods

### Full Fine-Tuning

Update all model parameters on task-specific data.

**Process:**
```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

**Considerations:**
- Requires significant GPU memory (full model in memory)
- Risk of catastrophic forgetting
- Best results but highest cost
- Typically needs 1000+ examples

### Parameter-Efficient Fine-Tuning (PEFT)

Update only a small subset of parameters.

#### LoRA (Low-Rank Adaptation)

**Concept:**
Instead of updating weight matrix W, add low-rank decomposition:
```
W' = W + BA
where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)
```

**Implementation:**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                      # Rank
    lora_alpha=32,             # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
print(f"Trainable params: {model.print_trainable_parameters()}")
# Typically 0.1-1% of total parameters
```

**Hyperparameters:**
| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `r` (rank) | 4-64 | Higher = more capacity, more params |
| `lora_alpha` | 16-64 | Scaling, often 2×r |
| `target_modules` | Attention layers | Which layers to adapt |
| `lora_dropout` | 0-0.1 | Regularization |

#### QLoRA (Quantized LoRA)

Combine 4-bit quantization with LoRA for memory efficiency.

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Add LoRA
model = get_peft_model(model, lora_config)
```

**Memory comparison:**
| Method | 7B Model Memory |
|--------|-----------------|
| Full FP32 | ~28 GB |
| Full FP16 | ~14 GB |
| LoRA FP16 | ~14 GB + ~100 MB |
| QLoRA 4-bit | ~4 GB + ~100 MB |

#### Other PEFT Methods

**Prefix Tuning:**
- Add learnable prefix tokens to input
- ~0.1% parameters
- Good for generation tasks

**Prompt Tuning:**
- Learn soft prompt embeddings
- Even fewer parameters
- Task-specific prompts

**Adapters:**
- Small bottleneck layers between transformer layers
- ~1-5% parameters
- Modular, can combine multiple adapters

### Comparison

| Method | Parameters | Memory | Quality | Speed |
|--------|------------|--------|---------|-------|
| Full Fine-Tune | 100% | High | Best | Slow |
| LoRA | 0.1-1% | Medium | Very Good | Medium |
| QLoRA | 0.1-1% | Low | Good | Medium |
| Prefix Tuning | 0.1% | Low | Good | Fast |
| Adapters | 1-5% | Medium | Good | Medium |

## Supervised Fine-Tuning (SFT)

### Data Format

**Instruction format:**
```json
{
  "instruction": "Summarize the following article",
  "input": "The article text goes here...",
  "output": "This article discusses..."
}
```

**Chat format:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ]
}
```

### Data Preparation

**Quality over quantity:**
- 100 high-quality examples > 10,000 low-quality
- Diverse examples covering edge cases
- Consistent format and style
- Human-verified outputs

**Data augmentation:**
```python
def augment_instruction(example, llm):
    # Rephrase instruction
    prompt = f"Rephrase this instruction: {example['instruction']}"
    new_instruction = llm.generate(prompt)
    
    return {
        "instruction": new_instruction,
        "input": example["input"],
        "output": example["output"]
    }
```

### Training Process

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",  # Or use formatting_func
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="./sft-model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
    ),
    peft_config=lora_config,  # Optional: use PEFT
)

trainer.train()
```

### Best Practices

1. **Start with strong base model**: Better base = better fine-tuned
2. **Use instruction-tuned base**: Already knows how to follow instructions
3. **Match training format to inference**: Same chat template
4. **Include system prompts in training**: If you'll use them at inference
5. **Validate on held-out set**: Prevent overfitting
6. **Monitor loss curves**: Should decrease smoothly

## Preference Tuning

### RLHF (Reinforcement Learning from Human Feedback)

**Three-stage process:**

**Stage 1: Supervised Fine-Tuning (SFT)**
- Train on high-quality demonstrations
- Creates initial instruction-following model

**Stage 2: Reward Model Training**
```python
# Preference data format
{
  "prompt": "Explain quantum computing",
  "chosen": "Quantum computing uses quantum bits...",  # Preferred
  "rejected": "Quantum computing is complicated..."   # Less preferred
}
```

Train reward model to predict human preferences:
```python
from trl import RewardTrainer

reward_trainer = RewardTrainer(
    model=reward_model,
    train_dataset=preference_dataset,
    args=TrainingArguments(...),
)
reward_trainer.train()
```

**Stage 3: RL Optimization (PPO)**
```python
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=sft_model,
    ref_model=sft_model,  # Reference for KL penalty
    tokenizer=tokenizer,
    reward_model=reward_model,
)

# Training loop
for batch in dataloader:
    queries = batch["query"]
    responses = ppo_trainer.generate(queries)
    rewards = reward_model(queries, responses)
    ppo_trainer.step(queries, responses, rewards)
```

**Challenges:**
- Complex pipeline
- Reward hacking
- Training instability
- High computational cost

### DPO (Direct Preference Optimization)

Simpler alternative that directly optimizes on preferences without separate reward model.

**Key insight:** Optimal policy can be derived directly from preferences.

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,  # KL penalty coefficient
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
)

dpo_trainer = DPOTrainer(
    model=sft_model,
    ref_model=sft_model,  # Reference model (frozen)
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
    args=dpo_config,
)

dpo_trainer.train()
```

**Advantages over RLHF:**
- No separate reward model
- More stable training
- Simpler implementation
- Lower computational cost

**Data format:**
```json
{
  "prompt": "Write a poem about nature",
  "chosen": "The gentle breeze whispers through trees...",
  "rejected": "Nature is nice. Trees are green..."
}
```

### Other Preference Methods

**IPO (Identity Preference Optimization):**
- Addresses DPO's length bias
- More robust to noisy preferences

**KTO (Kahneman-Tversky Optimization):**
- Works with binary feedback (good/bad)
- No need for paired preferences

### ORPO (Odds Ratio Preference Optimization)

ORPO eliminates the need for a separate SFT stage by combining instruction tuning with preference alignment in a single training phase.

**Key insight:** A minor penalty for disfavored responses during SFT is sufficient for preference alignment.

```python
from trl import ORPOTrainer, ORPOConfig

orpo_config = ORPOConfig(
    beta=0.1,  # Odds ratio weight
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    # No reference model needed!
)

orpo_trainer = ORPOTrainer(
    model=base_model,  # Start from base, not SFT model
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
    args=orpo_config,
)

orpo_trainer.train()
```

**Advantages:**
- Single-stage training (no separate SFT)
- No reference model required (saves memory)
- Competitive with DPO on benchmarks
- Simpler pipeline

### SimPO (Simple Preference Optimization)

SimPO uses average log probability as implicit reward, eliminating reference model dependency.

**Key innovation:** Reward = average log probability of sequence (not sum)

```python
from trl import SimPOTrainer, SimPOConfig

simpo_config = SimPOConfig(
    beta=2.0,  # Reward scaling
    gamma=0.5,  # Target reward margin
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    num_train_epochs=1,
)

simpo_trainer = SimPOTrainer(
    model=sft_model,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
    args=simpo_config,
)

simpo_trainer.train()
```

**Advantages over DPO:**
- No reference model (50% memory savings)
- Better length normalization
- Up to 6.4 points improvement on AlpacaEval 2
- Simpler implementation

### Preference Method Comparison

| Method | Reference Model | Stages | Memory | Quality |
|--------|-----------------|--------|--------|---------|
| **RLHF** | Yes + Reward | 3 | High | High |
| **DPO** | Yes | 2 (SFT + DPO) | Medium | High |
| **ORPO** | No | 1 | Low | High |
| **SimPO** | No | 2 (SFT + SimPO) | Low | Highest |
| **KTO** | Yes | 2 | Medium | Medium |

**Recommendation:** Start with ORPO for simplicity. Use SimPO for best quality. Use DPO if you need reference model for other purposes.

### Collecting Preference Data

**Human annotation:**
- Most reliable but expensive
- Use clear guidelines
- Multiple annotators for quality

**AI feedback:**
```python
def get_ai_preference(prompt, response_a, response_b, judge_model):
    judge_prompt = f"""Compare these two responses to the prompt.
    
    Prompt: {prompt}
    
    Response A: {response_a}
    Response B: {response_b}
    
    Which response is better? Answer A or B with brief explanation."""
    
    judgment = judge_model.generate(judge_prompt)
    return parse_preference(judgment)
```

**Synthetic preferences:**
- Generate multiple responses
- Use stronger model to rank
- Constitutional AI approach

## Synthetic Data Generation

Synthetic data from LLMs can dramatically reduce the need for human annotation.

### Self-Instruct

Generate instruction-response pairs from seed examples:

```python
def self_instruct(seed_tasks, llm, num_generate=1000):
    """
    Generate new instruction-response pairs from seed examples.
    Based on Stanford Alpaca approach.
    """
    generated = []
    
    for i in range(num_generate):
        # Sample seed examples for few-shot
        examples = random.sample(seed_tasks, k=3)
        
        prompt = f"""Generate a new instruction-response pair similar to these examples:

{format_examples(examples)}

New instruction:"""
        
        instruction = llm.generate(prompt)
        
        # Generate response
        response_prompt = f"Instruction: {instruction}\nResponse:"
        response = llm.generate(response_prompt)
        
        generated.append({
            "instruction": instruction,
            "response": response
        })
    
    return generated
```

### Evol-Instruct (WizardLM)

Evolve simple instructions into complex ones:

```python
def evol_instruct(instruction, llm, evolution_type="deepen"):
    """
    Evolve instructions to increase complexity.
    """
    evolution_prompts = {
        "deepen": f"""Make this instruction more complex by adding constraints:
Original: {instruction}
Evolved:""",
        
        "concretize": f"""Make this instruction more specific with concrete details:
Original: {instruction}
Evolved:""",
        
        "reasoning": f"""Rewrite to require multi-step reasoning:
Original: {instruction}
Evolved:""",
        
        "breadth": f"""Create a related but different instruction:
Original: {instruction}
New:"""
    }
    
    return llm.generate(evolution_prompts[evolution_type])

# Example evolution
simple = "Write a function to sort a list"
complex = evol_instruct(simple, llm, "deepen")
# "Write a function to sort a list of dictionaries by multiple keys,
#  handling None values and supporting both ascending and descending order"
```

### Generating Preference Data

Create preference pairs using model-as-judge:

```python
def generate_preference_data(instructions, llm, judge_model):
    """
    Generate preference pairs by sampling and judging.
    """
    preference_data = []
    
    for instruction in instructions:
        # Generate multiple responses with different temperatures
        responses = [
            llm.generate(instruction, temperature=t)
            for t in [0.3, 0.7, 1.0]
        ]
        
        # Judge pairwise
        for i, resp_a in enumerate(responses):
            for j, resp_b in enumerate(responses):
                if i >= j:
                    continue
                
                judgment = judge_model.generate(f"""
Which response is better for this instruction?

Instruction: {instruction}

Response A: {resp_a}

Response B: {resp_b}

Better response (A or B):""")
                
                if "A" in judgment:
                    chosen, rejected = resp_a, resp_b
                else:
                    chosen, rejected = resp_b, resp_a
                
                preference_data.append({
                    "prompt": instruction,
                    "chosen": chosen,
                    "rejected": rejected
                })
    
    return preference_data
```

### Quality Filtering

Filter synthetic data for quality:

```python
def filter_synthetic_data(data, quality_model, threshold=0.7):
    """
    Filter synthetic data using quality scoring.
    """
    filtered = []
    
    for item in data:
        # Score instruction quality
        instruction_score = quality_model.score(
            f"Rate instruction clarity (0-1): {item['instruction']}"
        )
        
        # Score response quality
        response_score = quality_model.score(
            f"Rate response helpfulness (0-1): {item['response']}"
        )
        
        # Check for hallucination/errors
        factual_score = quality_model.score(
            f"Rate factual accuracy (0-1): {item['response']}"
        )
        
        avg_score = (instruction_score + response_score + factual_score) / 3
        
        if avg_score >= threshold:
            filtered.append(item)
    
    return filtered
```

### Distillation from Stronger Models

Use outputs from stronger models to train smaller ones:

```python
def distill_from_teacher(prompts, teacher_model, student_tokenizer):
    """
    Generate training data by distilling from a stronger model.
    """
    training_data = []
    
    for prompt in prompts:
        # Get high-quality response from teacher
        teacher_response = teacher_model.generate(
            prompt,
            temperature=0.7,
            max_tokens=1024
        )
        
        training_data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": teacher_response}
            ]
        })
    
    return training_data

# Example: Distill GPT-4 into Llama
prompts = load_diverse_prompts()
training_data = distill_from_teacher(prompts, gpt4_client, llama_tokenizer)
```

### Synthetic Data Best Practices

| Aspect | Recommendation |
|--------|----------------|
| **Diversity** | Use diverse seed tasks, vary generation parameters |
| **Quality** | Always filter; use stronger model as judge |
| **Volume** | 1K-10K high-quality > 100K low-quality |
| **Deduplication** | Remove near-duplicates (embedding similarity) |
| **Validation** | Hold out test set from real data |
| **Mixing** | Combine synthetic with real data (80/20 ratio) |

## Alignment Considerations

### Instructional vs Value Alignment

**Instructional alignment:**
- Follow user instructions accurately
- Maintain helpful behavior
- Achieved through SFT

**Value alignment:**
- Refuse harmful requests
- Avoid biased outputs
- Achieved through RLHF/DPO with safety data

### Safety Fine-Tuning

Include safety examples in training:
```json
{
  "prompt": "How do I hack into someone's account?",
  "chosen": "I can't help with hacking into accounts as that would be illegal...",
  "rejected": "Here's how you could potentially access someone's account..."
}
```

### Avoiding Alignment Tax

Fine-tuning can degrade safety. Mitigations:
- Include safety data in fine-tuning mix
- Use safety-focused base models
- Evaluate safety metrics during training
- Apply safety fine-tuning after task fine-tuning

## Practical Recipes

### Recipe 1: Domain Adaptation (QLoRA)

For adapting to specific domain (legal, medical, etc.):

```python
# 1. Prepare domain data
dataset = load_domain_dataset()  # ~1000-10000 examples

# 2. Configure QLoRA
bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)
lora_config = LoraConfig(r=16, lora_alpha=32, ...)

# 3. Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=TrainingArguments(
        num_train_epochs=3,
        learning_rate=2e-4,
        ...
    )
)
trainer.train()

# 4. Merge and save
model = model.merge_and_unload()
model.save_pretrained("domain-adapted-model")
```

### Recipe 2: Output Format Control

For consistent JSON/structured output:

```python
# Training data with strict format
examples = [
    {
        "instruction": "Extract entities from: 'John works at Google'",
        "output": '{"persons": ["John"], "organizations": ["Google"]}'
    },
    # ... more examples
]

# Fine-tune with format-focused data
# Include edge cases and error handling
```

### Recipe 3: Style Transfer

For specific writing style:

```python
# Collect style examples
style_examples = [
    {"input": "generic text", "output": "text in target style"},
    # ...
]

# Fine-tune on style pairs
# Include diverse topics to generalize style
```

### Recipe 4: Task-Specific Agent

For tool-use or agent behavior:

```python
# Training data with tool calls
examples = [
    {
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"name": "get_weather", "arguments": {"city": "Paris"}}
            ]},
            {"role": "tool", "content": "Sunny, 22°C"},
            {"role": "assistant", "content": "The weather in Paris is sunny at 22°C."}
        ]
    }
]
```

## Evaluation

### During Training

Monitor:
- Training loss (should decrease)
- Validation loss (watch for overfitting)
- Learning rate schedule
- Gradient norms

### Post-Training

**Automated metrics:**
```python
def evaluate_model(model, test_set):
    results = {
        "exact_match": [],
        "f1": [],
        "format_compliance": []
    }
    
    for example in test_set:
        output = model.generate(example["input"])
        results["exact_match"].append(output == example["expected"])
        results["f1"].append(compute_f1(output, example["expected"]))
        results["format_compliance"].append(is_valid_format(output))
    
    return {k: np.mean(v) for k, v in results.items()}
```

**Human evaluation:**
- Side-by-side comparison
- Rating scales (1-5)
- Task-specific criteria

**Safety evaluation:**
- Test with adversarial prompts
- Check for capability degradation
- Verify refusal behavior

## Common Issues

### Catastrophic Forgetting

**Symptoms:** Model loses general capabilities after fine-tuning

**Solutions:**
- Use PEFT (LoRA) instead of full fine-tuning
- Include general data in training mix
- Lower learning rate
- Fewer epochs

### Overfitting

**Symptoms:** Low training loss, high validation loss

**Solutions:**
- More diverse training data
- Regularization (dropout, weight decay)
- Early stopping
- Data augmentation

### Mode Collapse

**Symptoms:** Model gives same/similar outputs

**Solutions:**
- More diverse training data
- Temperature during training
- Check for data imbalance

### Format Inconsistency

**Symptoms:** Model doesn't follow output format

**Solutions:**
- More format examples
- Stricter format in training data
- Include format in system prompt
- Post-processing validation

## References

### Papers
- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
- Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs"
- Ouyang et al. (2022). "Training language models to follow instructions with human feedback"
- Rafailov et al. (2023). "Direct Preference Optimization"
- Ziegler et al. (2019). "Fine-Tuning Language Models from Human Preferences"

### Resources
- Hugging Face PEFT documentation
- TRL (Transformer Reinforcement Learning) library
- Axolotl fine-tuning framework
- LLaMA-Factory
