# LLM Agents and Tool Use

This document covers the design and implementation of LLM-powered agents that can reason, plan, and take actions.

## What Are LLM Agents?

An LLM agent extends a language model with:
- **Memory**: Store and recall information across interactions
- **Tools**: Execute actions in the external world
- **Planning**: Break down complex tasks into steps
- **Reasoning**: Think through problems systematically

```
┌─────────────────────────────────────────────────────────────┐
│                        LLM Agent                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Memory  │  │ Planning │  │  Tools   │  │   LLM    │   │
│  │          │  │          │  │          │  │ (Brain)  │   │
│  │ - Short  │  │ - Goals  │  │ - Search │  │          │   │
│  │ - Long   │  │ - Steps  │  │ - Code   │  │ - Reason │   │
│  │ - Vector │  │ - Revise │  │ - APIs   │  │ - Decide │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Environment   │
                    │  (APIs, DBs,    │
                    │   File System)  │
                    └─────────────────┘
```

## Agent Reasoning Patterns

### ReAct (Reasoning + Acting)

Interleave reasoning traces with actions:

```
User: What's the population of the capital of France?

Thought: I need to find the capital of France first, then its population.
Action: search("capital of France")
Observation: Paris is the capital of France.

Thought: Now I need to find the population of Paris.
Action: search("population of Paris 2024")
Observation: Paris has a population of approximately 2.1 million.

Thought: I have the answer.
Answer: The capital of France is Paris, with a population of approximately 2.1 million.
```

**Implementation:**
```python
REACT_PROMPT = """Answer the question using the available tools.

Available tools:
{tools}

Use this format:
Thought: reasoning about what to do
Action: tool_name(arguments)
Observation: tool result
... (repeat as needed)
Thought: I have the answer
Answer: final answer

Question: {question}
"""

def react_agent(question, tools, llm, max_steps=10):
    prompt = REACT_PROMPT.format(tools=format_tools(tools), question=question)
    
    for step in range(max_steps):
        response = llm.generate(prompt)
        
        if "Answer:" in response:
            return extract_answer(response)
        
        action = extract_action(response)
        observation = execute_tool(action, tools)
        
        prompt += f"\nObservation: {observation}\n"
    
    return "Could not find answer within step limit"
```

### Chain-of-Thought (CoT)

Encourage step-by-step reasoning:

```
Question: If a train travels at 60 mph for 2.5 hours, how far does it go?

Let me solve this step by step:
1. I need to use the formula: distance = speed × time
2. Speed = 60 mph
3. Time = 2.5 hours
4. Distance = 60 × 2.5 = 150 miles

The train travels 150 miles.
```

**Variants:**
- **Zero-shot CoT**: Add "Let's think step by step"
- **Few-shot CoT**: Provide reasoning examples
- **Self-consistency**: Generate multiple chains, vote on answer

### Plan-and-Execute

Separate planning from execution:

```python
def plan_and_execute(task, tools, llm):
    # Step 1: Create plan
    plan_prompt = f"""Create a step-by-step plan to accomplish this task:
    Task: {task}
    
    Available tools: {format_tools(tools)}
    
    Plan:"""
    
    plan = llm.generate(plan_prompt)
    steps = parse_plan(plan)
    
    # Step 2: Execute each step
    results = []
    for step in steps:
        execution_prompt = f"""Execute this step:
        Step: {step}
        Previous results: {results}
        
        Use tools as needed."""
        
        result = execute_step(step, tools, llm)
        results.append(result)
    
    # Step 3: Synthesize final answer
    return synthesize(task, results, llm)
```

### Reflexion

Self-reflection and improvement:

```python
def reflexion_agent(task, tools, llm, max_attempts=3):
    memory = []
    
    for attempt in range(max_attempts):
        # Try to solve
        result = react_agent(task, tools, llm)
        
        # Evaluate
        evaluation = evaluate_result(result, task)
        
        if evaluation["success"]:
            return result
        
        # Reflect on failure
        reflection_prompt = f"""The previous attempt failed.
        Task: {task}
        Attempt: {result}
        Feedback: {evaluation["feedback"]}
        
        What went wrong and how can I do better?"""
        
        reflection = llm.generate(reflection_prompt)
        memory.append({"attempt": result, "reflection": reflection})
        
        # Update prompt with reflections
        tools = update_with_reflections(tools, memory)
    
    return "Failed after max attempts"
```

## Tool Use / Function Calling

### Defining Tools

**OpenAI-style function definitions:**
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]
```

### Tool Execution

```python
def execute_tool(tool_call, available_tools):
    tool_name = tool_call["name"]
    arguments = tool_call["arguments"]
    
    # Find and execute tool
    for tool in available_tools:
        if tool["function"]["name"] == tool_name:
            func = tool["implementation"]
            return func(**arguments)
    
    raise ValueError(f"Unknown tool: {tool_name}")

# Tool implementations
def search_web(query: str, num_results: int = 5) -> str:
    # Actual search implementation
    results = web_search_api(query, num_results)
    return format_search_results(results)

def calculate(expression: str) -> str:
    # Safe math evaluation
    result = safe_eval(expression)
    return str(result)
```

### Parallel Tool Calls

Modern APIs support calling multiple tools simultaneously:

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# Response may contain multiple tool calls
tool_calls = response.choices[0].message.tool_calls

# Execute in parallel
import asyncio

async def execute_all_tools(tool_calls):
    tasks = [execute_tool_async(tc) for tc in tool_calls]
    return await asyncio.gather(*tasks)

results = asyncio.run(execute_all_tools(tool_calls))
```

### Common Tool Categories

| Category | Examples |
|----------|----------|
| **Information** | Web search, Wikipedia, knowledge bases |
| **Computation** | Calculator, code interpreter, data analysis |
| **Communication** | Email, Slack, notifications |
| **Data** | Database queries, file operations, APIs |
| **Actions** | Browser automation, system commands |

## Memory Systems

### Short-Term Memory (Context)

The conversation history within context window:

```python
class ConversationMemory:
    def __init__(self, max_tokens=4000):
        self.messages = []
        self.max_tokens = max_tokens
    
    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        self._trim_to_fit()
    
    def _trim_to_fit(self):
        while self._count_tokens() > self.max_tokens:
            # Remove oldest messages (keep system prompt)
            if len(self.messages) > 1:
                self.messages.pop(1)
    
    def get_messages(self):
        return self.messages
```

### Long-Term Memory (Vector Store)

Store and retrieve past interactions:

```python
class LongTermMemory:
    def __init__(self, embedding_model, vector_store):
        self.embedder = embedding_model
        self.store = vector_store
    
    def store_memory(self, content, metadata=None):
        embedding = self.embedder.encode(content)
        self.store.upsert({
            "id": generate_id(),
            "embedding": embedding,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        })
    
    def recall(self, query, top_k=5):
        query_embedding = self.embedder.encode(query)
        results = self.store.query(query_embedding, top_k=top_k)
        return [r["content"] for r in results]
```

### Episodic Memory

Store complete interaction episodes:

```python
class EpisodicMemory:
    def __init__(self):
        self.episodes = []
    
    def store_episode(self, task, steps, outcome):
        episode = {
            "task": task,
            "steps": steps,
            "outcome": outcome,
            "success": outcome["success"],
            "timestamp": datetime.now()
        }
        self.episodes.append(episode)
    
    def recall_similar_episodes(self, task, top_k=3):
        # Find similar past tasks
        similar = []
        for ep in self.episodes:
            similarity = compute_similarity(task, ep["task"])
            similar.append((similarity, ep))
        
        similar.sort(reverse=True)
        return [ep for _, ep in similar[:top_k]]
```

### Working Memory

Scratchpad for current task:

```python
class WorkingMemory:
    def __init__(self):
        self.scratchpad = {}
        self.current_goal = None
        self.subgoals = []
        self.observations = []
    
    def set_goal(self, goal):
        self.current_goal = goal
        self.subgoals = []
        self.observations = []
    
    def add_subgoal(self, subgoal):
        self.subgoals.append(subgoal)
    
    def add_observation(self, observation):
        self.observations.append(observation)
    
    def store(self, key, value):
        self.scratchpad[key] = value
    
    def recall(self, key):
        return self.scratchpad.get(key)
    
    def get_context(self):
        return {
            "goal": self.current_goal,
            "subgoals": self.subgoals,
            "observations": self.observations,
            "scratchpad": self.scratchpad
        }
```

## Multi-Agent Systems

### Coordinator Pattern

Central agent delegates to specialists:

```python
class CoordinatorAgent:
    def __init__(self, specialist_agents, llm):
        self.specialists = specialist_agents
        self.llm = llm
    
    def process(self, task):
        # Determine which specialist to use
        routing_prompt = f"""Given this task, which specialist should handle it?
        
        Task: {task}
        
        Available specialists:
        {self._format_specialists()}
        
        Return the specialist name."""
        
        specialist_name = self.llm.generate(routing_prompt)
        specialist = self.specialists[specialist_name]
        
        # Delegate to specialist
        result = specialist.process(task)
        
        # Optionally synthesize or validate
        return self._synthesize(task, result)
```

### Debate Pattern

Agents argue different perspectives:

```python
class DebateSystem:
    def __init__(self, agents, judge, rounds=3):
        self.agents = agents
        self.judge = judge
        self.rounds = rounds
    
    def debate(self, question):
        positions = {agent.name: [] for agent in self.agents}
        
        for round in range(self.rounds):
            for agent in self.agents:
                # Agent makes argument considering others' positions
                argument = agent.argue(question, positions)
                positions[agent.name].append(argument)
        
        # Judge decides
        verdict = self.judge.decide(question, positions)
        return verdict
```

### Critic Pattern

One agent generates, another critiques:

```python
class GeneratorCritic:
    def __init__(self, generator, critic, max_iterations=3):
        self.generator = generator
        self.critic = critic
        self.max_iterations = max_iterations
    
    def generate(self, task):
        output = self.generator.generate(task)
        
        for i in range(self.max_iterations):
            critique = self.critic.critique(task, output)
            
            if critique["approved"]:
                return output
            
            # Revise based on critique
            output = self.generator.revise(task, output, critique["feedback"])
        
        return output
```

## Agent Frameworks

### LangChain Agents

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

# Define tools
tools = [
    Tool(
        name="Search",
        func=search_function,
        description="Search the web for information"
    ),
    Tool(
        name="Calculator",
        func=calculate_function,
        description="Perform calculations"
    )
]

# Create agent
llm = ChatOpenAI(model="gpt-4")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run
result = agent_executor.invoke({"input": "What is 25% of the US population?"})
```

### LlamaIndex Agents

```python
from llama_index.agent import ReActAgent
from llama_index.tools import FunctionTool
from llama_index.llms import OpenAI

# Define tools
def search(query: str) -> str:
    """Search for information."""
    return search_api(query)

search_tool = FunctionTool.from_defaults(fn=search)

# Create agent
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools([search_tool], llm=llm, verbose=True)

# Run
response = agent.chat("What's the weather in Tokyo?")
```

### AutoGen

```python
from autogen import AssistantAgent, UserProxyAgent

# Create agents
assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4"}
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"}
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to calculate fibonacci numbers"
)
```

## Model Context Protocol (MCP)

MCP is an open standard for connecting AI systems with external tools and data sources.

### MCP Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   MCP Client    │────▶│   MCP Server    │────▶│  External Tool  │
│  (LLM/Agent)    │◀────│  (Tool Provider)│◀────│   (API, DB)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Key concepts:**
- **Resources**: Data sources the server exposes (files, databases, APIs)
- **Tools**: Functions the server provides for the client to call
- **Prompts**: Pre-defined prompt templates the server offers

### MCP Server Example

```python
from mcp import Server, Tool, Resource

server = Server("my-tools")

@server.tool()
def search_database(query: str, limit: int = 10) -> list:
    """Search the product database."""
    return db.search(query, limit=limit)

@server.resource("products/{id}")
def get_product(id: str) -> dict:
    """Get product details by ID."""
    return db.get_product(id)

@server.prompt("product-summary")
def product_summary_prompt(product_id: str) -> str:
    """Generate a prompt for summarizing a product."""
    product = db.get_product(product_id)
    return f"Summarize this product: {product}"

server.run()
```

### MCP Client Usage

```python
from mcp import Client

# Connect to MCP server
client = Client("http://localhost:8080")

# List available tools
tools = client.list_tools()

# Call a tool
results = client.call_tool("search_database", {"query": "laptop", "limit": 5})

# Read a resource
product = client.read_resource("products/123")
```

**Benefits of MCP:**
- Standardized tool interface across providers
- Decoupled tool development from agent development
- Easy tool discovery and composition
- Security through capability-based access

## Structured Outputs

Ensure LLM outputs conform to specific schemas.

### JSON Mode

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract: John is 30 years old"}],
    response_format={"type": "json_object"}
)

# Guaranteed valid JSON
data = json.loads(response.choices[0].message.content)
```

### Structured Outputs with Schema

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str | None = None

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract: John is 30, works as engineer"}],
    response_format=Person
)

person = response.choices[0].message.parsed  # Typed Person object
```

### Grammar-Constrained Decoding

For self-hosted models, use grammar constraints:

```python
from outlines import models, generate

model = models.transformers("mistralai/Mistral-7B-v0.1")

# Define JSON schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

generator = generate.json(model, schema)
result = generator("Extract person info: John is 30 years old")
# Guaranteed to match schema
```

### Tool Call Structured Outputs

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "strict": True,  # Enable structured outputs
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"],
            "additionalProperties": False
        }
    }
}]

# Tool calls will always match the schema exactly
```

## Prompt Caching and Optimization

### Prompt Caching

Reuse computed attention states for repeated prompt prefixes.

**OpenAI/Anthropic automatic caching:**
```python
# First call - full computation
response1 = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": long_system_prompt},  # Cached
        {"role": "user", "content": "Question 1"}
    ]
)

# Second call - prefix cached (50-90% cost savings)
response2 = client.chat.completions.create(
    model="gpt-4o", 
    messages=[
        {"role": "system", "content": long_system_prompt},  # Reused from cache
        {"role": "user", "content": "Question 2"}
    ]
)
```

**Best practices for caching:**
- Put static content (system prompts, examples) at the beginning
- Keep dynamic content (user query) at the end
- Use consistent prompt prefixes across requests
- Minimum cacheable prefix: ~1024 tokens (varies by provider)

### KV Cache Optimization

For self-hosted models:

```python
# Precompute KV cache for static prefix
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")

# Encode static prefix once
prefix_tokens = tokenizer.encode(system_prompt, return_tensors="pt")
with torch.no_grad():
    prefix_outputs = model(prefix_tokens, use_cache=True)
    cached_kv = prefix_outputs.past_key_values

# Reuse for multiple queries
for query in queries:
    query_tokens = tokenizer.encode(query, return_tensors="pt")
    outputs = model(
        query_tokens,
        past_key_values=cached_kv,  # Reuse cached prefix
        use_cache=True
    )
```

**Cost savings from caching:**

| Provider | Cache Hit Discount |
|----------|-------------------|
| OpenAI | 50% off cached tokens |
| Anthropic | 90% off cached tokens |
| Self-hosted | ~8-60x latency reduction |

## Production Considerations

### Error Handling

```python
class RobustAgent:
    def __init__(self, agent, max_retries=3):
        self.agent = agent
        self.max_retries = max_retries
    
    def run(self, task):
        for attempt in range(self.max_retries):
            try:
                result = self.agent.run(task)
                if self._validate_result(result):
                    return result
            except ToolExecutionError as e:
                logger.warning(f"Tool error: {e}")
                continue
            except LLMError as e:
                logger.error(f"LLM error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return self._fallback_response(task)
```

### Rate Limiting

```python
from ratelimit import limits, sleep_and_retry

class RateLimitedAgent:
    @sleep_and_retry
    @limits(calls=60, period=60)  # 60 calls per minute
    def call_llm(self, prompt):
        return self.llm.generate(prompt)
    
    @sleep_and_retry
    @limits(calls=100, period=60)
    def call_tool(self, tool, args):
        return tool(**args)
```

### Monitoring and Logging

```python
import structlog

logger = structlog.get_logger()

class MonitoredAgent:
    def run(self, task):
        run_id = generate_run_id()
        
        logger.info("agent_run_started", run_id=run_id, task=task)
        
        start_time = time.time()
        steps = []
        
        try:
            for step in self._execute(task):
                steps.append(step)
                logger.info("agent_step", 
                    run_id=run_id,
                    step_type=step["type"],
                    step_content=step["content"]
                )
            
            result = self._finalize(steps)
            
            logger.info("agent_run_completed",
                run_id=run_id,
                duration=time.time() - start_time,
                num_steps=len(steps),
                success=True
            )
            
            return result
            
        except Exception as e:
            logger.error("agent_run_failed",
                run_id=run_id,
                error=str(e),
                steps_completed=len(steps)
            )
            raise
```

### Cost Control

```python
class CostAwareAgent:
    def __init__(self, agent, budget_per_task=1.0):
        self.agent = agent
        self.budget = budget_per_task
        self.spent = 0
    
    def run(self, task):
        self.spent = 0
        
        for step in self.agent.iterate(task):
            step_cost = self._estimate_cost(step)
            
            if self.spent + step_cost > self.budget:
                logger.warning("Budget exceeded, stopping early")
                return self._early_termination(task)
            
            self.spent += step_cost
            yield step
    
    def _estimate_cost(self, step):
        # Estimate based on tokens
        input_tokens = count_tokens(step["input"])
        output_tokens = count_tokens(step["output"])
        return (input_tokens * 0.00001) + (output_tokens * 0.00003)
```

### Sandboxing Tool Execution

```python
import subprocess
import tempfile

class SandboxedCodeExecutor:
    def __init__(self, timeout=30, memory_limit="256m"):
        self.timeout = timeout
        self.memory_limit = memory_limit
    
    def execute(self, code: str) -> str:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            
            try:
                result = subprocess.run(
                    ["docker", "run", "--rm",
                     "--memory", self.memory_limit,
                     "--network", "none",  # No network access
                     "-v", f"{f.name}:/code.py:ro",
                     "python:3.11-slim",
                     "python", "/code.py"],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                return result.stdout or result.stderr
            except subprocess.TimeoutExpired:
                return "Execution timed out"
```

## Common Patterns and Anti-Patterns

### Patterns

**Graceful degradation:**
```python
def agent_with_fallback(task):
    try:
        return advanced_agent.run(task)
    except Exception:
        return simple_agent.run(task)  # Simpler but more reliable
```

**Progressive disclosure:**
```python
def progressive_agent(task, detail_level="summary"):
    result = agent.run(task)
    
    if detail_level == "summary":
        return summarize(result)
    elif detail_level == "detailed":
        return result
    elif detail_level == "full":
        return result + agent.get_reasoning_trace()
```

### Anti-Patterns

**Infinite loops:**
```python
# BAD: No termination condition
while True:
    result = agent.step()

# GOOD: Max steps and success condition
for step in range(max_steps):
    result = agent.step()
    if result.is_complete:
        break
```

**Unbounded tool calls:**
```python
# BAD: Agent can call tools indefinitely
# GOOD: Limit tool calls per task
if tool_call_count > max_tool_calls:
    return "Exceeded tool call limit"
```

**No validation:**
```python
# BAD: Trust tool output blindly
# GOOD: Validate tool responses
result = tool.execute(args)
if not validate_result(result):
    return handle_invalid_result(result)
```

## References

### Papers
- Yao et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models"
- Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- Shinn et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning"
- Wang et al. (2023). "Self-Consistency Improves Chain of Thought Reasoning"
- Park et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior"

### Frameworks
- LangChain documentation
- LlamaIndex documentation
- AutoGen documentation
- CrewAI documentation
