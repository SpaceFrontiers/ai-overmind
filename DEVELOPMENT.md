# Skill Development Guide

This guide provides detailed instructions for developing Agent Skills using AI and research tools.

## Table of Contents

- [Overview](#overview)
- [Research Tools](#research-tools)
- [Development Workflow](#development-workflow)
- [Space Frontiers Integration](#space-frontiers-integration)
- [AI-Powered Generation](#ai-powered-generation)
- [Quality Assurance](#quality-assurance)
- [Examples](#examples)

## Overview

This repository uses a research-driven approach to skill development:

1. **Research**: Gather authoritative information from papers, documentation, and community knowledge
2. **Synthesize**: Extract patterns, best practices, and expert recommendations
3. **Generate**: Use AI to create comprehensive skill content
4. **Validate**: Test and refine based on real-world usage

## Research Tools

### Space Frontiers MCP Server

Space Frontiers provides access to vast knowledge sources for building accurate, well-researched skills.

#### Available Sources

- **journal-article**: All scholarly articles (ArXiv, BioRxiv, MedRxiv, PubMed)
- **wiki**: Wikipedia articles
- **pubmed**: PubMed medical literature
- **arxiv**: ArXiv preprints
- **biorxiv**: BioRxiv preprints
- **medrxiv**: MedRxiv preprints
- **telegram**: Telegram posts and messages
- **reddit**: Reddit discussions
- **youtube**: YouTube video transcripts

#### Core Functions

```python
# 1. Search across sources
mcp2_search(
    query="your search query",
    source="journal-article",  # Optional: specify source
    limit=20                   # Number of results
)

# 2. Get document with filtered content
mcp2_get_document(
    document_uri="doi://10.xxxx/example",  # From search results
    query="specific topic within document",
    mode="focused"  # or "wide" for more coverage
)

# 3. Get document metadata only (fast)
mcp2_get_document_metadata(
    document_uri="doi://10.xxxx/example"
)
```

### Context7 MCP Server

For up-to-date library documentation:

```python
# 1. Resolve library ID
mcp0_resolve_library_id(
    libraryName="library name"
)

# 2. Get documentation
mcp0_get_library_docs(
    context7CompatibleLibraryID="/org/project",
    topic="specific topic",
    mode="code",  # or "info" for conceptual guides
    page=1
)
```

## Development Workflow

### Step 1: Define Skill Requirements

Before researching, clarify:

- **Purpose**: What task does this skill address?
- **Target Users**: Who will use this skill?
- **Scope**: What's included/excluded?
- **Success Criteria**: How do you measure effectiveness?

Example:
```yaml
Skill: Machine Learning Model Evaluator
Purpose: Guide proper evaluation of ML models
Target: Data scientists and ML engineers
Scope: Classification, regression, and clustering metrics
Success: Users can select and interpret appropriate metrics
```

### Step 2: Research Phase

#### Broad Research
Start with general searches to understand the landscape:

```python
# Search for academic papers and articles
results = mcp2_search(
    query="machine learning model evaluation best practices",
    source="journal-article",
    limit=20
)

# Explore community discussions
community_results = mcp2_search(
    query="ML model evaluation common mistakes",
    source="reddit",
    limit=10
)
```

#### Deep Dive Research
For each relevant result, retrieve detailed content:

```python
# Get specific document content
document = mcp2_get_document(
    document_uri=results[0]['uri'],  # From search results
    query="evaluation metrics and when to use them",
    mode="wide"  # Get comprehensive coverage
)
```

#### Library Documentation
If skill involves specific libraries:

```python
# Resolve library
library = mcp0_resolve_library_id(libraryName="scikit-learn")

# Get code-focused documentation
docs = mcp0_get_library_docs(
    context7CompatibleLibraryID="/scikit-learn/scikit-learn",
    topic="model evaluation",
    mode="code",
    page=1
)
```

### Step 3: Information Synthesis

Organize research findings:

1. **Core Concepts**: Fundamental principles and definitions
2. **Best Practices**: Expert-recommended approaches
3. **Common Patterns**: Recurring techniques and methods
4. **Pitfalls**: Mistakes to avoid
5. **Examples**: Real-world applications
6. **Citations**: Source references for credibility

Create a research summary document:

```markdown
# Research Summary: ML Model Evaluation

## Core Concepts
- Accuracy vs Precision/Recall tradeoff
- Cross-validation importance
- [Source: Paper XYZ, doi://...]

## Best Practices
1. Always use cross-validation (5-10 folds)
   - Source: "Best Practices in ML" (ArXiv:...)
2. Match metric to business objective
   - Source: Industry Survey (Reddit: r/MachineLearning)

## Common Pitfalls
- Data leakage during evaluation
- Using accuracy for imbalanced datasets
- [Multiple sources confirm this]

## Code Examples
[Gather working examples from documentation]
```

### Step 4: AI-Powered Skill Generation

Use Claude to generate the skill based on research:

```plaintext
Prompt Example:
"Based on the following research summary, create an Agent Skill for ML model 
evaluation. The skill should:
- Follow the SKILL.md format with YAML frontmatter
- Include clear instructions for choosing appropriate metrics
- Provide code examples in Python (scikit-learn)
- List common pitfalls and how to avoid them
- Include citations to source materials
- Use progressive disclosure (reference additional files as needed)

Research Summary:
[Paste your synthesized research]
"
```

### Step 5: Structure and Organize

Organize generated content:

```
ml-model-evaluator/
├── SKILL.md              # Main skill file
├── metrics-reference.md  # Detailed metrics guide
├── examples.md           # Extended examples
└── resources/
    └── evaluation.py     # Helper scripts
```

**SKILL.md** should contain:
- YAML frontmatter (name, description)
- Overview and when to use
- Core instructions
- Basic examples
- References to additional files

**metrics-reference.md** (referenced from SKILL.md):
- Comprehensive metric definitions
- Mathematical formulas
- When to use each metric
- Citations to papers

### Step 6: Validation and Testing

Test the generated skill:

1. **Accuracy Check**
   - Verify code examples run correctly
   - Cross-reference facts with source documents
   - Ensure citations are accurate

2. **Usability Test**
   - Use the skill for real tasks
   - Check if instructions are clear
   - Verify examples are helpful

3. **AI Testing**
   - Load skill into Claude
   - Test with various prompts
   - Ensure Claude uses skill appropriately

4. **Peer Review**
   - Have domain experts review
   - Check for technical accuracy
   - Validate best practices

### Step 7: Iteration

Based on testing feedback:

- Fix inaccuracies
- Clarify confusing sections
- Add missing examples
- Update citations
- Improve organization

## Space Frontiers Integration

### Search Strategies

#### Comprehensive Search
Cast a wide net initially:

```python
# Search multiple sources
sources = ["journal-article", "wiki", "reddit"]
all_results = []

for source in sources:
    results = mcp2_search(
        query="your topic",
        source=source,
        limit=20
    )
    all_results.extend(results)
```

#### Focused Search
Target specific knowledge:

```python
# Academic research only
papers = mcp2_search(
    query="neural network optimization techniques",
    source="arxiv",
    limit=10
)

# Practical community knowledge
discussions = mcp2_search(
    query="neural network training tips",
    source="reddit",
    limit=10
)
```

#### Iterative Refinement
Start broad, narrow down:

```python
# Round 1: Broad search
initial = mcp2_search(query="data science workflows", limit=20)

# Round 2: Based on findings, search more specifically
refined = mcp2_search(query="data cleaning best practices pandas", limit=10)

# Round 3: Get specific documents
for result in refined[:5]:
    doc = mcp2_get_document(
        document_uri=result['uri'],
        query="data cleaning techniques",
        mode="focused"
    )
```

### Document Retrieval Patterns

#### Quick Overview Pattern
When you need just the gist:

```python
# Get metadata only (fast)
metadata = mcp2_get_document_metadata(document_uri=uri)

# Use focused mode with specific query
doc = mcp2_get_document(
    document_uri=uri,
    query="specific aspect only",
    mode="focused"  # Returns 5 snippets
)
```

#### Comprehensive Study Pattern
When you need deep understanding:

```python
# Use wide mode for extensive content
doc = mcp2_get_document(
    document_uri=uri,
    query="topic",
    mode="wide"  # Returns 20 snippets
)

# Extract all relevant sections
sections = doc['content'].split('\n\n')
```

#### Multi-Source Verification Pattern
Cross-reference information:

```python
# Search same topic across sources
academic = mcp2_search(query="topic", source="arxiv", limit=5)
community = mcp2_search(query="topic", source="reddit", limit=5)
wiki = mcp2_search(query="topic", source="wiki", limit=3)

# Compare findings across sources
# Look for consensus and conflicts
```

## AI-Powered Generation

### Effective Prompting

When asking Claude to generate skills:

```plaintext
System: You are an expert skill creator for Claude Agent Skills.

Task: Create a comprehensive skill for [TOPIC]

Context:
- Target users: [WHO]
- Use cases: [SCENARIOS]
- Required knowledge level: [BEGINNER/INTERMEDIATE/ADVANCED]

Research provided:
[PASTE RESEARCH SUMMARY]

Requirements:
1. Follow SKILL.md format with YAML frontmatter
2. Include practical examples with code
3. Cite sources from research
4. Use progressive disclosure
5. Add troubleshooting section
6. Include security considerations if applicable

Output format: Complete SKILL.md file
```

### Generation Best Practices

- **Provide Context**: Give Claude all research findings
- **Be Specific**: Clear requirements yield better results
- **Iterate**: Generate, review, refine, repeat
- **Validate**: Always verify AI-generated content
- **Cite Sources**: Include references from research
- **Test Examples**: Ensure code actually works

### Content Quality Checks

Before finalizing AI-generated content:

```markdown
Checklist:
- [ ] YAML frontmatter is valid
- [ ] Description clearly states when to use skill
- [ ] Instructions are step-by-step and actionable
- [ ] Examples use real, working code
- [ ] Citations reference actual sources
- [ ] Security considerations addressed
- [ ] Error handling included
- [ ] Edge cases covered
- [ ] Progressive disclosure used appropriately
- [ ] Formatting is consistent
```

## Quality Assurance

### Research Quality

- **Source Diversity**: Multiple types of sources
- **Source Authority**: Academic papers, official docs, expert discussions
- **Currency**: Recent publications and discussions
- **Consensus**: Multiple sources agree on key points

### Content Quality

- **Accuracy**: Facts verified against sources
- **Completeness**: All important aspects covered
- **Clarity**: Easy to understand and follow
- **Practicality**: Examples work in real scenarios
- **Organization**: Logical flow and structure

### Technical Quality

- **Code Works**: All examples execute successfully
- **Best Practices**: Follows current standards
- **Security**: No vulnerabilities introduced
- **Performance**: Efficient approaches recommended
- **Compatibility**: Works with specified versions

## Examples

### Example 1: Creating a Python Testing Skill

```python
# Step 1: Research
search_results = mcp2_search(
    query="python testing best practices pytest",
    source="journal-article",
    limit=15
)

# Step 2: Get documentation
pytest_docs = mcp0_resolve_library_id(libraryName="pytest")
pytest_guide = mcp0_get_library_docs(
    context7CompatibleLibraryID="/pytest-dev/pytest",
    topic="testing best practices",
    mode="code",
    page=1
)

# Step 3: Get community insights
reddit_discussions = mcp2_search(
    query="pytest tips and tricks",
    source="reddit",
    limit=10
)

# Step 4: Synthesize research
# [Create research summary document]

# Step 5: Generate skill with Claude
# [Prompt Claude with research summary]

# Step 6: Validate examples
# [Test all code examples]
```

### Example 2: Creating a Data Visualization Skill

```python
# Research academic best practices
viz_papers = mcp2_search(
    query="data visualization best practices effectiveness",
    source="journal-article",
    limit=10
)

# Get library documentation
matplotlib_docs = mcp0_get_library_docs(
    context7CompatibleLibraryID="/matplotlib/matplotlib",
    topic="visualization guidelines",
    mode="code"
)

# Community knowledge on common mistakes
mistakes = mcp2_search(
    query="data visualization mistakes to avoid",
    source="reddit",
    limit=10
)

# Detailed document retrieval
for paper in viz_papers[:3]:
    content = mcp2_get_document(
        document_uri=paper['uri'],
        query="visualization design principles",
        mode="wide"
    )
    # [Process and extract key principles]

# Generate comprehensive skill with examples
# [Use Claude to synthesize all research into SKILL.md]
```

### Example 3: Multi-Stage Research

```python
# Stage 1: Broad understanding
overview = mcp2_search(
    query="microservices architecture patterns",
    source="wiki",
    limit=5
)

# Stage 2: Academic foundation
research = mcp2_search(
    query="microservices design patterns research",
    source="journal-article",
    limit=15
)

# Stage 3: Practical implementation
community = mcp2_search(
    query="microservices implementation experience",
    source="reddit",
    limit=15
)

# Stage 4: Deep dive on specific patterns
for result in research[:5]:
    detailed = mcp2_get_document(
        document_uri=result['uri'],
        query="service decomposition and communication patterns",
        mode="wide"
    )
    # [Extract patterns and best practices]

# Stage 5: Generate skill with progressive disclosure
# Main SKILL.md: Overview and when to use
# patterns.md: Detailed pattern catalog
# examples.md: Real-world implementations
# antipatterns.md: Common mistakes
```

## Tips and Best Practices

### Research Tips

1. **Start Broad**: Begin with general searches to understand scope
2. **Go Deep**: Retrieve full documents for key papers
3. **Cross-Reference**: Verify information across multiple sources
4. **Update Regularly**: Re-search periodically for new findings
5. **Document Sources**: Keep track of URIs and citations

### Generation Tips

1. **Quality Over Speed**: Take time to research thoroughly
2. **Validate Everything**: Test all code and verify all claims
3. **Think Progressive**: Structure for progressive disclosure
4. **Include Edge Cases**: Don't just show happy path
5. **Make it Practical**: Focus on actionable guidance

### Collaboration Tips

1. **Share Research**: Document findings for other contributors
2. **Peer Review**: Have others validate technical accuracy
3. **Iterate Publicly**: Use issues and PRs for discussion
4. **Credit Sources**: Always cite where information came from
5. **Stay Humble**: Be open to corrections and improvements

## Troubleshooting

### Common Issues

**Issue**: Search returns too many irrelevant results
**Solution**: Be more specific in query, filter by source type

**Issue**: Document retrieval is slow
**Solution**: Use `mcp2_get_document_metadata()` first, retrieve full content only when needed

**Issue**: Code examples don't work
**Solution**: Test with code execution, get latest docs from Context7

**Issue**: Information conflicts across sources
**Solution**: Prioritize academic papers and official docs, note conflicts in skill

**Issue**: Generated content is too generic
**Solution**: Include more specific examples from research, add real-world scenarios

## Resources

- [Agent Skills Specification](./spec/README.md)
- [Contributing Guidelines](./CONTRIBUTING.md)
- [Space Frontiers Documentation](https://github.com/space-frontiers/mcp)
- [Context7 Documentation](https://context7.com)
- [Anthropic Skills Documentation](https://anthropic.com/skills)

---

**Remember**: Great skills are built on great research. Take the time to gather authoritative information before generating content.
