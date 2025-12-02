# Agent Skills Repository

Public repository for Agent Skills that extend Claude's capabilities to handle specialized tasks.

## What are Skills?

Skills are folders of instructions, scripts, and resources that Claude loads dynamically to improve performance on specialized tasks. Skills teach Claude how to complete specific tasks in a repeatable way, whether that's creating documents with your company's brand guidelines, analyzing data using your organization's specific workflows, or automating personal tasks.

## About This Repository

This repository contains skills that demonstrate what's possible with Claude's skills system. These skills range from creative applications (art, music, design) to technical tasks (testing web apps, system automation) to enterprise workflows (communications, branding, etc.).

Each skill is self-contained in its own folder with a `SKILL.md` file containing the instructions and metadata that Claude uses. Browse through these skills to get inspiration for your own skills or to understand different patterns and approaches.

## Repository Structure

- **[./skills](./skills)**: Skill examples organized by category:
  - Creative & Design
  - Development & Technical
  - Enterprise & Communication
  - Personal Productivity
- **[./spec](./spec)**: The Agent Skills specification
- **[./template](./template)**: Skill template to get started quickly
- **[QUICKSTART.md](./QUICKSTART.md)**: 5-minute quick start guide
- **[DEVELOPMENT.md](./DEVELOPMENT.md)**: Complete development guide with AI and research tools

## Getting Started

**→ New here? Start with [QUICKSTART.md](./QUICKSTART.md)**

### Using Skills

To use any skill from this repository:

1. Download or clone the skill folder you're interested in
2. Upload it to Claude via Claude.ai, Claude Code, or the API
3. Claude will automatically load and use the skill when relevant

For detailed instructions, see the [Skills Documentation](https://support.claude.com/en/articles/12512180-using-skills-in-claude).

### Creating Your Own Skill

Skills are simple to create - just a folder with a `SKILL.md` file containing YAML frontmatter and instructions. You can use the [template](./template) in this repository as a starting point:

```markdown
---
name: my-skill-name
description: A clear description of what this skill does and when to use it
---

# My Skill Name

[Add your instructions here that Claude will follow when this skill is active]

## Examples
- Example usage 1
- Example usage 2

## Guidelines
- Guideline 1
- Guideline 2
```

The frontmatter requires only two fields:
- `name` - A unique identifier for your skill (lowercase, hyphens for spaces)
- `description` - A complete description of what the skill does and when to use it

The markdown content below contains the instructions, examples, and guidelines that Claude will follow.

For more details, see [How to create custom skills](https://support.claude.com/en/articles/12512198-creating-custom-skills).

## Key Concepts

### Progressive Disclosure

Skills use progressive disclosure to manage context efficiently:
1. **Level 1**: The `name` and `description` in YAML frontmatter tell Claude when to use the skill
2. **Level 2**: The full `SKILL.md` content provides detailed instructions
3. **Level 3+**: Additional referenced files (e.g., `reference.md`, `examples.md`) that Claude loads only when needed

This approach allows skills to contain extensive documentation without overwhelming the context window.

### Skill Structure

A typical skill directory contains:
```
my-skill/
├── SKILL.md          # Required: Main skill file with frontmatter and instructions
├── reference.md      # Optional: Additional reference material
├── examples.md       # Optional: Detailed examples
└── resources/        # Optional: Supporting files (scripts, data, etc.)
```

## AI-Powered Skill Development

This repository leverages AI to generate and enhance skills efficiently. When creating new skills, follow this research-driven approach:

### Using Space Frontiers for Research

Before building skills, gather authoritative information using **Space Frontiers MCP** (Model Context Protocol server):

```bash
# Example workflow for creating a new skill:
# 1. Search for relevant documentation and research
mcp2_search(query="topic or library name", limit=20)

# 2. Retrieve specific documents for detailed information
mcp2_get_document(document_uri="doi://...", query="specific aspect")

# 3. Get document metadata for citations
mcp2_get_document_metadata(document_uri="doi://...")
```

#### Space Frontiers Sources

Space Frontiers provides access to:
- **Academic Papers**: PubMed, ArXiv, BioRxiv, MedRxiv
- **Documentation**: Wikipedia, technical standards
- **Community Knowledge**: Reddit discussions, Telegram posts, YouTube transcripts

#### Skill Development Workflow

1. **Research Phase**
   - Use `mcp2_search()` to find relevant papers, documentation, and discussions
   - Retrieve detailed content with `mcp2_get_document()` for specific topics
   - Gather multiple perspectives from different sources

2. **Content Synthesis**
   - Synthesize information from retrieved documents
   - Extract best practices, common patterns, and expert recommendations
   - Identify authoritative sources for citations in skill instructions

3. **Skill Generation**
   - Use AI (Claude) to generate skill content based on research
   - Include citations and references to source material
   - Ensure instructions are based on verified, up-to-date information

4. **Validation**
   - Test generated skills with real use cases
   - Verify technical accuracy against source documents
   - Iterate based on performance

### Example: Creating a Data Science Skill

```python
# Step 1: Research best practices
search_results = mcp2_search(
    query="pandas data analysis best practices",
    source="journal-article",
    limit=10
)

# Step 2: Get detailed documentation
document = mcp2_get_document(
    document_uri="library://pandas/docs",
    query="data cleaning and preprocessing techniques",
    mode="code"
)

# Step 3: Generate skill using researched content
# Claude synthesizes information into SKILL.md format
# Including code examples, best practices, and citations
```

### Best Practices for AI-Generated Skills

- **Research First**: Always gather authoritative sources before generating content
- **Cite Sources**: Include references to papers, documentation, or expert articles
- **Verify Accuracy**: Cross-reference AI-generated content with source materials
- **Test Thoroughly**: Validate that examples and code snippets actually work
- **Iterate**: Refine skills based on real-world usage and feedback
- **Stay Current**: Use Space Frontiers to access latest documentation and research

### Tools for Skill Development

- **Space Frontiers MCP**: Document retrieval and research
- **Context7 MCP**: Up-to-date library documentation (use `mcp0_resolve-library-id` then `mcp0_get-library-docs`)
- **Claude**: AI-powered content generation and synthesis
- **Code Execution**: Test and validate skill examples

This approach ensures skills are built on solid research foundations while leveraging AI for efficient content generation.

**→ For detailed instructions, see [DEVELOPMENT.md](./DEVELOPMENT.md)**

## Learn More

- [What are skills?](https://support.claude.com/en/articles/12512176-what-are-skills)
- [Using skills in Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude)
- [Creating custom skills](https://support.claude.com/en/articles/12512198-creating-custom-skills)
- [Equipping agents for the real world with Agent Skills](https://anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)

## Contributing

We welcome contributions! Please feel free to submit pull requests with new skills or improvements to existing ones.

## License

This repository contains both open source (Apache 2.0) and source-available skills. Check individual skill folders for specific license information.

## Disclaimer

These skills are provided as examples and starting points. Review and test all skills before using them in production environments. Skills may execute code or access external resources, so always verify their contents and behavior meet your security requirements.
