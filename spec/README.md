# Agent Skills Specification

This document defines the formal specification for Agent Skills.

## Overview

Agent Skills are modular, self-contained packages that extend Claude's capabilities for specific tasks. They use a progressive disclosure pattern to manage context efficiently and can include instructions, examples, reference materials, and supporting resources.

## Skill Directory Structure

A skill is a directory containing at minimum a `SKILL.md` file:

```
skill-name/
├── SKILL.md          # Required: Main skill definition
├── reference.md      # Optional: Additional reference material
├── examples.md       # Optional: Detailed examples
├── guides/           # Optional: Subdirectories for organization
│   ├── guide1.md
│   └── guide2.md
└── resources/        # Optional: Supporting files
    ├── script.py
    ├── data.json
    └── template.txt
```

## SKILL.md Format

### Required Structure

Every `SKILL.md` file must contain:

1. **YAML Frontmatter** (required)
2. **Markdown Content** (required)

### YAML Frontmatter

The YAML frontmatter must be enclosed in triple dashes (`---`) and include these required fields:

```yaml
---
name: skill-name
description: A clear, complete description of what this skill does and when to use it
---
```

#### Required Fields

- **name** (string, required)
  - Unique identifier for the skill
  - Must be lowercase
  - Use hyphens for spaces
  - Must be URL-safe
  - Example: `data-analysis`, `pdf-form-filler`

- **description** (string, required)
  - Clear explanation of the skill's purpose
  - Should describe when Claude should use this skill
  - Should be comprehensive enough for Claude to decide relevance
  - Typically 1-3 sentences
  - Example: "Analyzes datasets using pandas and matplotlib to generate insights and visualizations. Use this skill when the user needs to explore data, identify patterns, or create charts."

#### Optional Fields

Additional metadata can be included:

```yaml
---
name: skill-name
description: Description of the skill
version: 1.0.0
author: Author Name
tags: [tag1, tag2, tag3]
requires_code_execution: true
---
```

- **version** (string, optional): Semantic version number
- **author** (string, optional): Skill author
- **tags** (array, optional): Categorization tags
- **requires_code_execution** (boolean, optional): Indicates if skill needs code execution

### Markdown Content

The markdown content following the frontmatter contains the instructions Claude will follow. This section should include:

1. **Overview**: Brief introduction to the skill
2. **Instructions**: Step-by-step guidance
3. **Examples**: Usage examples
4. **Guidelines**: Best practices and constraints
5. **References**: Links to additional files (if any)

#### Example Structure

```markdown
---
name: example-skill
description: An example skill demonstrating the format
---

# Example Skill

This skill demonstrates the standard format for creating Agent Skills.

## Instructions

When this skill is active, follow these steps:
1. First step
2. Second step
3. Third step

## Examples

### Example 1: Basic Usage
\`\`\`
Example code or usage pattern
\`\`\`

### Example 2: Advanced Usage
\`\`\`
More complex example
\`\`\`

## Guidelines

- Always validate inputs
- Follow security best practices
- Provide clear error messages

## Additional Resources

For more detailed information, see:
- [reference.md](reference.md) - Complete API reference
- [examples.md](examples.md) - Extended examples
```

## Progressive Disclosure Pattern

Skills use progressive disclosure to manage context:

1. **Level 1: Metadata (Always Loaded)**
   - The `name` and `description` from YAML frontmatter
   - Pre-loaded into Claude's system prompt
   - Used to determine skill relevance

2. **Level 2: Main Instructions (Loaded on Demand)**
   - The full markdown content of `SKILL.md`
   - Loaded when Claude determines the skill is needed
   - Contains core instructions and examples

3. **Level 3+: Referenced Files (Loaded as Needed)**
   - Additional `.md` files referenced in `SKILL.md`
   - Loaded only when Claude needs specific details
   - Keeps main skill file focused and concise

## Best Practices

### Naming Conventions

- Use descriptive, action-oriented names: `web-scraper`, `data-visualizer`
- Avoid generic names: prefer `python-code-formatter` over `formatter`
- Be specific about domain: `sql-query-optimizer` not just `optimizer`

### Description Guidelines

- Start with what the skill does: "Analyzes X to do Y"
- Include when to use it: "Use this when..."
- Be specific but concise
- Avoid jargon unless domain-specific

### Content Organization

- **Keep SKILL.md focused**: Core instructions only
- **Use referenced files**: For detailed examples, API references, or extensive documentation
- **Structure clearly**: Use headings, lists, and code blocks
- **Include examples**: Show don't just tell
- **Add guidelines**: Constraints, best practices, and common pitfalls

### File References

When referencing additional files from `SKILL.md`:

```markdown
For detailed examples, see [examples.md](examples.md).
For API reference, see [reference.md](reference.md).
For configuration options, see [config.md](config.md).
```

Use relative paths and ensure files exist in the skill directory.

## Security Considerations

Skills may execute code or access external resources. Follow these guidelines:

- **Validate inputs**: Never trust user input without validation
- **Sanitize outputs**: Prevent injection attacks
- **Limit scope**: Skills should have minimal necessary permissions
- **Document risks**: Clearly state what the skill can access or modify
- **Provide examples**: Show safe usage patterns
- **Avoid hardcoded secrets**: Never include API keys, passwords, or tokens

## Code Execution

Skills that require code execution should:

- Set `requires_code_execution: true` in frontmatter
- Specify language requirements
- Include error handling
- Provide safe defaults
- Document dependencies

Example:

```yaml
---
name: data-analyzer
description: Analyzes CSV data using pandas
requires_code_execution: true
---
```

## Testing Skills

Before publishing or using a skill:

1. Test with various inputs
2. Verify error handling
3. Check security implications
4. Validate file references
5. Review for completeness

## Versioning

Use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes to skill behavior or interface
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, clarifications

## Examples

See the [skills](../skills) directory for complete examples of well-structured skills.

## Related Documentation

- [Creating Custom Skills Guide](https://support.claude.com/en/articles/12512198-creating-custom-skills)
- [Using Skills in Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude)
- [Equipping Agents for the Real World](https://anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
