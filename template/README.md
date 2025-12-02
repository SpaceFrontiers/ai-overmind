# Skill Template

This directory contains a template for creating new Agent Skills.

## Quick Start

1. Copy this entire `template` directory
2. Rename it to your skill name (lowercase, hyphens for spaces)
3. Edit `SKILL.md` with your skill's content
4. Update the YAML frontmatter with your skill's metadata
5. Add any additional files as needed

## Minimal Example

For a minimal skill, you only need `SKILL.md` with:

```markdown
---
name: my-skill
description: What this skill does and when to use it
---

# My Skill

Instructions for Claude to follow...
```

## Extended Example

For more complex skills, add additional files:

```
my-skill/
├── SKILL.md           # Main skill file (required)
├── reference.md       # Detailed reference (optional)
├── examples.md        # Extended examples (optional)
└── resources/         # Supporting files (optional)
    └── helper.py
```

Reference additional files from `SKILL.md`:

```markdown
For more examples, see [examples.md](examples.md).
```

## Best Practices

- **Be specific**: Clear, focused skills work better than general ones
- **Use examples**: Show don't just tell
- **Keep it modular**: Break complex skills into referenced files
- **Test thoroughly**: Verify the skill works as expected
- **Document well**: Include guidelines and common pitfalls

## Need Help?

- See [spec/README.md](../spec/README.md) for the complete specification
- Browse [skills/](../skills/) for working examples
- Read the [main README](../README.md) for more information
