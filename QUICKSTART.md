# Quick Start Guide

Get started with Agent Skills development in 5 minutes.

## For Users: Using Existing Skills

### Step 1: Browse Skills
```bash
cd skills/
ls -R
```

Explore skills by category:
- `creative-design/` - Content writing, design
- `development-technical/` - Code review, API docs
- `enterprise-communication/` - Email composing
- `personal-productivity/` - Research summaries

### Step 2: Choose a Skill
```bash
# Example: Copy the code reviewer skill
cp -r skills/development-technical/code-reviewer ~/my-skills/
```

### Step 3: Use with Claude
- Upload the skill folder to Claude.ai
- Or use with Claude Code
- Or integrate via API

Claude will automatically load and use the skill when relevant!

## For Developers: Creating New Skills

### Method 1: Manual Creation (5 minutes)

1. **Copy Template**
   ```bash
   cp -r template/ skills/my-category/my-new-skill/
   cd skills/my-category/my-new-skill/
   ```

2. **Edit SKILL.md**
   - Update YAML frontmatter (name, description)
   - Write instructions
   - Add examples
   - Include guidelines

3. **Test**
   - Upload to Claude
   - Test with real scenarios
   - Iterate

### Method 2: AI-Powered Creation (Recommended)

1. **Research Your Topic**
   ```python
   # Search for information
   mcp2_search(
       query="your topic best practices",
       source="journal-article",
       limit=20
   )
   
   # Get detailed content
   mcp2_get_document(
       document_uri="doi://...",
       query="specific aspect"
   )
   ```

2. **Generate with Claude**
   ```plaintext
   "Based on the following research, create an Agent Skill for [TOPIC]:
   
   [Paste research summary]
   
   Requirements:
   - Follow SKILL.md format
   - Include code examples
   - Add best practices
   - Cite sources"
   ```

3. **Validate and Test**
   - Verify code examples work
   - Test with Claude
   - Check citations

4. **Submit**
   ```bash
   git checkout -b feature/my-skill
   git add skills/my-category/my-new-skill/
   git commit -m "Add my-new-skill"
   git push origin feature/my-skill
   # Create Pull Request
   ```

## Essential Files

- **[README.md](./README.md)** - Repository overview
- **[DEVELOPMENT.md](./DEVELOPMENT.md)** - Complete development guide with Space Frontiers
- **[CONTRIBUTING.md](./CONTRIBUTING.md)** - Contribution guidelines
- **[spec/README.md](./spec/README.md)** - Technical specification
- **[template/SKILL.md](./template/SKILL.md)** - Skill template

## Minimal SKILL.md Example

```markdown
---
name: my-skill
description: What this skill does and when to use it
---

# My Skill

Instructions for Claude to follow...

## Examples
- Example 1
- Example 2

## Guidelines
- Guideline 1
- Guideline 2
```

That's it! Just 2 required fields in YAML + markdown instructions.

## Common Commands

```bash
# List all skills
find skills/ -name "SKILL.md"

# Validate YAML frontmatter
head -n 10 skills/*/SKILL.md

# Search for skills by topic
grep -r "description:" skills/*/SKILL.md

# Count total skills
find skills/ -name "SKILL.md" | wc -l
```

## Research Tools Quick Reference

### Space Frontiers MCP

```python
# Search
mcp2_search(query="topic", source="journal-article", limit=20)

# Get document
mcp2_get_document(document_uri="doi://...", query="aspect", mode="focused")

# Get metadata
mcp2_get_document_metadata(document_uri="doi://...")
```

Sources: `journal-article`, `wiki`, `pubmed`, `arxiv`, `reddit`, `youtube`

### Context7 MCP

```python
# Resolve library
mcp0_resolve_library_id(libraryName="library-name")

# Get docs
mcp0_get_library_docs(
    context7CompatibleLibraryID="/org/project",
    topic="topic",
    mode="code"
)
```

## Examples

### Example 1: Using Code Reviewer Skill

1. Upload `skills/development-technical/code-reviewer/` to Claude
2. Show Claude your code
3. Ask: "Please review this code"
4. Claude uses the skill automatically!

### Example 2: Creating a New Skill

```bash
# Research
mcp2_search(query="python async programming best practices", limit=15)

# Generate with Claude
"Create a skill for async Python programming based on [research]"

# Test
# Upload to Claude and try: "Help me write async code"

# Submit
git add skills/development-technical/async-python/
git commit -m "Add async Python skill"
```

## Getting Help

- **Questions?** Open an [issue](https://github.com/your-repo/issues)
- **Examples?** Browse [skills/](./skills/)
- **Specs?** Read [spec/README.md](./spec/README.md)
- **Development?** See [DEVELOPMENT.md](./DEVELOPMENT.md)

## Next Steps

1. ✅ Browse existing skills in `skills/` directory
2. ✅ Try using a skill with Claude
3. ✅ Read [DEVELOPMENT.md](./DEVELOPMENT.md) for AI-powered creation
4. ✅ Create your first skill using the template
5. ✅ Share your skill via Pull Request

**Happy skill building!** 🚀
