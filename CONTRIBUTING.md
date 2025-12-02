# Contributing to Agent Skills Repository

Thank you for your interest in contributing to the Agent Skills Repository! This document provides guidelines for contributing new skills and improvements.

## How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs or suggest improvements
- Provide clear descriptions and examples
- Include relevant context about your use case

### Submitting Skills

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/ai-overmind.git
   cd ai-overmind
   ```

2. **Create a New Branch**
   ```bash
   git checkout -b feature/my-new-skill
   ```

3. **Add Your Skill**
   - Choose the appropriate category in `skills/` directory
   - Create a new folder with your skill name (lowercase, hyphens)
   - Follow the [skill specification](./spec/README.md)
   - Use the [template](./template/SKILL.md) as starting point
   - **Recommended**: Use [AI-powered generation workflow](./DEVELOPMENT.md) with Space Frontiers research

4. **Test Your Skill**
   - Verify YAML frontmatter is valid
   - Test with Claude to ensure it works as expected
   - Check all file references are correct
   - Ensure examples are clear and working

5. **Submit Pull Request**
   - Provide clear description of the skill
   - Explain use cases and benefits
   - Include any testing notes
   - Reference related issues if applicable

## AI-Powered Skill Generation

We strongly encourage using AI and research tools to create high-quality, well-researched skills:

### Research-Driven Approach

1. **Use Space Frontiers MCP** to gather authoritative information:
   - Search academic papers, documentation, and community discussions
   - Retrieve detailed content from relevant sources
   - Cross-reference information across multiple sources

2. **Use Context7 MCP** for up-to-date library documentation:
   - Get latest API references and examples
   - Access code-focused documentation
   - Find conceptual guides and best practices

3. **Synthesize with AI**:
   - Let Claude generate skill content based on research
   - Include citations to source materials
   - Validate all AI-generated content against sources

### Recommended Workflow

```bash
# 1. Research your topic
mcp2_search(query="your topic", limit=20)

# 2. Retrieve relevant documents
mcp2_get_document(document_uri="...", query="specific aspect")

# 3. Generate skill with Claude using research
# 4. Validate and test generated content
# 5. Submit with research citations
```

**For complete instructions, see [DEVELOPMENT.md](./DEVELOPMENT.md)**

## Skill Quality Guidelines

### Required Elements
- [ ] Valid YAML frontmatter with `name` and `description`
- [ ] Clear, comprehensive instructions
- [ ] At least 2-3 practical examples
- [ ] Guidelines section with best practices
- [ ] Proper formatting and structure

### Best Practices
- [ ] Clear, action-oriented skill name
- [ ] Description that helps Claude know when to use it
- [ ] Well-organized content with appropriate headings
- [ ] Progressive disclosure (referenced files for detailed info)
- [ ] Security considerations documented if applicable
- [ ] No hardcoded secrets or sensitive information

### Style Guide
- Use Markdown formatting consistently
- Code blocks should specify language
- Lists should be properly formatted
- Headers should be hierarchical (H1 → H2 → H3)
- File references should use relative paths

## Skill Categories

Add your skill to the appropriate category:

- **creative-design/**: Art, design, content creation
- **development-technical/**: Coding, testing, documentation
- **enterprise-communication/**: Business communication, branding
- **personal-productivity/**: Organization, planning, efficiency

If your skill doesn't fit existing categories, propose a new one in your PR.

## Testing Checklist

Before submitting, verify:

- [ ] YAML frontmatter is valid
- [ ] All file references work correctly
- [ ] Examples are accurate and clear
- [ ] Skill works as intended when used with Claude
- [ ] No security issues or hardcoded secrets
- [ ] Documentation is clear and complete
- [ ] Follows repository structure and style

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on the skills and ideas, not the people
- Give credit where credit is due
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

- Open an issue for questions about contributing
- Check existing skills for examples and patterns
- Review the [specification](./spec/README.md) for technical details

## Skill Review Process

1. **Initial Review**: Maintainers check basic requirements
2. **Testing**: Skill is tested with Claude
3. **Feedback**: Suggestions for improvements if needed
4. **Approval**: Skill is merged when ready
5. **Documentation**: Added to skills catalog

## Recognition

Contributors will be:
- Listed in skill metadata as author
- Mentioned in release notes
- Added to contributors list

Thank you for helping make Claude more capable!
