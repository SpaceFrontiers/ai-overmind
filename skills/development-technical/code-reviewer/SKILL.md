---
name: code-reviewer
description: Reviews code for quality, best practices, security issues, and potential bugs. Use this skill when the user asks to review code, check code quality, or identify improvements in their codebase.
version: 1.0.0
tags: [development, code-quality, review, security]
---

# Code Reviewer

This skill helps you perform thorough code reviews, identifying quality issues, security concerns, and opportunities for improvement.

## Instructions

When reviewing code, follow this systematic approach:

1. **Understand Context**: Read the code to understand its purpose and structure
2. **Check Functionality**: Verify the code does what it's supposed to do
3. **Review Quality**: Assess code organization, naming, and clarity
4. **Security Audit**: Look for security vulnerabilities
5. **Performance Check**: Identify potential performance issues
6. **Best Practices**: Verify adherence to language-specific conventions
7. **Provide Feedback**: Structure feedback as constructive suggestions

## Review Checklist

### Functionality
- [ ] Code achieves its intended purpose
- [ ] Edge cases are handled
- [ ] Error conditions are properly managed
- [ ] Logic is correct and complete

### Code Quality
- [ ] Clear and descriptive naming
- [ ] Appropriate comments and documentation
- [ ] DRY principle (Don't Repeat Yourself)
- [ ] Single Responsibility Principle
- [ ] Appropriate abstraction levels
- [ ] Readable and maintainable structure

### Security
- [ ] Input validation and sanitization
- [ ] No hardcoded secrets or credentials
- [ ] Proper authentication and authorization
- [ ] SQL injection prevention
- [ ] XSS prevention
- [ ] CSRF protection where applicable

### Performance
- [ ] Efficient algorithms and data structures
- [ ] No unnecessary loops or operations
- [ ] Appropriate use of caching
- [ ] Database queries optimized
- [ ] Resource cleanup (files, connections, etc.)

### Testing
- [ ] Testable code structure
- [ ] Appropriate test coverage
- [ ] Tests are meaningful and clear

## Feedback Structure

Organize your review feedback in this format:

### Summary
Brief overview of the code and overall assessment.

### Strengths
Highlight what's done well:
- Good practice 1
- Good practice 2

### Issues Found

#### Critical 🔴
Issues that must be fixed (security, bugs, data loss):
- Issue description with line reference
- Why it's critical
- Suggested fix

#### Important 🟡
Significant improvements needed:
- Issue description with line reference
- Impact if not fixed
- Suggested fix

#### Minor 🟢
Nice-to-have improvements:
- Suggestion with line reference
- Benefit of implementing
- Optional: suggested fix

### Recommendations
- Overall suggestions for improvement
- Architectural recommendations
- Best practice reminders

## Examples

### Example 1: Python Function Review

```python
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
```

**Review**:
- 🔴 **Critical Security Issue**: SQL injection vulnerability. User input is directly interpolated into query.
- **Fix**: Use parameterized queries: `db.execute("SELECT * FROM users WHERE id = ?", (user_id,))`

### Example 2: JavaScript Code Review

```javascript
function calculateTotal(items) {
  let total = 0;
  for(let i = 0; i < items.length; i++) {
    total += items[i].price * items[i].quantity;
  }
  return total;
}
```

**Review**:
- ✅ **Strength**: Clear purpose and correct logic
- 🟢 **Minor**: Consider using `reduce()` for more idiomatic JavaScript
- 🟡 **Important**: No validation of items array or numeric values

## Guidelines

### Be Constructive
- Focus on the code, not the person
- Explain the "why" behind suggestions
- Provide examples of better approaches
- Acknowledge good practices

### Prioritize Issues
- Security and bugs first
- Then performance and maintainability
- Finally style and minor improvements

### Consider Context
- Project requirements and constraints
- Team conventions and standards
- Performance vs. readability tradeoffs
- Development stage (prototype vs. production)

### Provide Actionable Feedback
- Be specific about what to change
- Show example code when helpful
- Reference documentation or best practices
- Suggest concrete next steps

## Language-Specific Considerations

### Python
- PEP 8 style guide
- Type hints for clarity
- Context managers for resources
- List comprehensions vs. loops

### JavaScript/TypeScript
- ESLint rules
- Modern ES6+ syntax
- Async/await over callbacks
- TypeScript types and interfaces

### Java
- SOLID principles
- Exception handling
- Resource management (try-with-resources)
- Coding conventions

### Go
- Effective Go guidelines
- Error handling patterns
- Interface usage
- Goroutine and channel safety

## Common Pitfalls to Watch For

- **Magic numbers**: Use named constants
- **Deep nesting**: Refactor for clarity
- **Global state**: Minimize side effects
- **Long functions**: Break into smaller pieces
- **Unclear naming**: Use descriptive names
- **Missing error handling**: Always handle errors
- **Premature optimization**: Clarity first
- **Ignoring edge cases**: Test boundaries

## Research-Backed Insights

### Effectiveness of Static Analysis Tools
Recent empirical research (Charoenwet et al., 2024) on static application security testing (SAST) reveals:
- **52% detection rate**: A single SAST can produce warnings in vulnerable functions of 52% of real-world vulnerability-contributing commits
- **76% false positive challenge**: At least 76% of warnings in vulnerable functions are irrelevant to actual vulnerabilities
- **13% efficiency gain**: Prioritizing changed functions with SAST warnings reduces Initial False Alarm by 13%
- **22% gap remains**: 22% of vulnerabilities remain undetected due to SAST rule limitations

**Practical Takeaway**: Use SAST tools as assistants, not replacements. They improve efficiency but require human judgment to filter false positives and catch gaps.

### Code Security Audit Best Practices
Research on Python SAST tools (Ma et al., 2022) demonstrates:
- **Abstract Syntax Tree (AST) analysis** combined with data flow relationships improves vulnerability detection
- **Plug-in architecture** allows customization of security rules for specific contexts
- **Fast and efficient**: SAST doesn't require runtime environment configuration
- **Early detection**: Finding security issues at code layer allows lower-cost fixes

### Comprehensive Security Analysis
Survey research on security vulnerability detection (Chess & McGraw, 2004) identifies key vulnerability categories:
- **Access control vulnerabilities** (broken authentication, authorization)
- **Information flow vulnerabilities** (injection flaws, XSS, unvalidated input)
- **API conformance issues** (improper cryptographic usage, insecure storage)
- **Stack-based and role-based access control** require different analysis approaches

**Key Insight**: Security reviews must address multiple vulnerability types systematically - no single tool or technique catches everything.

## Additional Resources

For more detailed guidelines, see [review-guidelines.md](review-guidelines.md) for language-specific best practices.

## References

- Charoenwet, W., Thongtanunam, P., Pham, V-T., & Treude, C. (2024). "An Empirical Study of Static Analysis Tools for Secure Code Review". arXiv:2407.12241
- Ma, L., Yang, H., Xu, J., Yang, Z., Lao, Q., & Yuan, D. (2022). "Code Analysis with Static Application Security Testing for Python Program". Journal of Signal Processing Systems, 94(11), 1169-1182.
- Chess, B., & McGraw, G. (2004). "Static analysis for security". IEEE Security & Privacy.
