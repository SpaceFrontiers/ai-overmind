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

## The Guiding Principle

**Favor approving code that improves overall code health, even if not perfect.**

Based on Google's engineering practices, reviewers should approve a change once it definitively improves the system's overall code health, rather than seeking perfection. Balance forward progress against the importance of suggested changes. The goal is **continuous improvement**, not perfection.

- There is no "perfect" code, only better code
- Don't delay reviews for days/weeks over minor polish
- Prefix non-critical suggestions with "Nit:" to indicate they're optional
- Technical facts and data overrule opinions and personal preferences
- Accept author's approach if they can demonstrate it's equally valid

## Review Size Optimization

**Critical Finding**: Research shows review effectiveness drops significantly beyond 200-400 lines of code.

### Optimal Review Scope
- **Target**: 200-400 lines per review session
- **Maximum effectiveness**: First 200 lines (highest defect detection rate)
- **Sharp decline**: After 400 lines, reviewers miss significantly more issues
- **Cognitive load**: Larger changes overwhelm working memory capacity

### For Authors
- Break large changes into smaller, logical increments
- Submit multiple sequential pull requests rather than one massive change
- Use feature flags to deploy partially complete features
- Consider creating intermediate commits for easier review

### For Reviewers
- Request change breakdown if a PR exceeds 400 lines
- Take breaks between reviewing large changes
- Review in multiple sessions rather than forcing completion
- Focus first on critical paths, then supporting code

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

### Modern Code Review (Pull Request) Process
- [ ] Clear PR title and description
- [ ] References related issue/ticket
- [ ] All CI/CD checks passing (tests, linting, security scans)
- [ ] No merge conflicts
- [ ] Appropriate reviewers assigned based on expertise
- [ ] Changes are atomic and focused on single concern

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
- **Mentoring opportunity**: Share knowledge about languages, frameworks, or design principles
- Use "Nit:" prefix for educational but non-critical comments

### Prioritize Issues
- Security and bugs first (block merging if critical)
- Then performance and maintainability
- Finally style and minor improvements
- **Apply the guiding principle**: Does this change improve overall code health?

### Consider Context
- Project requirements and constraints
- Team conventions and standards
- Performance vs. readability tradeoffs
- Development stage (prototype vs. production)
- **Reviewer capacity**: Working memory limitations affect review quality
- **Change complexity**: Both conceptual and scope complexity matter

### Provide Actionable Feedback
- Be specific about what to change
- Show example code when helpful
- Reference documentation or best practices
- Suggest concrete next steps
- **Data over opinions**: Support suggestions with facts, research, or metrics
- **Resolve conflicts collaboratively**: Escalate if consensus isn't reached

### Optimize Cognitive Load
- Review related code changes together (maintain context)
- Focus on delocalized defects (issues spanning multiple files)
- Take breaks to avoid mental fatigue
- Use automated tools to handle routine checks
- Review code in logical order (dependencies first, then dependents)

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
- **Delocalized defects**: Issues spanning multiple files/modules (hardest to find)
- **Scope complexity**: Too much code to understand holistically
- **Conceptual complexity**: Intricate logic that's hard to follow

## Modern Pull Request Workflow

### GitHub Flow (Industry Standard)
1. **Create descriptive branch** from main (e.g., "fix-oauth2-token-refresh")
2. **Commit regularly**, pushing to same-named remote branch
3. **Open pull request** when ready for feedback or help
4. **Request reviews** from relevant experts (@mention or assign)
5. **Address feedback** through discussion and additional commits
6. **Merge after approval** and passing all automated checks
7. **Deploy to production** (or main branch merges auto-deploy)

### Pull Request Best Practices
- **Link to issues**: Reference ticket/issue number in PR description
- **Describe changes**: Explain what, why, and how
- **Add screenshots**: For UI changes, include before/after visuals
- **Request specific reviewers**: Based on domain expertise, not just availability
- **Respond promptly**: Don't let PRs stagnate (aim for <24hr turnaround)
- **Resolve conversations**: Mark discussions as resolved when addressed

### Reviewer Selection
Research shows working memory capacity affects review quality. Consider:
- **Domain expertise**: Reviewers familiar with the codebase area
- **Fresh perspective**: Rotate reviewers to prevent blind spots
- **Specialization**: Match reviewer skills to change type (security, performance, etc.)
- **Cognitive capacity**: Don't overload reviewers with too many concurrent reviews
- **Delocalized changes**: Assign reviewers with higher working memory for complex, multi-file changes

## Automation Integration

### Automate Routine Checks
Research confirms SAST tools have ~52% detection rate but 76% false positives. Use them as **assistants, not replacements**.

#### Pre-Review Automation
- **Linting**: Style violations, code formatting (ESLint, Pylint, RuboCop)
- **Static analysis**: Type checking, dead code detection (TypeScript, mypy, SonarQube)
- **Security scanning**: SAST tools (CodeQL, Snyk, Bandit)
- **Test execution**: Unit, integration, and E2E tests in CI/CD
- **Code coverage**: Ensure new code has adequate test coverage
- **Performance benchmarks**: Detect performance regressions

#### During Review
- **AI-assisted reviews**: GitHub Copilot, Amazon CodeGuru (for suggestions, not decisions)
- **Automated comments**: Bots that flag common issues
- **Change visualization**: Diff tools, code metrics dashboards

#### Post-Review
- **Automated merging**: After all checks pass and approvals received
- **Deployment pipelines**: CI/CD to staging/production
- **Monitoring**: Track post-deployment metrics

### CI/CD Integration
1. **Block merging** if any automated checks fail
2. **Require passing tests** before human review begins
3. **Generate review artifacts**: Coverage reports, performance benchmarks, security scans
4. **Automated rollback**: If deployment causes issues
5. **Fast feedback loops**: Tests should complete in <10 minutes

## Research-Backed Insights

### Code Review Effectiveness & Cognitive Load (Baum et al., 2019)
Empirical research on 50 professional developers reveals:
- **Working memory matters**: Higher working memory capacity correlates with better detection of **delocalized defects** (bugs spanning multiple files)
- **Size kills effectiveness**: Review effectiveness significantly drops for larger, more complex changes
- **Optimal change size**: Keep reviews focused and small for maximum defect detection
- **Cognitive overload**: Mental load increases with change size and complexity, reducing review quality

**Practical Implications**:
1. **Reviewer selection**: Assign complex, multi-file changes to reviewers with proven ability to track cross-cutting concerns
2. **Change decomposition**: Break large features into smaller, reviewable increments
3. **Reduce mental load**: Use automated tools for routine checks so reviewers can focus on complex logic
4. **Support reviewers**: Provide change summaries, dependency graphs, or context to reduce cognitive burden

### Code Review Outcomes & Change Types (Fregnan et al., 2022)
Analysis of three major open-source projects shows:
- **Majority changes focus on evolvability**: Documentation, structure, and maintainability improvements
- **Type-level changes dominate**: Most review changes happen at class/module level
- **Self-initiated changes**: Most improvements aren't triggered by reviewer comments—authors self-improve
- **Size correlates with changes**: Number of review changes relates to initial patch size and new lines added

**Key Insight**: Good code review culture encourages authors to proactively improve code quality, not just respond to reviewer demands.

### Modern Code Review at Scale (Microsoft Research)
Microsoft's CodeFlow insights:
- **Treat review as first-class citizen**: Track and evaluate review metrics, not just a checkbox
- **Define clear objectives**: Know what you're optimizing for (quality, knowledge sharing, compliance)
- **Measure what matters**: Track metrics aligned with objectives (defect escape rate, review time, etc.)
- **Organic evolution**: No one-size-fits-all; adapt process to team needs
- **Review speed matters**: Fast reviews encourage continuous improvement; slow reviews discourage participation

### Effectiveness of Static Analysis Tools (Charoenwet et al., 2024)
Recent empirical research on SAST reveals:
- **52% detection rate**: A single SAST can produce warnings in vulnerable functions of 52% of real-world vulnerability-contributing commits
- **76% false positive challenge**: At least 76% of warnings in vulnerable functions are irrelevant to actual vulnerabilities
- **13% efficiency gain**: Prioritizing changed functions with SAST warnings reduces Initial False Alarm by 13%
- **22% gap remains**: 22% of vulnerabilities remain undetected due to SAST rule limitations

**Practical Takeaway**: Use SAST tools as assistants, not replacements. They improve efficiency but require human judgment to filter false positives and catch gaps.

### Code Security Audit Best Practices (Ma et al., 2022)
Research on Python SAST tools demonstrates:
- **Abstract Syntax Tree (AST) analysis** combined with data flow relationships improves vulnerability detection
- **Plug-in architecture** allows customization of security rules for specific contexts
- **Fast and efficient**: SAST doesn't require runtime environment configuration
- **Early detection**: Finding security issues at code layer allows lower-cost fixes

### Comprehensive Security Analysis (Chess & McGraw, 2004)
Survey research on security vulnerability detection identifies key vulnerability categories:
- **Access control vulnerabilities** (broken authentication, authorization)
- **Information flow vulnerabilities** (injection flaws, XSS, unvalidated input)
- **API conformance issues** (improper cryptographic usage, insecure storage)
- **Stack-based and role-based access control** require different analysis approaches

**Key Insight**: Security reviews must address multiple vulnerability types systematically - no single tool or technique catches everything.

### Industry Best Practices Summary (2024)
From comprehensive industry research:
1. **Define clear goals and metrics**: Know what success looks like and measure it
2. **Keep reviews short and focused**: 200-400 lines optimal for defect detection
3. **Automate where possible**: Free human reviewers for complex judgment calls
4. **Encourage constructive feedback**: Focus on improvement, support with data
5. **Rotate reviewers**: Prevent bottlenecks, share knowledge, avoid silos
6. **Use checklists**: Ensure consistency and reduce bias across reviews
7. **Track and improve**: Continuously refine the review process based on metrics

## Additional Resources

For more detailed guidelines, see [review-guidelines.md](review-guidelines.md) for language-specific best practices.

## References

### Academic Research
- Baum, T., Schneider, K., & Bacchelli, A. (2019). "Associating working memory capacity and code change ordering with code review performance". Empirical Software Engineering, 24(4), 1762-1798.
- Charoenwet, W., Thongtanunam, P., Pham, V-T., & Treude, C. (2024). "An Empirical Study of Static Analysis Tools for Secure Code Review". arXiv:2407.12241
- Chess, B., & McGraw, G. (2004). "Static analysis for security". IEEE Security & Privacy.
- Fregnan, E., Petrulio, F., & Bacchelli, A. (2022). "A Large-Scale Study on the Effects of Code Review on Code Quality". ACM/IEEE International Conference on Software Engineering (ICSE).
- Fu, Q., et al. (2017). "Code Review and Cooperative Pair Programming Best Practice". International Journal of Software Engineering & Applications, 8(4), 13-19.
- Ma, L., Yang, H., Xu, J., Yang, Z., Lao, Q., & Yuan, D. (2022). "Code Analysis with Static Application Security Testing for Python Program". Journal of Signal Processing Systems, 94(11), 1169-1182.

### Industry Standards
- Google. (2024). "Google Engineering Practices - Code Review". https://google.github.io/eng-practices/review/
- Kim, G., Humble, J., Debois, P., Willis, J., & Forsgren, N. (2021). "The DevOps Handbook" (2nd Edition). IT Revolution Press.
- Microsoft Research. "CodeFlow: Improving the Code Review Process at Microsoft". ACM Queue.

### Additional Reading
- SmartBear. "Best Practices for Code Review". https://smartbear.com/learn/code-review/best-practices-for-peer-code-review/
- GitHub. "About pull request reviews". https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests
