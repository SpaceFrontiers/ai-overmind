# Data-Intensive Systems Architect Skill

A comprehensive Agent skill for designing reliable, scalable, and maintainable data-intensive applications based on principles from Martin Kleppmann's "Designing Data-Intensive Applications" (DDIA) and distributed systems best practices.

## Overview

This skill provides systematic guidance for architecting systems that handle large-scale data storage, processing, replication, and distributed computing. It emphasizes evidence-based decision-making using principles from authoritative sources.

## When to Use This Skill

Activate this skill when:
- Designing new data-intensive applications
- Scaling existing systems beyond single-machine capacity
- Evaluating technology choices for data storage/processing
- Architecting distributed systems
- Debugging performance or reliability issues
- Planning capacity and growth
- Conducting architecture reviews
- Making build vs buy decisions for data infrastructure

## Key Features

### Core Principles
- **Reliability:** Design systems that work correctly even when things go wrong
- **Scalability:** Handle increasing load while maintaining performance
- **Maintainability:** Build systems that are easy to operate, understand, and evolve

### Comprehensive Guidance
- Data model selection (relational, document, graph, wide-column)
- Storage engine considerations (B-trees vs LSM-trees)
- Replication strategies (single-leader, multi-leader, leaderless)
- Partitioning approaches (hash vs range)
- Transaction management (ACID vs eventual consistency)
- Event-driven architecture patterns

### Decision Frameworks
- CAP theorem trade-offs (Consistency vs Availability)
- Latency vs Throughput optimization
- Normalization vs Denormalization
- Strong vs Eventual Consistency
- Read vs Write optimization
- Simplicity vs Performance
- Vertical vs Horizontal scaling

## Contents

### Main Skill File
**[SKILL.md](SKILL.md)** - Complete skill with instructions, guidelines, and checklists

### Supporting Documentation
- **[design-principles.md](design-principles.md)** - Detailed design principles and implementation patterns
  - Foundation principles (failure tolerance, metrics, load parameters)
  - Data modeling principles
  - Distributed systems principles  
  - Operational principles
  - Performance principles

- **[trade-offs.md](trade-offs.md)** - Comprehensive trade-off analysis framework
  - Consistency vs Availability (CAP)
  - Latency vs Throughput
  - Normalization vs Denormalization
  - Strong vs Eventual Consistency
  - Read vs Write Optimization
  - Simplicity vs Performance
  - Vertical vs Horizontal Scaling

- **[examples.md](examples.md)** - Real-world architecture examples
  - E-Commerce Platform
  - Social Media Feed
  - Analytics Platform
  - Multi-Region Application
  - Pattern catalog and testing strategies

## Quick Start

### 1. Load the Skill
Upload the entire `data-intensive-systems-architect` directory to Claude.

### 2. Describe Your Requirements
Provide context about your system:
```
I'm building a [type of application] that needs to:
- Handle [X] requests/second
- Store [Y] GB of data
- Support [consistency/availability/latency] requirements
- Scale to [expected growth]
```

### 3. Get Recommendations
The skill will help you:
- Choose appropriate data models
- Select storage technologies
- Design replication strategy
- Plan partitioning scheme
- Make informed trade-offs

## Example Usage

**Scenario:** Designing a social media application

**Query:** "I'm building a social media platform with 100M users. Posts should appear in real-time feeds. How should I architect the feed generation?"

**Skill Response:** 
The skill will guide you through:
1. Understanding load parameters (read/write ratio, fan-out)
2. Evaluating data models (Cassandra for posts, Redis for feeds)
3. Choosing between fan-out strategies (push vs pull vs hybrid)
4. Designing for eventual consistency
5. Planning partitioning and caching
6. Monitoring and optimization

## Design Methodology

### 1. Requirements First
- Define load parameters
- Identify latency requirements
- Determine consistency needs
- Assess availability targets

### 2. Systematic Evaluation
- Match data models to access patterns
- Choose storage engines for workload
- Select replication based on consistency/availability
- Design partitioning for scale

### 3. Trade-off Analysis
- Explicitly evaluate options
- Document decisions (ADRs)
- Consider operational complexity
- Plan for evolution

### 4. Validation
- Measure actual performance
- Monitor key metrics
- Iterate based on data

## Key Principles

> **No perfect solution exists. Identify what you're optimizing for and make intentional, context-aware trade-offs rather than defaulting blindly.**

### Golden Rules
1. **Design for failure** - Assume components will fail
2. **Measure what matters** - Track percentile latencies, not averages
3. **Understand your load** - Know your access patterns and growth
4. **Choose data models wisely** - Match to query patterns
5. **Start simple** - Add complexity only when needed
6. **Plan for evolution** - Systems change, make change easy
7. **Document decisions** - Record trade-offs and rationale

## Based on Authoritative Sources

This skill synthesizes principles from:
- **"Designing Data-Intensive Applications"** by Martin Kleppmann (O'Reilly, 2017)
- AWS Well-Architected Framework - Reliability Pillar
- Distributed Systems research papers
- Production experience from large-scale systems (Twitter, LinkedIn, Netflix, Uber)

## Progressive Disclosure

The skill uses progressive disclosure to manage complexity:
- **Level 1:** SKILL.md provides core principles and checklists
- **Level 2:** design-principles.md offers detailed implementation guidance
- **Level 3:** trade-offs.md enables deep trade-off analysis
- **Level 4:** examples.md shows real-world applications

Reference additional files as needed based on your specific questions.

## Anti-Patterns Avoided

The skill helps you avoid common mistakes:
- ❌ Premature optimization
- ❌ Ignoring percentile latencies  
- ❌ Network fallacy assumptions
- ❌ Distributed transaction overuse
- ❌ Poor partition key selection
- ❌ Blind technology choices

## Contributing

To improve this skill:
1. Test with real architecture decisions
2. Document outcomes (what worked, what didn't)
3. Share edge cases and lessons learned
4. Suggest additional examples or patterns

## License

This skill is part of the AI Overmind Agent Skills repository.

## Version

**Version:** 1.0.0  
**Last Updated:** December 2024  
**Author:** AI Overmind

---

**Need help?** Start with [SKILL.md](SKILL.md) for comprehensive guidance, then explore supporting documents for deeper dives into specific topics.
