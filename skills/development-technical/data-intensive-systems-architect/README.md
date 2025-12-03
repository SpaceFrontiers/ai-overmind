# Data-Intensive Systems Architect

> **Agent Skill for designing reliable, scalable, and maintainable data-intensive applications**

## Quick Reference

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[SKILL.md](SKILL.md)** | Core skill with principles, checklists, decision matrices | Start here for any architecture question |
| **[algorithms.md](algorithms.md)** | Consensus, partitioning, probabilistic data structures | Algorithm selection and implementation |
| **[design-principles.md](design-principles.md)** | Detailed design patterns and implementation guidance | Deep dive on specific principles |
| **[database-internals.md](database-internals.md)** | Storage engines, B-Trees, LSM-Trees, MVCC, WAL | Storage technology decisions |
| **[concurrency-primitives.md](concurrency-primitives.md)** | Lock-free algorithms, synchronization, consensus | High-performance concurrency |
| **[trade-offs.md](trade-offs.md)** | CAP, CRDTs, observability, time-series, saga patterns | Evaluating architectural options |
| **[examples.md](examples.md)** | E-commerce, social media, analytics architectures | Real-world reference patterns |

## When to Use This Skill

- Designing new data-intensive applications
- Scaling systems beyond single-machine capacity
- Evaluating storage/processing technology choices
- Architecting distributed systems
- Debugging performance or reliability issues
- Conducting architecture reviews

## Core Principles

1. **Reliability** - Systems work correctly even when things go wrong
2. **Scalability** - Handle increasing load while maintaining performance  
3. **Maintainability** - Easy to operate, understand, and evolve

## Key Decision Frameworks

| Trade-off | Choose A When | Choose B When |
|-----------|---------------|---------------|
| **CP vs AP** | Correctness critical (finance, inventory) | Availability critical (social, caching) |
| **B-Tree vs LSM** | Read-heavy OLTP | Write-heavy ingestion |
| **Strong vs Eventual** | Transactions, booking | Likes, analytics |
| **Normalized vs Denormalized** | Write-heavy, consistency | Read-heavy, performance |
| **Lock-based vs Lock-free** | Simplicity, low contention | Max throughput, high contention |

## Golden Rules

> **No perfect solution exists. Make intentional, context-aware trade-offs.**

1. **Design for failure** - Assume components will fail
2. **Measure what matters** - Track p95/p99 latencies, not averages
3. **Start simple** - Add complexity only when needed
4. **Document decisions** - Record trade-offs and rationale

## Sources

- **"Designing Data-Intensive Applications"** - Martin Kleppmann (O'Reilly, 2017)
- **"Database Internals"** - Alex Petrov (O'Reilly, 2019)
- **"Designing Distributed Systems"** - Brendan Burns (O'Reilly, 2nd Ed, 2024)
- **"Concurrent Programming"** - Michel Raynal (Springer, 2013)
- Production systems: RocksDB, PostgreSQL, Cassandra, CockroachDB, Hekaton

---

**Version:** 1.5.0 | **Updated:** December 2025
