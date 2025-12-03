# Trade-Off Analysis Framework

> Part of [Data-Intensive Systems Architect](SKILL.md) skill

Comprehensive framework for analyzing trade-offs in data-intensive system design.

## Core Trade-Off Dimensions

### 1. Consistency vs Availability (CAP Theorem)

**The Fundamental Trade-off:**
In a distributed system during a network partition, you must choose between:
- **Consistency (C):** All nodes see the same data at the same time
- **Availability (A):** Every request receives a response (success or failure)
- **Partition Tolerance (P):** System continues operating despite network failures

**Reality:** You must always have Partition tolerance (networks fail), so choose between CP or AP.

#### CP Systems (Consistency + Partition Tolerance)

**Characteristics:**
- Reject writes/reads that cannot guarantee consistency
- Wait for consensus before responding
- May become unavailable during partitions

**Examples:**
- Traditional RDBMS in distributed mode
- HBase, MongoDB (with strong consistency settings)
- Zookeeper, etcd

**Use when:**
- Correctness is critical (financial transactions, inventory)
- Stale data causes serious problems
- Willing to sacrifice availability for accuracy

**Trade-offs:**
- ✅ Always correct data
- ✅ Simpler reasoning about state
- ❌ Lower availability during failures
- ❌ Higher latency (coordination overhead)

#### AP Systems (Availability + Partition Tolerance)

**Characteristics:**
- Always accept reads/writes
- Allow temporary inconsistencies
- Eventually converge to consistent state

**Examples:**
- Cassandra
- DynamoDB
- DNS
- Riak

**Use when:**
- Availability more important than immediate consistency
- Can tolerate stale reads
- Eventually consistent is acceptable

**Trade-offs:**
- ✅ Always available
- ✅ Better performance (no coordination)
- ❌ Temporary inconsistencies
- ❌ Complex conflict resolution
- ❌ Application must handle eventual consistency

#### Decision Matrix

| Use Case | Choose | Reasoning |
|----------|--------|-----------|
| Bank account balance | CP | Can't allow inconsistent balance |
| Social media likes | AP | Slightly stale count acceptable |
| Shopping cart | CP | Avoid overselling |
| Product catalog | AP | Stale data not critical |
| Seat reservations | CP | Can't double-book |
| News feed | AP | Old posts showing briefly is fine |

### 2. Latency vs Throughput

**The Trade-off:**
Optimizing for individual request speed (latency) often conflicts with optimizing for total work completed (throughput).

#### Low Latency Optimization

**Techniques:**
- In-memory caching
- Pre-computation
- Geographic distribution (data close to users)
- Synchronous processing
- Small batch sizes

**Characteristics:**
- Fast individual requests
- Good user experience
- May sacrifice total throughput
- Higher resource utilization

**Use when:**
- User-facing applications
- Real-time interactions
- Interactive queries

**Example:**
```python
# Low latency: Process immediately
def handle_request(request):
    result = process_immediately(request)  # 10ms
    return result

# p99 latency: 15ms
# Throughput: 100 req/sec
```

#### High Throughput Optimization

**Techniques:**
- Batching operations
- Asynchronous processing
- Larger buffer sizes
- Pipelining
- Compression

**Characteristics:**
- More total work completed
- Individual requests may wait
- Better resource efficiency
- Higher overall system capacity

**Use when:**
- Batch processing
- Analytics workloads
- Background jobs
- ETL pipelines

**Example:**
```python
# High throughput: Batch processing
def handle_requests(requests):
    batch = collect_batch(requests, size=1000)  # Wait to fill batch
    results = process_batch(batch)  # 500ms for 1000 items
    return results

# p99 latency: 600ms (worse)
# Throughput: 2000 req/sec (better)
```

#### Decision Framework

| Metric | Low Latency | High Throughput |
|--------|-------------|-----------------|
| Individual request time | Minimize | Accept higher |
| Total work/second | May be lower | Maximize |
| Resource efficiency | Lower | Higher |
| User experience | Better | Variable |
| Cost efficiency | Lower | Higher |

### 3. Normalization vs Denormalization

**The Trade-off:**
Store data normalized (avoiding duplication) or denormalized (duplicating for performance).

#### Normalized Data

**Approach:**
- Single source of truth
- No duplication
- Relationships via foreign keys

**Schema example:**
```sql
-- Normalized
CREATE TABLE users (id, name, email);
CREATE TABLE posts (id, user_id, content);
CREATE TABLE comments (id, post_id, user_id, text);

-- Query requires joins
SELECT p.content, u.name, c.text
FROM posts p
JOIN users u ON p.user_id = u.id
JOIN comments c ON c.post_id = p.id;
```

**Advantages:**
- ✅ No data duplication
- ✅ Updates in one place
- ✅ Consistency guaranteed
- ✅ Less storage space

**Disadvantages:**
- ❌ Slower reads (joins required)
- ❌ Complex queries
- ❌ Difficult to scale horizontally

**Use when:**
- Write-heavy workloads
- Data changes frequently
- Strict consistency required
- Smaller datasets

#### Denormalized Data

**Approach:**
- Duplicate data for performance
- Embed related data together
- Avoid joins

**Schema example:**
```javascript
// Denormalized
{
  "post_id": "123",
  "content": "Hello world",
  "author_name": "John",      // Duplicated from users
  "author_email": "j@e.com",  // Duplicated from users
  "comments": [
    {
      "text": "Nice post",
      "author_name": "Jane"    // Duplicated from users
    }
  ]
}
```

**Advantages:**
- ✅ Fast reads (no joins)
- ✅ Data locality
- ✅ Easier to scale horizontally
- ✅ Simpler queries

**Disadvantages:**
- ❌ Data duplication
- ❌ Updates in multiple places
- ❌ Risk of inconsistency
- ❌ More storage space

**Use when:**
- Read-heavy workloads
- Data rarely changes
- Query performance critical
- Eventual consistency acceptable

#### Hybrid Approach

Many systems use both:
```javascript
// Frequently queried fields denormalized
{
  "post_id": "123",
  "author_name": "John",  // Denormalized
  "author_id": "456"      // Normalized reference for details
}

// Full author details fetched separately when needed
```

### 4. Strong vs Eventual Consistency

**The Trade-off:**
Guarantee all reads see latest writes vs allow temporary inconsistencies for better performance/availability.

#### Strong Consistency

**Guarantees:**
- Read always returns most recent write
- Linearizability: operations appear instantaneous
- No stale data ever

**Implementation:**
- Synchronous replication
- Distributed locks
- Consensus protocols (Paxos, Raft)
- Two-phase commit

**Characteristics:**
- Higher latency (coordination required)
- Lower availability (may block on failures)
- Simpler application logic

**Use cases:**
- Financial transactions
- Inventory management
- Authentication systems
- Booking/reservation systems

**Example:**
```python
# Strong consistency
def transfer_money(from_account, to_account, amount):
    with distributed_transaction():
        debit(from_account, amount)   # Both succeed or both fail
        credit(to_account, amount)    # Immediately consistent
```

#### Eventual Consistency

**Guarantees:**
- Reads may return stale data temporarily
- All replicas eventually converge
- No guaranteed order of updates

**Implementation:**
- Asynchronous replication
- Conflict-free replicated data types (CRDTs)
- Last-write-wins
- Application-level conflict resolution

**Characteristics:**
- Lower latency (no coordination)
- Higher availability (always accept writes)
- More complex application logic

**Use cases:**
- Social media (likes, views)
- Caching layers
- Analytics dashboards
- Collaborative editing

**Example:**
```python
# Eventual consistency
def like_post(post_id, user_id):
    # Write to local replica immediately
    local_replica.increment_likes(post_id)
    
    # Propagate to other replicas asynchronously
    replicate_async(post_id, "increment_likes")
    
    # Different users may see different like counts temporarily
```

#### Consistency Spectrum

```
Strong                    Causal                    Eventual
  |                         |                          |
  |                         |                          |
Linearizable ─── Sequential ─── Causal ─── Read-your-writes ─── Eventual
  ↑                                                              ↑
  Slower, Less Available                    Faster, More Available
```

### 5. Read Optimization vs Write Optimization

**The Trade-off:**
Optimize storage and indexing for reads or writes, rarely both equally.

#### Read-Optimized Systems

**Storage structure:** B-Trees
- Data organized for fast lookups
- Indexing overhead acceptable
- Random access friendly

**Characteristics:**
- Fast point queries
- Efficient range scans
- Good for OLTP workloads
- Slower writes (maintain indexes)

**Examples:** PostgreSQL, MySQL (InnoDB)

**Techniques:**
```sql
-- Heavy indexing
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_user_created ON users(created_at);
CREATE INDEX idx_user_name ON users(name);

-- Materialized views for complex queries
CREATE MATERIALIZED VIEW user_stats AS
  SELECT user_id, COUNT(*) as post_count
  FROM posts
  GROUP BY user_id;
```

**Use when:**
- Read/write ratio > 10:1
- Interactive queries
- OLTP systems
- User-facing applications

#### Write-Optimized Systems

**Storage structure:** LSM-Trees (Log-Structured Merge Trees)
- Append-only writes
- Batch compaction
- Sequential I/O

**Characteristics:**
- Very fast writes
- Slower reads (may scan multiple files)
- Good for OLAP workloads
- Efficient compression

**Examples:** Cassandra, RocksDB, HBase

**Techniques:**
```python
# Optimizations
- Append-only writes (sequential I/O)
- Minimal indexing
- Batch writes
- Asynchronous compaction

# Trade-off
- Writes: O(1) - very fast
- Reads: O(log n) - slower (check multiple levels)
```

**Use when:**
- Write/read ratio > 1:10
- High write throughput needed
- Time-series data
- Logging, metrics collection

#### Hybrid Strategies

**Separate read and write paths:**
```
Writes ──► LSM-Tree ──► Compaction ──► Read-optimized store
                                              │
                                              ▼
                                          Analytics
```

**CQRS (Command Query Responsibility Segregation):**
- Separate write model (normalized, write-optimized)
- Separate read model (denormalized, read-optimized)
- Sync via events

### 6. Simplicity vs Performance

**The Trade-off:**
Simple, maintainable architecture vs highly optimized, complex system.

#### Simplicity First

**Characteristics:**
- Standard technologies
- Monolithic or simple services
- Minimal abstractions
- Conventional patterns

**Advantages:**
- ✅ Easy to understand
- ✅ Fast development
- ✅ Fewer bugs
- ✅ Lower operational overhead
- ✅ Easier hiring (common tech)

**Disadvantages:**
- ❌ May not handle extreme scale
- ❌ Less optimized performance
- ❌ Harder to optimize later

**When to choose:**
- Startup/MVP phase
- Small to medium scale
- Team expertise limited
- Time-to-market critical

**Example:**
```python
# Simple: Single PostgreSQL database
class UserService:
    def get_user(self, user_id):
        return db.query("SELECT * FROM users WHERE id = ?", user_id)
```

#### Performance First

**Characteristics:**
- Specialized technologies
- Complex caching layers
- Custom optimizations
- Microservices architecture

**Advantages:**
- ✅ Maximum performance
- ✅ Handles extreme scale
- ✅ Optimized for specific needs

**Disadvantages:**
- ❌ Complex to understand
- ❌ Harder to maintain
- ❌ More bugs
- ❌ Higher operational costs
- ❌ Difficult hiring

**When to choose:**
- Proven performance bottleneck
- Scale requirements demand it
- Have resources for maintenance
- Performance is competitive advantage

**Example:**
```python
# Complex: Multi-layer caching, sharding
class UserService:
    def get_user(self, user_id):
        # L1: Check local cache
        if user := local_cache.get(user_id):
            return user
        
        # L2: Check distributed cache
        if user := redis_cache.get(user_id):
            local_cache.set(user_id, user)
            return user
        
        # L3: Query appropriate shard
        shard = self.get_shard(user_id)
        user = shard.query("SELECT * FROM users WHERE id = ?", user_id)
        
        # Populate caches
        redis_cache.set(user_id, user)
        local_cache.set(user_id, user)
        return user
```

**Decision rule:** Start simple. Add complexity only when:
1. You have proven performance problems
2. Simple solution can't meet requirements
3. You have resources to maintain complexity

### 7. Vertical vs Horizontal Scaling

**The Trade-off:**
Bigger machines (scale up) vs more machines (scale out).

#### Vertical Scaling (Scale Up)

**Approach:**
- Add more CPU, RAM, disk to existing machine
- Single powerful server

**Advantages:**
- ✅ Simpler architecture
- ✅ No distributed system complexity
- ✅ ACID transactions easy
- ✅ Lower operational overhead

**Disadvantages:**
- ❌ Hard limits (largest available machine)
- ❌ Single point of failure
- ❌ Downtime for upgrades
- ❌ Diminishing returns on cost

**Use when:**
- Dataset fits on one machine
- Simplicity valued
- Early stage growth
- Budget allows powerful hardware

**Cost curve:**
```
Cost
  │     ┌─ Vertical scaling
  │    ╱
  │   ╱ (exponential cost)
  │  ╱
  │ ╱
  └────────────
       Capacity
```

#### Horizontal Scaling (Scale Out)

**Approach:**
- Add more machines to cluster
- Distribute load across many servers

**Advantages:**
- ✅ No hard limits
- ✅ Fault tolerance (redundancy)
- ✅ Linear cost scaling
- ✅ Rolling upgrades (no downtime)

**Disadvantages:**
- ❌ Distributed system complexity
- ❌ Network latency
- ❌ Consistency challenges
- ❌ Higher operational overhead

**Use when:**
- Growth expected beyond one machine
- High availability required
- Geographic distribution needed
- Cost efficiency at scale

**Cost curve:**
```
Cost
  │ ┌─ Horizontal scaling
  │╱
  │  (linear cost)
  │
  │
  └────────────
       Capacity
```

#### Practical Approach

**Start vertical, plan horizontal:**
1. Begin with powerful single machine
2. Design architecture to scale horizontally later
3. Switch when hitting limits or availability needs

**Hybrid approach:**
```
┌─────────────┐
│  Load       │
│  Balancer   │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
┌──▼──┐ ┌──▼──┐   ← Horizontal scaling
│App  │ │App  │     (stateless layer)
│Srv 1│ │Srv 2│
└──┬──┘ └──┬──┘
   │       │
   └───┬───┘
       │
   ┌───▼───┐
   │       │        ← Vertical scaling
   │  DB   │          (stateful layer)
   │(big)  │
   └───────┘
```

## Decision-Making Framework

### Step 1: Identify Constraints

Document your requirements:
- [ ] Performance targets (latency, throughput)
- [ ] Consistency requirements
- [ ] Availability requirements (SLA)
- [ ] Scale expectations (current and future)
- [ ] Budget constraints
- [ ] Team expertise
- [ ] Time-to-market pressure

### Step 2: Prioritize Trade-offs

Rank what matters most (1-5):
- [ ] Consistency
- [ ] Availability
- [ ] Latency
- [ ] Throughput
- [ ] Simplicity
- [ ] Cost

### Step 3: Choose Appropriate Pattern

Based on priorities, select:
- Data model
- Storage engine
- Replication strategy
- Partitioning approach
- Consistency model
- Scaling strategy

### Step 4: Document Decision

Use Architecture Decision Records (ADRs):
```markdown
# ADR-001: Use PostgreSQL with Read Replicas

## Context
- Need to scale read capacity
- 80% reads, 20% writes
- Eventual consistency acceptable for reads

## Decision
Single-leader PostgreSQL with async read replicas

## Consequences
+ Fast read scaling (add replicas)
+ Simple write path (single leader)
+ Familiar technology
- Replication lag (stale reads)
- Leader is single point of failure for writes
```

### Step 5: Monitor and Iterate

- Measure actual performance
- Validate assumptions
- Adjust as requirements change

## Common Trade-Off Patterns

### Pattern 1: Read-Heavy Social App

**Requirements:** High read volume, eventual consistency OK
**Choices:**
- ✅ Denormalized data model
- ✅ Heavy caching (CDN, Redis)
- ✅ Read replicas
- ✅ Eventual consistency
- ✅ AP system

### Pattern 2: Financial Transaction System

**Requirements:** Strong consistency, high correctness
**Choices:**
- ✅ Normalized data model
- ✅ ACID transactions
- ✅ Synchronous replication
- ✅ Strong consistency
- ✅ CP system

### Pattern 3: IoT Data Collection

**Requirements:** Massive write throughput
**Choices:**
- ✅ LSM-tree storage (Cassandra)
- ✅ Time-series partitioning
- ✅ Eventual consistency
- ✅ Write-optimized
- ✅ AP system

### Pattern 4: E-commerce Platform

**Requirements:** Balanced reads/writes, inventory accuracy
**Choices:**
- ✅ Hybrid normalized/denormalized
- ✅ Strong consistency for inventory
- ✅ Eventual consistency for product catalog
- ✅ Read replicas for product browsing
- ✅ Different consistency per use case

## Advanced Topics

### Conflict-Free Replicated Data Types (CRDTs)

CRDTs enable optimistic replication where replicas proceed independently and always converge deterministically—replicas that receive the same updates have equivalent state, regardless of order.

#### Two Main Approaches

**Operation-based CRDTs (Op-based):**
- Propagate commutative update operations between replicas
- Require exactly-once delivery (reliable causal broadcast)
- Lower bandwidth for small operations
- More complex delivery requirements

**State-based CRDTs (CvRDTs):**
- Propagate entire state between replicas
- Use merge functions (join-semilattice) to combine states
- Idempotent—can send state multiple times
- Higher bandwidth but simpler delivery

#### Common CRDT Types

| Type | Description | Use Case |
|------|-------------|----------|
| **G-Counter** | Grow-only counter | Likes, views, analytics |
| **PN-Counter** | Positive-negative counter | Inventory (add/remove) |
| **G-Set** | Grow-only set | Tags, followers |
| **OR-Set** | Observed-remove set | Shopping cart items |
| **LWW-Register** | Last-writer-wins register | User profile fields |
| **MV-Register** | Multi-value register | Conflict-preserving edits |

#### G-Counter Example

```python
class GCounter:
    def __init__(self, node_id, num_nodes):
        self.node_id = node_id
        self.counts = [0] * num_nodes
    
    def increment(self):
        self.counts[self.node_id] += 1
    
    def value(self):
        return sum(self.counts)
    
    def merge(self, other):
        # Take max of each position
        for i in range(len(self.counts)):
            self.counts[i] = max(self.counts[i], other.counts[i])
```

**Key insight:** Merge is commutative, associative, and idempotent—order doesn't matter.

#### When to Use CRDTs

| Scenario | CRDT Suitable? | Rationale |
|----------|----------------|-----------|
| Social media likes | ✅ Yes | Eventual accuracy acceptable |
| Shopping cart | ✅ Yes | OR-Set handles add/remove |
| Bank balance | ❌ No | Requires strong consistency |
| Collaborative editing | ✅ Yes | Specialized text CRDTs |
| Inventory count | ⚠️ Maybe | Risk of overselling |

**Trade-off:** CRDTs sacrifice strong consistency for availability and partition tolerance. They guarantee convergence but not immediate consistency.

### Observability: The Three Pillars

Observability is the ability to understand system state from external outputs. Essential for debugging distributed systems.

#### Metrics

**Definition:** Numeric measurements aggregated over time.

**Types:**
- **Counters:** Monotonically increasing (requests, errors)
- **Gauges:** Point-in-time values (CPU, memory, queue depth)
- **Histograms:** Distribution of values (latency percentiles)

**Best Practices:**
```python
# Good: Meaningful, actionable metrics
request_latency_seconds = Histogram(
    'request_latency_seconds',
    'Request latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

error_count = Counter(
    'http_errors_total',
    'Total HTTP errors',
    labels=['status_code', 'endpoint']
)
```

**Key metrics to track:**
- **RED:** Rate, Errors, Duration (for services)
- **USE:** Utilization, Saturation, Errors (for resources)

#### Logs

**Definition:** Timestamped, immutable records of discrete events.

**Structured logging:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "ERROR",
  "service": "order-service",
  "trace_id": "abc123",
  "span_id": "def456",
  "message": "Payment failed",
  "user_id": "user-789",
  "order_id": "order-012",
  "error_code": "INSUFFICIENT_FUNDS"
}
```

**Best Practices:**
- Use structured (JSON) logging
- Include correlation IDs (trace_id, span_id)
- Log at appropriate levels (DEBUG, INFO, WARN, ERROR)
- Avoid logging sensitive data (PII, credentials)

#### Traces

**Definition:** End-to-end request flow across services.

**Components:**
- **Trace:** Complete journey of a request
- **Span:** Single operation within a trace
- **Context propagation:** Passing trace IDs between services

```
Trace: user-request-abc123
├── Span: api-gateway (50ms)
│   ├── Span: auth-service (10ms)
│   └── Span: order-service (35ms)
│       ├── Span: inventory-check (8ms)
│       ├── Span: payment-service (20ms)
│       └── Span: notification-service (5ms)
```

**Key insight:** Traces reveal latency bottlenecks and failure points in distributed request flows.

#### Observability Trade-offs

| Aspect | Low Overhead | High Fidelity |
|--------|--------------|---------------|
| Sampling | Sample 1% of traces | Capture all traces |
| Retention | 7 days | 90 days |
| Cardinality | Few label values | Many label values |
| Cost | Lower | Higher |

### Time-Series Database Considerations

Time-series data (metrics, IoT, financial) has unique characteristics requiring specialized storage.

#### Characteristics of Time-Series Data

- **Write-heavy:** Continuous ingestion, rare updates
- **Time-ordered:** Natural ordering by timestamp
- **Append-only:** Historical data rarely modified
- **Query patterns:** Range queries, aggregations, downsampling

#### Storage Optimizations

**Compression techniques:**
- **Delta encoding:** Store differences between consecutive values
- **Run-length encoding:** Compress repeated values
- **Gorilla compression:** Facebook's timestamp/value compression (10x)

**Partitioning strategies:**
```
data/
  year=2024/
    month=01/
      day=15/
        hour=10/
          data.parquet
```

**Time-based partitioning benefits:**
- Efficient range queries
- Easy data lifecycle management (TTL)
- Parallel query execution

#### TSDB vs RDBMS

| Aspect | TSDB | RDBMS |
|--------|------|-------|
| Write throughput | Very high | Moderate |
| Point updates | Poor | Good |
| Range queries | Excellent | Good |
| Joins | Limited | Excellent |
| Compression | 10-20x | 2-3x |
| Schema flexibility | High | Low |

**Use TSDB when:**
- Ingesting millions of data points/second
- Queries are primarily time-range based
- Data is append-only or rarely updated
- Storage efficiency is critical

### Saga Pattern: Choreography vs Orchestration

Sagas manage distributed transactions across microservices without distributed locks.

#### Choreography (Event-driven)

Each service publishes events; other services react.

```
Order Service ──publish──► OrderCreated
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         Inventory       Payment        Notification
         Service         Service         Service
              │               │               │
              ▼               ▼               ▼
        StockReserved   PaymentProcessed  EmailSent
```

**Pros:**
- Loose coupling
- No single point of failure
- Services are independent

**Cons:**
- Hard to understand flow
- Difficult to debug
- Cyclic dependencies possible

#### Orchestration (Central coordinator)

Central orchestrator directs the saga.

```
                    ┌─────────────┐
                    │ Orchestrator│
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      Inventory       Payment        Notification
      Service         Service         Service
```

**Pros:**
- Clear flow visibility
- Easier debugging
- Centralized error handling

**Cons:**
- Single point of failure
- Tighter coupling to orchestrator
- Orchestrator can become complex

#### Compensation Actions

When a step fails, execute compensating transactions:

```python
class OrderSaga:
    def execute(self, order):
        try:
            # Forward flow
            inventory_id = self.reserve_inventory(order)
            payment_id = self.process_payment(order)
            self.send_notification(order)
        except PaymentError:
            # Compensation
            self.release_inventory(inventory_id)
            raise
        except NotificationError:
            # Payment succeeded, notification failed
            # Log for manual retry, don't compensate payment
            self.log_notification_failure(order)
```

#### Saga Selection Guide

| Factor | Choreography | Orchestration |
|--------|--------------|---------------|
| Team structure | Multiple teams | Single team |
| Complexity | Simple flows | Complex flows |
| Debugging needs | Low | High |
| Coupling tolerance | Low | Moderate |
| Failure handling | Distributed | Centralized |

## Summary

There are no universally correct choices, only appropriate trade-offs for your context. The key is to:

1. **Understand the trade-offs** - Know what you're giving up
2. **Match to requirements** - Choose based on actual needs
3. **Document decisions** - Record why you chose A over B
4. **Measure outcomes** - Validate your choices
5. **Iterate** - Adjust as requirements evolve

> "All engineering is about trade-offs. The mark of a good engineer is knowing which trade-offs to make."

## References

- Almeida, P.S. (2023). "Approaches to Conflict-free Replicated Data Types"
- Albuquerque, C. & Correia, F.F. (2025). "Tracing and Metrics Design Patterns for Monitoring Cloud-native Applications"
- Zhao, X. & Haller, P. (2020). "Replicated data types that unify eventual consistency and observable atomic consistency"
- Research on Saga patterns in microservices architectures
