# Design Principles for Data-Intensive Applications

This document provides detailed design principles derived from "Designing Data-Intensive Applications" and distributed systems research.

## Table of Contents

1. [Foundation Principles](#foundation-principles)
2. [Data Modeling Principles](#data-modeling-principles)
3. [Distributed Systems Principles](#distributed-systems-principles)
4. [Operational Principles](#operational-principles)
5. [Performance Principles](#performance-principles)

## Foundation Principles

### 1. Design for Failure (Fault Tolerance)

**Principle:** Assume all components will eventually fail. Build systems that continue operating correctly despite failures.

**Implementation strategies:**
- **Redundancy:** No single points of failure
  - Multiple replicas of data
  - Multiple instances of services
  - Multi-AZ or multi-region deployment
  
- **Graceful degradation:** System provides reduced functionality rather than total failure
  - Serve stale data when source unavailable
  - Disable non-critical features under load
  - Return cached results when backend slow
  
- **Failure detection and recovery:**
  - Health checks and heartbeats
  - Automatic failover mechanisms
  - Self-healing infrastructure
  
- **Retries with backoff:**
  ```python
  # Exponential backoff with jitter
  import random
  import time
  
  def retry_with_backoff(func, max_retries=5):
      for attempt in range(max_retries):
          try:
              return func()
          except TransientError:
              if attempt == max_retries - 1:
                  raise
              # Exponential backoff: 2^attempt * 100ms, plus jitter
              sleep_time = (2 ** attempt) * 0.1 + random.uniform(0, 0.1)
              time.sleep(sleep_time)
  ```

**Real-world examples:**
- Netflix Chaos Monkey: Randomly terminates instances to ensure resilience
- Amazon's "cell-based architecture": Failure in one cell doesn't affect others

### 2. Measure What Matters

**Principle:** Track metrics that reflect actual user experience, not just system averages.

**Key metrics:**

**Latency percentiles:**
- **p50 (median):** Half of requests faster than this
- **p95:** 95% of requests faster than this
- **p99:** 99% of requests faster than this
- **p99.9:** Only 1 in 1000 requests slower

**Why percentiles matter:**
- Averages hide outliers that affect real users
- Tail latencies often indicate systemic issues
- Slowest requests may come from high-value users (more data)

```python
# Track percentile latencies
import numpy as np

latencies = [...]  # collect request latencies
p50 = np.percentile(latencies, 50)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)
p999 = np.percentile(latencies, 99.9)
```

**Other critical metrics:**
- **Throughput:** Requests per second, transactions per second
- **Error rate:** Percentage of failed requests
- **Saturation:** Resource utilization (CPU, memory, disk I/O, network)
- **Availability:** Uptime percentage (e.g., 99.9% = ~8.76 hours downtime/year)

### 3. Understand Your Load Parameters

**Principle:** Different systems have different load characteristics. Understand yours before designing.

**Load parameters to measure:**
- **Request rate:** Requests per second (peak vs average)
- **Data volume:** Total data size, growth rate
- **Read/Write ratio:** Proportion of reads vs writes
- **Fan-out:** Number of dependent requests per user action
- **Concurrency:** Simultaneous active users or connections
- **Data distribution:** Hot keys, temporal patterns, geographic distribution

**Example - Twitter timeline:**
- **Problem:** Display user's home timeline
- **Load parameters:**
  - 300k read requests/sec (home timeline loads)
  - 12k writes/sec (tweet creation)
  - Average 75 followers per user
  - Max 30M followers for celebrities
  - High fan-out on writes (push tweets to all followers)

**Design implications:**
- Pre-compute timelines for most users (push model)
- Special handling for celebrity accounts (pull model)
- Hybrid approach based on follower count

## Data Modeling Principles

### 4. Choose Data Model Based on Access Patterns

**Principle:** Your data model should optimize for how data is queried, not just how it's structured logically.

#### Relational Model
**Best for:**
- Complex joins across tables
- Strict schema enforcement
- Strong consistency requirements
- Ad-hoc queries with flexible filtering

**Schema example:**
```sql
-- Normalized relational schema
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE,
    email VARCHAR(100)
);

CREATE TABLE posts (
    post_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id),
    content TEXT,
    created_at TIMESTAMP
);

CREATE TABLE comments (
    comment_id SERIAL PRIMARY KEY,
    post_id INT REFERENCES posts(post_id),
    user_id INT REFERENCES users(user_id),
    content TEXT
);
```

#### Document Model
**Best for:**
- Self-contained documents with related data
- Schema flexibility
- Data locality (related data stored together)
- One-to-many relationships within document

**Schema example:**
```json
{
  "user_id": "123",
  "username": "john_doe",
  "email": "john@example.com",
  "posts": [
    {
      "post_id": "456",
      "content": "Hello world",
      "created_at": "2024-01-01T10:00:00Z",
      "comments": [
        {
          "comment_id": "789",
          "user_id": "999",
          "content": "Great post!"
        }
      ]
    }
  ]
}
```

**Trade-off:** Duplication vs joins
- Document model: Data duplication, no joins needed
- Relational model: Normalized, joins required

### 5. Denormalize for Read Performance

**Principle:** In read-heavy systems, duplicate data to avoid expensive joins.

**When to denormalize:**
- Read/write ratio heavily favors reads
- Join operations becoming bottleneck
- Data doesn't change frequently
- Consistency can be eventual

**Example:**
```javascript
// Instead of joining users + posts
// Store denormalized data
{
  "post_id": "123",
  "content": "Hello world",
  "author": {
    "user_id": "456",
    "username": "john_doe",  // Denormalized
    "avatar_url": "..."       // Denormalized
  },
  "comments_count": 42        // Denormalized aggregate
}
```

**Maintenance cost:**
- Must update denormalized data when source changes
- Risk of inconsistency if updates fail
- More storage space required

### 6. Use Appropriate Indexes

**Principle:** Indexes speed up reads but slow down writes. Choose strategically.

**Index types:**

**B-Tree indexes (default in most databases):**
- Good for: Equality and range queries
- Structure: Sorted tree, O(log n) lookups
- Use when: Querying on specific columns frequently

**Hash indexes:**
- Good for: Exact match lookups only
- Structure: Hash table, O(1) lookups
- Use when: Point queries, no range scans needed

**LSM-Tree indexes:**
- Good for: Write-heavy workloads
- Structure: Log-structured, sequential writes
- Use when: High write throughput priority

**Full-text indexes:**
- Good for: Text search queries
- Structure: Inverted index
- Use when: Search functionality needed

**Trade-offs:**
```python
# Without index: Full table scan O(n)
SELECT * FROM users WHERE email = 'john@example.com';

# With index on email: O(log n) lookup
CREATE INDEX idx_users_email ON users(email);

# Cost: Each INSERT/UPDATE must update index
# Storage: Additional space for index structure
```

## Distributed Systems Principles

### 7. Embrace Eventual Consistency (When Appropriate)

**Principle:** Strict consistency is expensive in distributed systems. Many use cases can tolerate eventual consistency.

**Consistency levels:**

**Strong consistency:**
- All nodes see same data at same time
- Requires coordination (costly in distributed systems)
- Examples: Single-leader replication with synchronous writes

**Eventual consistency:**
- All nodes will eventually converge to same state
- No coordination required (better performance/availability)
- Examples: DNS, Cassandra with quorum settings

**When eventual consistency is acceptable:**
- Social media likes/views counts
- Analytics and metrics (slightly stale okay)
- Caching layers
- Recommendation systems

**When strong consistency required:**
- Financial transactions
- Inventory management
- User authentication
- Booking systems (avoid double-booking)

### 8. Partition for Scalability

**Principle:** Distribute data across nodes to scale beyond single machine limits.

**Partitioning strategies:**

**Hash partitioning:**
```python
# Hash function determines partition
partition = hash(user_id) % num_partitions

# Pros: Even distribution, no hotspots
# Cons: Range queries require scatter-gather
```

**Range partitioning:**
```python
# Key ranges assigned to partitions
if user_id < 1000:
    partition = 0
elif user_id < 2000:
    partition = 1
# ...

# Pros: Efficient range queries, data locality
# Cons: Risk of hotspots (e.g., recent timestamps)
```

**Composite partitioning:**
```python
# Partition by hash, sub-partition by range
partition = hash(tenant_id) % num_partitions
# Within partition, sort by timestamp for range queries

# Pros: Combines benefits of both approaches
# Cons: More complexity
```

**Partition key selection guidelines:**
- High cardinality (many distinct values)
- Uniform distribution (avoid hotspots)
- Aligns with query patterns
- Stable over time (avoid frequent rebalancing)

### 9. Replicate for Availability and Performance

**Principle:** Multiple copies of data provide fault tolerance and enable read scaling.

**Replication patterns:**

**Leader-follower (single-leader):**
```
     ┌─────────┐
     │ Leader  │ ◄── All writes
     └────┬────┘
          │ Replicate
    ┌─────┴──────┐
    │            │
┌───▼──┐    ┌───▼──┐
│Follower│    │Follower│ ◄── Reads
└──────┘    └──────┘
```

**Synchronous vs Asynchronous replication:**

| Aspect | Synchronous | Asynchronous |
|--------|-------------|--------------|
| Consistency | Strong | Eventual |
| Write latency | Higher (wait for followers) | Lower (immediate) |
| Durability | Guaranteed on followers | Risk of data loss |
| Availability | Lower (followers must be available) | Higher |

**Replication lag handling:**
```javascript
// Read-your-writes consistency
// User should see their own writes immediately
async function updateProfile(userId, data) {
  await writeToLeader(userId, data);
  
  // Read from leader for this user's next request
  // Or wait for replication lag before reading from follower
  await sleep(estimatedReplicationLag);
  
  return readFromFollower(userId);
}
```

### 10. Design for Network Partitions

**Principle:** Network failures will happen. Systems must handle partitions gracefully.

**CAP Theorem:** In presence of network Partition, choose Consistency OR Availability

**CP systems (Consistency + Partition tolerance):**
- Reject requests that can't guarantee consistency
- Example: Traditional relational databases in split-brain scenario
- Use when: Correctness more important than availability

**AP systems (Availability + Partition tolerance):**
- Accept requests even if can't guarantee consistency
- Example: DynamoDB, Cassandra with eventual consistency
- Use when: Availability more important than immediate consistency

**Practical approach:**
```python
# Handle partition with circuit breaker
class CircuitBreaker:
    def __init__(self, failure_threshold=5):
        self.failures = 0
        self.threshold = failure_threshold
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func):
        if self.state == "OPEN":
            # Fail fast, don't try to call
            raise CircuitOpenError("Circuit breaker is OPEN")
        
        try:
            result = func()
            self.on_success()
            return result
        except NetworkError:
            self.on_failure()
            raise
    
    def on_failure(self):
        self.failures += 1
        if self.failures >= self.threshold:
            self.state = "OPEN"
    
    def on_success(self):
        self.failures = 0
        self.state = "CLOSED"
```

## Operational Principles

### 11. Observability is Non-Negotiable

**Principle:** You cannot operate what you cannot observe.

**Three pillars of observability:**

**1. Metrics (quantitative data):**
```python
# Track business and system metrics
metrics = {
    # Application metrics
    "requests_per_second": 1250,
    "error_rate": 0.002,  # 0.2%
    "p99_latency_ms": 245,
    
    # Infrastructure metrics
    "cpu_utilization": 0.65,
    "memory_used_gb": 12.5,
    "disk_io_ops": 3500,
    
    # Business metrics
    "active_users": 15000,
    "transactions_per_min": 890
}
```

**2. Logs (discrete events):**
```python
import logging

# Structured logging
logger.info("user_login", extra={
    "user_id": "12345",
    "ip_address": "192.168.1.1",
    "session_id": "abc-def-ghi",
    "timestamp": "2024-01-01T10:30:00Z",
    "duration_ms": 145
})
```

**3. Traces (request flows):**
```python
# Distributed tracing
with tracer.start_span("handle_checkout") as span:
    span.set_tag("user_id", user_id)
    
    with tracer.start_span("validate_cart", child_of=span):
        validate_cart(cart_id)
    
    with tracer.start_span("process_payment", child_of=span):
        process_payment(payment_info)
    
    with tracer.start_span("create_order", child_of=span):
        create_order(order_data)
```

**Alerting best practices:**
- Alert on symptoms (user-facing issues) not causes
- Actionable alerts only (can you fix it now?)
- Clear runbooks for each alert
- Avoid alert fatigue (tune thresholds)

### 12. Plan for Schema Evolution

**Principle:** Schemas will change. Design for evolution from the start.

**Strategies:**

**1. Backward compatibility (new code reads old data):**
```javascript
// Schema v1
{
  "name": "John Doe"
}

// Schema v2 - add optional field
{
  "name": "John Doe",
  "email": "john@example.com"  // Optional, has default
}

// Old data still valid, new field absent
```

**2. Forward compatibility (old code reads new data):**
```javascript
// Old code must ignore unknown fields
function parseUser(data) {
  return {
    name: data.name
    // Ignore fields we don't understand
  };
}
```

**3. Schema versioning:**
```python
# Include version in data
{
  "schema_version": 2,
  "data": {
    "name": "John Doe",
    "email": "john@example.com"
  }
}

# Code handles multiple versions
def parse_user(raw_data):
    version = raw_data.get("schema_version", 1)
    
    if version == 1:
        return parse_v1(raw_data["data"])
    elif version == 2:
        return parse_v2(raw_data["data"])
    else:
        raise UnsupportedVersionError(version)
```

**4. Additive changes only:**
- Add new fields (don't remove)
- Add new optional parameters
- Keep old fields deprecated but functional
- Migrate data in background

### 13. Automate Operations

**Principle:** Manual operations don't scale. Automate for reliability and efficiency.

**Automation priorities:**

**1. Deployment:**
- Continuous integration/deployment pipelines
- Automated testing (unit, integration, end-to-end)
- Canary deployments and rollbacks
- Infrastructure as code

**2. Scaling:**
- Auto-scaling based on metrics
- Predictive scaling for known patterns
- Automatic load balancing

**3. Recovery:**
- Automated failover
- Self-healing infrastructure
- Automatic backup and restore

**4. Monitoring:**
- Automated anomaly detection
- Self-adjusting thresholds
- Automated incident creation

## Performance Principles

### 14. Optimize for Common Case

**Principle:** Make the frequent operations fast, even if rare operations become slower.

**Example - Database query optimization:**
```sql
-- Common case: Lookup user by ID (99% of queries)
CREATE INDEX idx_users_id ON users(user_id);

-- Rare case: Complex analytics query (1% of queries)
-- Acceptable to be slower, run in separate analytics database
```

**80/20 rule:**
- 80% of load comes from 20% of operations
- Optimize those 20% aggressively
- Acceptable to have slower path for remaining 80%

### 15. Cache Strategically

**Principle:** Caching can dramatically improve performance but adds complexity.

**Caching layers:**
```
┌──────────┐
│  Client  │ ◄── Browser cache
└────┬─────┘
     │
┌────▼─────┐
│   CDN    │ ◄── Edge cache
└────┬─────┘
     │
┌────▼─────┐
│   App    │ ◄── Application cache (Redis)
└────┬─────┘
     │
┌────▼─────┐
│    DB    │ ◄── Query cache, buffer pool
└──────────┘
```

**Cache invalidation strategies:**

**1. Time-to-live (TTL):**
```python
# Cache expires after fixed time
cache.set("user:123", user_data, ttl=3600)  # 1 hour
```

**2. Write-through:**
```python
# Update cache on every write
def update_user(user_id, data):
    db.update(user_id, data)
    cache.set(f"user:{user_id}", data)
```

**3. Cache-aside:**
```python
# Application manages cache
def get_user(user_id):
    # Try cache first
    cached = cache.get(f"user:{user_id}")
    if cached:
        return cached
    
    # On miss, load from DB and cache
    user = db.get(user_id)
    cache.set(f"user:{user_id}", user)
    return user
```

**Cache considerations:**
- What to cache (hot data, expensive computations)
- Cache size limits (eviction policy: LRU, LFU)
- Cache consistency (how to invalidate)
- Cache stampede (many requests for same missing key)

### 16. Batch and Pipeline Operations

**Principle:** Reduce overhead by grouping operations.

**Batching:**
```python
# Instead of individual writes
for item in items:
    db.write(item)  # 1000 round trips

# Batch writes
db.batch_write(items)  # 1 round trip
```

**Pipelining:**
```python
# Redis pipeline example
import redis

r = redis.Redis()
pipe = r.pipeline()

for key, value in data.items():
    pipe.set(key, value)

# Execute all commands at once
pipe.execute()
```

**Trade-offs:**
- Latency vs throughput
- Individual request latency increases
- Overall throughput increases significantly
- Partial failure handling more complex

## Summary

These principles form the foundation for designing reliable, scalable, and maintainable data-intensive applications. The key is not to apply all principles rigidly, but to understand the trade-offs and choose appropriately for your context.

**Golden rules:**
1. **Understand your requirements first** (don't pick solutions before understanding problems)
2. **Measure before optimizing** (premature optimization is the root of all evil)
3. **Start simple, add complexity only when needed** (YAGNI - You Aren't Gonna Need It)
4. **Make trade-offs explicit** (document why you chose A over B)
5. **Plan for evolution** (systems change, make change easy)
