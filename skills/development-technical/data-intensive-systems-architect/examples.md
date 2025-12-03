# Real-World Architecture Examples

> Part of [Data-Intensive Systems Architect](SKILL.md) skill

Practical examples of data-intensive system architectures demonstrating DDIA principles in real-world scenarios.

## Example 1: E-Commerce Platform

### Requirements
- Handle 10,000 requests/second peak
- Product catalog: 1M products
- User base: 5M users
- Order processing: Strong consistency required
- Product browsing: Eventual consistency acceptable

### Architecture Decisions

#### Data Model Choice
```javascript
// Product Catalog - Document Database (MongoDB)
{
  "product_id": "abc123",
  "name": "Laptop",
  "description": "...",
  "price": 999.99,
  "inventory_count": 50,  // Cached, eventually consistent
  "categories": ["electronics", "computers"],
  "reviews": [  // Embedded for locality
    {"rating": 5, "comment": "Great!"}
  ]
}

// Orders - Relational Database (PostgreSQL)
CREATE TABLE orders (
  order_id SERIAL PRIMARY KEY,
  user_id INT NOT NULL,
  total_amount DECIMAL(10,2),
  status VARCHAR(50),
  created_at TIMESTAMP
);

CREATE TABLE order_items (
  item_id SERIAL PRIMARY KEY,
  order_id INT REFERENCES orders(order_id),
  product_id VARCHAR(50),
  quantity INT,
  price DECIMAL(10,2)
);
```

**Rationale:**
- Product catalog uses document model for flexible schema and fast reads
- Orders use relational model for ACID transactions and complex queries

#### Scaling Strategy
```
                    ┌─────────┐
                    │   CDN   │ ← Static assets
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │Load Bal.│
                    └────┬────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼────┐      ┌────▼────┐     ┌────▼────┐
   │  App    │      │  App    │     │  App    │ ← Horizontal scaling
   │ Server  │      │ Server  │     │ Server  │
   └────┬────┘      └────┬────┘     └────┬────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
            ┌────────────┼────────────┐
            │            │            │
       ┌────▼────┐  ┌────▼────┐ ┌────▼────┐
       │ Redis   │  │MongoDB  │ │Postgres │
       │ Cache   │  │(Products│ │(Orders) │
       └─────────┘  └─────────┘ └─────────┘
```

#### Implementation Highlights

**1. Product Search with Caching:**
```python
class ProductService:
    def get_product(self, product_id):
        # L1: Check cache (Redis)
        cached = redis.get(f"product:{product_id}")
        if cached:
            return json.loads(cached)
        
        # L2: Query MongoDB
        product = mongo.products.find_one({"_id": product_id})
        
        # Cache for 1 hour (eventual consistency OK)
        redis.setex(f"product:{product_id}", 3600, json.dumps(product))
        
        return product
```

**2. Order Processing with Strong Consistency:**
```python
class OrderService:
    def create_order(self, user_id, items):
        with postgres.transaction():
            # Check inventory (strong consistency)
            for item in items:
                inventory = self.get_inventory_for_update(item.product_id)
                if inventory < item.quantity:
                    raise InsufficientInventoryError()
            
            # Create order
            order_id = self.insert_order(user_id, items)
            
            # Decrement inventory
            for item in items:
                self.decrement_inventory(item.product_id, item.quantity)
            
            return order_id
```

### Monitoring
- Track p99 latency for product page loads
- Monitor inventory synchronization lag
- Alert on order processing failures
- Track cache hit ratio

---

## Example 2: Social Media Feed

### Requirements
- 100M daily active users
- 500M posts/day
- Real-time feed updates
- High read/write ratio (100:1)
- Eventual consistency acceptable

### Architecture Decisions

#### Data Model
```javascript
// User Profile - PostgreSQL
CREATE TABLE users (
  user_id BIGSERIAL PRIMARY KEY,
  username VARCHAR(50) UNIQUE,
  email VARCHAR(100),
  created_at TIMESTAMP
);

// Posts - Cassandra (write-optimized)
CREATE TABLE posts (
  user_id BIGINT,
  post_id UUID,
  content TEXT,
  created_at TIMESTAMP,
  PRIMARY KEY (user_id, created_at, post_id)
) WITH CLUSTERING ORDER BY (created_at DESC);

// Feed - Pre-computed (Redis)
// Key: user:123:feed
// Value: Sorted set of post_ids with timestamps
```

**Rationale:**
- Cassandra for posts: Write-heavy, time-series data
- Redis for feeds: Fast read access, in-memory
- PostgreSQL for users: Structured data, relationships

#### Feed Generation Strategy

**Option 1: Fan-out on Write (Push)**
```python
# When user posts, push to all followers' feeds
def create_post(user_id, content):
    post_id = cassandra.insert_post(user_id, content)
    
    # Get followers (from cache/DB)
    followers = get_followers(user_id)
    
    # Push to each follower's feed (async)
    for follower_id in followers:
        redis.zadd(
            f"user:{follower_id}:feed",
            {post_id: timestamp}
        )
```

**Pros:** Fast reads (pre-computed)
**Cons:** Slow writes for users with many followers

**Option 2: Fan-out on Read (Pull)**
```python
# When user reads feed, pull from followed users
def get_feed(user_id):
    following = get_following(user_id)
    
    # Pull recent posts from each followed user
    all_posts = []
    for followed_id in following:
        posts = cassandra.get_recent_posts(followed_id, limit=100)
        all_posts.extend(posts)
    
    # Sort and return top posts
    return sorted(all_posts, key=lambda p: p.created_at)[:50]
```

**Pros:** Fast writes, no fan-out cost
**Cons:** Slow reads (compute on demand)

**Hybrid Approach (Recommended):**
```python
# Use push for most users, pull for celebrities
def create_post(user_id, content):
    post_id = cassandra.insert_post(user_id, content)
    follower_count = get_follower_count(user_id)
    
    if follower_count < 10000:
        # Fan-out on write for regular users
        fan_out_to_followers(user_id, post_id)
    else:
        # Celebrities: fan-out on read
        mark_as_celebrity_post(user_id, post_id)

def get_feed(user_id):
    # Get pre-computed feed
    feed = redis.zrange(f"user:{user_id}:feed", 0, 50)
    
    # Merge with celebrity posts (pull)
    celebrities = get_followed_celebrities(user_id)
    for celeb_id in celebrities:
        recent = cassandra.get_recent_posts(celeb_id, limit=10)
        feed.extend(recent)
    
    return sorted(feed)[:50]
```

### Partitioning Strategy
```python
# Partition posts by user_id (hash)
partition = hash(user_id) % num_partitions

# Within partition, order by timestamp
# Efficient queries: "Get user's posts" and "Get recent posts"
```

### Monitoring
- Feed generation latency (p99)
- Replication lag in Cassandra
- Redis memory usage
- Celebrity post fan-out time

---

## Example 3: Analytics Platform

### Requirements
- Ingest 1M events/second
- Store 1 year of data (30TB)
- Complex aggregation queries
- Batch processing acceptable
- Historical data analysis

### Architecture

```
Events ──► Kafka ──► Stream Processor ──► Warehouse
                         │                     │
                         └─► Real-time ────────┤
                              Dashboard    Query Engine
```

#### Technology Choices

**Ingestion: Apache Kafka**
- High throughput message queue
- Persistent log
- Replay capability

**Stream Processing: Apache Flink**
- Real-time aggregations
- Windowing operations
- Exactly-once semantics

**Storage: Columnar Database (ClickHouse/Snowflake)**
- Optimized for analytical queries
- Compression
- Parallel query execution

#### Data Pipeline
```python
# Kafka Producer
def track_event(event_type, user_id, properties):
    event = {
        "event_type": event_type,
        "user_id": user_id,
        "timestamp": time.time(),
        "properties": properties
    }
    kafka_producer.send("events", value=event)

# Flink Stream Processing
def process_stream():
    events = FlinkKafkaConsumer("events", ...)
    
    # Real-time aggregations
    events \
        .key_by(lambda e: e["event_type"]) \
        .time_window(minutes(5)) \
        .reduce(lambda a, b: a + b) \
        .add_sink(RedisSink())
    
    # Batch to warehouse
    events \
        .time_window(hours(1)) \
        .write_to_parquet() \
        .sink_to_s3()

# Load to warehouse
def load_to_warehouse():
    # Daily batch job
    clickhouse.insert_from_s3(
        "s3://bucket/events/2024-01-01/*.parquet"
    )
```

#### Query Optimization
```sql
-- Columnar storage optimizes this
SELECT 
  event_type,
  COUNT(*) as count,
  AVG(session_duration) as avg_duration
FROM events
WHERE date >= '2024-01-01'
  AND date < '2024-02-01'
  AND user_country = 'US'
GROUP BY event_type;

-- Only scans relevant columns
-- Compressed storage
-- Parallel execution across partitions
```

### Partitioning
```
-- Partition by date (time-series)
-- Sub-partition by event_type (common filter)
events/
  year=2024/
    month=01/
      day=01/
        event_type=page_view/
        event_type=purchase/
```

### Monitoring
- Kafka lag (producer vs consumer)
- Stream processing throughput
- Query latency percentiles
- Storage utilization

---

## Example 4: Multi-Region Application

### Requirements
- Global user base
- Low latency worldwide
- 99.99% availability
- Data sovereignty compliance

### Architecture

```
          Americas          Europe          Asia
             │                 │              │
        ┌────▼────┐       ┌────▼────┐   ┌────▼────┐
        │Region   │       │Region   │   │Region   │
        │US-East  │       │EU-West  │   │AP-South │
        └────┬────┘       └────┬────┘   └────┬────┘
             │                 │              │
        Multi-region replication (async)
```

#### Data Placement Strategy

**Global Data (User profiles):**
```python
# Replicate across all regions
# Write to nearest region
# Async replication to others
# Eventual consistency

class UserService:
    def update_profile(self, user_id, data):
        # Write to local region
        local_db.update(user_id, data)
        
        # Async replicate to other regions
        replicate_async(user_id, data, regions=["eu", "asia"])
```

**Regional Data (GDPR compliance):**
```python
# EU user data stays in EU
# US user data stays in US

class DataService:
    def store_user_data(self, user_id, data):
        region = self.get_user_region(user_id)
        
        if region == "EU":
            eu_db.insert(user_id, data)
        elif region == "US":
            us_db.insert(user_id, data)
        # ...
```

#### Conflict Resolution
```python
# Last-write-wins with vector clocks
class ConflictResolver:
    def resolve(self, version_a, version_b):
        if version_a.vector_clock > version_b.vector_clock:
            return version_a
        elif version_b.vector_clock > version_a.vector_clock:
            return version_b
        else:
            # Concurrent writes - application-specific resolution
            return self.merge(version_a, version_b)
```

### Monitoring
- Cross-region replication lag
- Latency from each region
- Conflict resolution rate
- Data residency compliance

---

## Key Patterns Summary

### Pattern Catalog

| Pattern | Use When | Example |
|---------|----------|---------|
| Cache-Aside | Read-heavy, relatively static data | Product catalogs |
| Write-Through Cache | Data must be in cache | Session data |
| Fan-out on Write | More reads than followers | Social feeds |
| Fan-out on Read | More followers than reads | Celebrity posts |
| CQRS | Different read/write patterns | E-commerce |
| Event Sourcing | Audit trail needed | Financial systems |
| Saga | Distributed transactions | Order processing |
| Circuit Breaker | Prevent cascade failures | Microservices |
| Bulkhead | Isolate failures | Resource pools |
| Rate Limiting | Protect from overload | API endpoints |

### Decision Tree

```
Start
  │
  ├─ High writes? ──Yes──► LSM-tree (Cassandra)
  │        │
  │       No
  │        │
  ├─ Complex joins? ──Yes──► Relational DB
  │        │
  │       No
  │        │
  ├─ Document-like? ──Yes──► Document DB
  │        │
  │       No
  │        │
  └─ Graph relationships? ──Yes──► Graph DB
```

---

## Testing Strategies

### Load Testing
```python
# Simulate realistic load
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)
    
    @task(3)  # 3x more common than other tasks
    def view_product(self):
        product_id = random.choice(self.product_ids)
        self.client.get(f"/products/{product_id}")
    
    @task(1)
    def create_order(self):
        self.client.post("/orders", json={...})
```

### Chaos Testing
```python
# Netflix Chaos Monkey style
import random

def chaos_test():
    # Randomly terminate instances
    instances = get_running_instances()
    victim = random.choice(instances)
    terminate_instance(victim)
    
    # Verify system recovers
    assert system_is_healthy(), "System failed to recover"
```

### Consistency Testing
```python
# Jepsen-style
def test_linearizability():
    # Multiple clients perform operations
    operations = []
    
    # Client 1 writes
    operations.append(("write", "key1", "value1", timestamp1))
    
    # Client 2 reads
    operations.append(("read", "key1", result, timestamp2))
    
    # Verify linearizability
    assert is_linearizable(operations)
```

---

## References

- Twitter's timeline architecture: https://blog.twitter.com/engineering/
- Uber's schemaless datastore: https://eng.uber.com/schemaless-part-one/
- Netflix's Chaos Engineering: https://netflixtechblog.com/
- LinkedIn's data infrastructure: https://engineering.linkedin.com/

