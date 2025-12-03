# Database Internals: Storage Engine Deep Dive

> Part of [Data-Intensive Systems Architect](SKILL.md) skill

Deep implementation details about database storage engines, disk-based data structures, and distributed database internals. Based on "Database Internals" by Alex Petrov (O'Reilly, 2019).

---

## Table of Contents

1. [Storage Engine Architecture](#storage-engine-architecture)
2. [B-Tree Implementation Details](#b-tree-implementation-details)
3. [LSM-Tree Architecture](#lsm-tree-architecture)
4. [Page Organization and File Formats](#page-organization-and-file-formats)
5. [Transaction Processing and Recovery](#transaction-processing-and-recovery)
6. [Concurrency Control Mechanisms](#concurrency-control-mechanisms)
7. [Write-Ahead Logging (WAL)](#write-ahead-logging-wal)
8. [Buffer Management](#buffer-management)
9. [Compaction Strategies](#compaction-strategies)
10. [Storage Engine Selection Criteria](#storage-engine-selection-criteria)

---

## Storage Engine Architecture

### Core Components

Modern database systems are built with **pluggable storage engines** that separate data storage from query processing:

```
┌─────────────────────────────────────────┐
│         Query Processing Layer          │
│  (Parser, Optimizer, Execution Engine)  │
├─────────────────────────────────────────┤
│         Transaction Manager             │
│      (ACID, Concurrency Control)        │
├─────────────────────────────────────────┤
│         Storage Engine Layer            │
│  (B-Trees, LSM-Trees, Page Management)  │
├─────────────────────────────────────────┤
│         Buffer Manager / Cache          │
├─────────────────────────────────────────┤
│      Physical Storage (Disk/SSD)        │
└─────────────────────────────────────────┘
```

**Key Insight:** Storage engines like BerkeleyDB, LevelDB, RocksDB, LMDB, and WiredTiger are developed independently and can be embedded into different DBMS systems. Examples:
- **MySQL**: InnoDB, MyISAM, MyRocks (RocksDB)
- **MongoDB**: WiredTiger, In-Memory, MMAPv1 (deprecated)
- **PostgreSQL**: Built-in heap storage with B-tree indexes

### Mutable vs Immutable Storage Structures

**Mutable Structures (In-Place Update):**
- Update data directly in its location on disk
- Examples: B-Trees, heap files
- Trade-off: Requires locking, complex concurrency control, write amplification

**Immutable Structures (Copy-on-Write):**
- Never modify existing data, always append new versions
- Examples: LSM-Trees, log-structured storage
- Trade-off: Better write throughput, requires compaction, read amplification

---

## B-Tree Implementation Details

### B-Tree Structure and Invariants

B-Trees (and their variant B⁺-Trees) are the foundation of most OLTP databases.

**Key Properties:**
- **High Fanout:** Each node contains N keys and N+1 pointers to children
- **Balanced:** All leaf nodes are at the same depth
- **Sorted:** Keys within nodes are sorted for binary search
- **Node Types:**
  - **Root Node:** Entry point to the tree
  - **Internal Nodes:** Index entries (separator keys) guiding search
  - **Leaf Nodes:** Actual data records or pointers to data

**B⁺-Tree Optimization:**
```
Traditional B-Tree: Values stored in all nodes (root, internal, leaf)
B⁺-Tree: Values ONLY in leaf nodes, internal nodes only have separator keys
```

**Why B⁺-Trees Dominate:**
- Higher fanout in internal nodes (more keys fit per page)
- All operations touch only leaf level for data access
- Sequential leaf scanning via sibling pointers

### Node Operations

**Search Operation:**
```python
def search(key, node):
    if node.is_leaf():
        return node.find_value(key)
    
    # Binary search for correct subtree
    child_index = binary_search_child(key, node.keys)
    child = load_page(node.pointers[child_index])
    return search(key, child)
```

**Insertion with Node Splits:**
```
1. Locate correct leaf node via search
2. If leaf has space: Insert key-value pair
3. If leaf is full:
   - Split node into two halves
   - Promote middle key to parent
   - If parent full, recursively split upward
   - May create new root (tree grows in height)
```

**Deletion with Node Merges:**
```
1. Locate and remove key-value pair
2. If node underflows (< minimum occupancy):
   - Try to borrow from sibling
   - If sibling can't lend: Merge with sibling
   - Recursively handle parent underflow
```

### Disk-Based B-Tree Optimizations

**Page Layout:**
```
┌──────────────────────────────────────┐
│        Page Header (metadata)        │
│  - Page ID, Type, Free Space Pointer │
├──────────────────────────────────────┤
│          Cell Pointers Array         │
│   [offset_1, offset_2, ..., offset_n]│
├──────────────────────────────────────┤
│          Free Space                  │
├──────────────────────────────────────┤
│          Cell Data (Keys + Values)   │
│   Stored from bottom up              │
└──────────────────────────────────────┘
```

**Slotted Page Organization:**
- Cells (key-value pairs) grow from bottom
- Cell pointers array grows from top
- Free space in middle
- Enables variable-size records without fragmentation

**Overflow Pages:**
- When value doesn't fit in single page
- Store overflow pointer in main page
- Chain overflow pages together
- Example: MySQL InnoDB uses overflow for large BLOBs

### Advanced B-Tree Variants

#### Blink-Trees (Lock-Free Traversal)
- Add **right-sibling pointers** at all levels
- Enable lock-free reads during concurrent splits
- Split protocol:
  1. Create new right sibling
  2. Link it via sibling pointer
  3. Update parent (can happen later)
- Readers follow sibling links if key moved during split

#### Lazy B-Trees (WiredTiger)
- Delay structural modifications (splits/merges)
- Use **update buffers** to accumulate changes in memory
- Benefits:
  - Reduced write amplification
  - Better write throughput
  - Trades read latency (must check buffers)

#### Bw-Trees (LLAMA - Latch-Free)
- No in-place updates, uses delta chains
- Mapping table translates logical page IDs to physical addresses
- Append-only delta records on top of base pages
- Consolidation merges deltas into new base page
- Enables lock-free operations via atomic CAS

### B-Tree Performance Characteristics

**Strengths:**
- ✅ O(log N) point lookups
- ✅ Efficient range scans (sequential leaf traversal)
- ✅ Good for read-heavy workloads
- ✅ Predictable performance

**Weaknesses:**
- ❌ Write amplification (entire page written for small update)
- ❌ Page splits cause fragmentation
- ❌ Random I/O for scattered inserts
- ❌ Latching/locking overhead in concurrent environments

---

## LSM-Tree Architecture

### Log-Structured Merge Tree Fundamentals

LSM-Trees replace random I/O with sequential I/O by batching writes in memory and periodically merging to disk.

**Core Concept:**
```
Writes → MemTable (in-memory, sorted) → Flush → L0 SSTables → Compact → L1, L2, ... Ln
```

### Multi-Level LSM-Tree Structure

```
┌─────────────────────────────────────────┐
│  MemTable (RAM)                         │
│  - Skip List / AVL Tree                 │
│  - Accumulates writes                   │
└────────────┬────────────────────────────┘
             ↓ Flush when full
┌─────────────────────────────────────────┐
│  Level 0 (L0) - Disk                    │
│  - Unsorted SSTables (may overlap)      │
│  - Each flush creates new SSTable       │
└────────────┬────────────────────────────┘
             ↓ Compact
┌─────────────────────────────────────────┐
│  Level 1 (L1) - Disk                    │
│  - Sorted, non-overlapping SSTables     │
│  - Size: ~10x L0                        │
└────────────┬────────────────────────────┘
             ↓ Compact
┌─────────────────────────────────────────┐
│  Level 2+ (L2, L3, ..., Ln) - Disk      │
│  - Each level ~10x previous level       │
│  - Exponentially decreasing update freq │
└─────────────────────────────────────────┘
```

### LSM-Tree Components

**1. Write-Ahead Log (WAL):**
```
Purpose: Durability before MemTable flush
Process:
  1. Append operation to WAL (sequential write)
  2. Insert into MemTable
  3. Acknowledge write to client
Recovery: Replay WAL to reconstruct MemTable after crash
```

**2. MemTable:**
- In-memory sorted structure (Skip List, Red-Black Tree, AVL)
- Fast O(log N) inserts and lookups
- Becomes immutable when full → flushed to L0 as SSTable
- New MemTable created for subsequent writes

**3. SSTable (Sorted String Table):**
```
Structure:
┌──────────────────────────────────────┐
│  Index Block                         │
│  - Key ranges → Data block offsets   │
├──────────────────────────────────────┤
│  Data Blocks                         │
│  - Sorted key-value pairs            │
│  - Compressed (snappy, zstd, lz4)    │
├──────────────────────────────────────┤
│  Bloom Filter                        │
│  - Probabilistic key existence test  │
├──────────────────────────────────────┤
│  Footer (metadata, checksums)        │
└──────────────────────────────────────┘
```

**4. Bloom Filters:**
- Probabilistic data structure: "Key definitely NOT present" or "Key might be present"
- Avoids disk reads for non-existent keys
- Size-accuracy trade-off: 10 bits/key ≈ 1% false positive rate
- **Critical optimization** for LSM read performance

### LSM-Tree Operations

**Write Path:**
```
1. Append to WAL (durability)
2. Insert into MemTable
3. Return success immediately
Background:
4. MemTable full? → Flush to L0 SSTable
5. L0 exceeds threshold? → Trigger compaction to L1
6. Ln exceeds threshold? → Compact to L(n+1)
```

**Read Path (Expensive!):**
```
1. Check MemTable (most recent data)
2. Check Immutable MemTable (being flushed)
3. For each level L0 → Ln:
   a. Check Bloom filter (skip if key not present)
   b. Search SSTable index
   c. Read data block if key might exist
4. Merge results (newest version wins)
```

**Delete Operation:**
- Write **tombstone** marker (special delete flag)
- Tombstone propagates through compaction
- Actual deletion happens during compaction when tombstone reaches oldest level

### Compaction Strategies

#### Leveled Compaction (LevelDB, RocksDB default)
```
Structure:
  L0: 4 SSTables (overlapping)
  L1: 10 MB (non-overlapping)
  L2: 100 MB (non-overlapping)
  L3: 1 GB (non-overlapping)

Process:
  - Pick SSTable from Ln
  - Identify overlapping SSTables in L(n+1)
  - Merge-sort and write new SSTables to L(n+1)
  - Delete old SSTables
```

**Pros:**
- Best space amplification (least duplication)
- Predictable query performance
- Good for read-heavy workloads

**Cons:**
- High write amplification (data rewritten multiple times)
- Compaction overhead affects tail latency

#### Size-Tiered Compaction (Cassandra, ScyllaDB)
```
Structure:
  - Group SSTables by similar size
  - Merge when N SSTables of similar size accumulate
  - No fixed levels, just size tiers

Process:
  - Wait for 4 SSTables of ~same size
  - Merge into single larger SSTable
  - Result goes into next tier
```

**Pros:**
- Lower write amplification
- Better for write-heavy workloads
- Simpler compaction logic

**Cons:**
- Higher space amplification (temporary duplication)
- Reads scan multiple SSTables per tier

#### Time-Window Compaction
```
Use case: Time-series data with TTL
Strategy:
  - Partition SSTables by time window (hour, day)
  - Compact within windows
  - Delete entire expired windows
```

**Benefit:** Fast, efficient bulk deletion for time-series

### LSM-Tree Performance Characteristics

**Strengths:**
- ✅ Excellent write throughput (sequential writes)
- ✅ High write amplification tolerance (flash-friendly)
- ✅ Efficient compression (sorted data)
- ✅ Handles write-heavy workloads (ingestion, logging)

**Weaknesses:**
- ❌ Read amplification (check multiple levels)
- ❌ Compaction overhead (CPU, I/O spikes)
- ❌ Space amplification during compaction
- ❌ Unpredictable tail latency (compaction storms)

### Real-World LSM Implementations

| Engine | Used By | Key Features |
|--------|---------|--------------|
| **LevelDB** | Chrome, Riak | Original Google implementation |
| **RocksDB** | Facebook, CockroachDB, TiDB | Optimized for SSD, tunable compaction |
| **Cassandra Engine** | Apache Cassandra | Size-tiered compaction, distributed |
| **WiredTiger** | MongoDB | LSM + B-Tree hybrid options |
| **ScyllaDB Engine** | ScyllaDB | C++ rewrite, per-core architecture |

---

## Page Organization and File Formats

### On-Disk Page Structure

**Fixed-Size Pages:**
- Typically 4KB, 8KB, or 16KB (matching OS page size)
- Atomic write unit (important for crash consistency)
- Trade-off: Larger pages → less metadata overhead, more write amplification

**Page Header (Metadata):**
```c
struct PageHeader {
    uint64_t page_id;           // Unique identifier
    uint16_t page_type;         // Leaf, Internal, Overflow
    uint16_t cell_count;        // Number of records
    uint16_t free_space_offset; // Start of free space
    uint64_t lsn;               // Log Sequence Number (recovery)
    uint32_t checksum;          // Data integrity
};
```

### Binary Encoding

**Fixed-Size Data Types:**
```
Integer (32-bit): 4 bytes
Long (64-bit): 8 bytes
Float (32-bit): IEEE 754, 4 bytes
Double (64-bit): IEEE 754, 8 bytes
Boolean: 1 byte (0 or 1)

Endianness: Little-endian (LSB first) or Big-endian (MSB first)
```

**Variable-Size Data (Strings, BLOBs):**
```
Approach 1: Length-Prefixed
  [length: 4 bytes][data: N bytes]

Approach 2: Null-Terminated (C-style strings)
  [data: N bytes][0x00]

Approach 3: Offset-Based (Slotted Pages)
  [offset: 2 bytes] → points to actual data elsewhere in page
```

### Cell Layout (Key-Value Storage)

```
┌─────────────────────────────────────────┐
│  Cell Header                            │
│  - Key size                             │
│  - Value size                           │
│  - Flags (deleted, overflow, etc.)      │
├─────────────────────────────────────────┤
│  Key Data                               │
│  - Actual key bytes                     │
├─────────────────────────────────────────┤
│  Value Data or Overflow Pointer         │
│  - Inline value if small                │
│  - Overflow page pointer if large       │
└─────────────────────────────────────────┘
```

### Checksumming for Data Integrity

**Purpose:** Detect corruption from disk errors, bit rot, software bugs

**Common Algorithms:**
- **CRC32:** Fast, good error detection, used widely
- **xxHash:** Extremely fast, modern alternative
- **SHA-256:** Cryptographic strength (overkill for integrity)

**Checksum Placement:**
```
Option 1: Per-page checksum (in page header)
Option 2: Per-cell checksum (in cell header)
Option 3: External checksum file

Trade-off: Per-cell = better granularity, more overhead
```

### File Format Versioning

**Why Needed:** Backward/forward compatibility as format evolves

**Strategies:**
```
1. Magic Number + Version Header
   [0xDEADBEEF][Version: 2][... rest of file ...]

2. Feature Flags
   Bits indicate which optional features are used

3. Schema Evolution
   Add fields with default values for old readers
```

---

## Transaction Processing and Recovery

### ACID Properties Deep Dive

**Atomicity:**
- All-or-nothing: Transaction commits fully or aborts entirely
- Implementation: Undo log (rollback changes) or shadow paging (copy-on-write)

**Consistency:**
- Database integrity constraints maintained
- Application-level invariants preserved
- Implementation: Constraint checking, triggers, cascades

**Isolation:**
- Concurrent transactions don't interfere
- Illusion of serial execution
- Implementation: Locking (pessimistic) or MVCC (optimistic)

**Durability:**
- Committed transactions survive crashes
- Implementation: Write-Ahead Logging (WAL), force-log-at-commit

### Write-Ahead Logging (WAL)

**Fundamental Rule:** Log changes BEFORE modifying data on disk

**WAL Protocol:**
```
1. Write operation received
2. Append log record to WAL buffer
3. Flush WAL buffer to disk (fsync) ← Durability point
4. Modify data page in buffer pool
5. Acknowledge transaction commit
6. Lazy write: Data page flushed to disk later
```

**Log Record Structure:**
```c
struct LogRecord {
    uint64_t lsn;              // Log Sequence Number (monotonic)
    uint64_t transaction_id;   
    uint8_t  log_type;         // BEGIN, INSERT, UPDATE, DELETE, COMMIT, ABORT
    uint64_t page_id;          // Affected page
    uint16_t offset;           // Location within page
    uint32_t before_size;      
    byte[]   before_image;     // Old value (for undo)
    uint32_t after_size;
    byte[]   after_image;      // New value (for redo)
};
```

**Log Types:**
- **Physical Logging:** Log exact byte changes at specific offsets
  - Pros: Fast recovery
  - Cons: Large log size, tied to physical layout
  
- **Logical Logging:** Log high-level operations (e.g., "INSERT row X")
  - Pros: Smaller logs, independent of physical layout
  - Cons: Complex recovery, hard to undo

- **Physiological Logging (Hybrid):** Log physical page + logical operation
  - Used by most modern systems (PostgreSQL, MySQL)

### ARIES Recovery Algorithm

**ARIES** = Algorithm for Recovery and Isolation Exploiting Semantics

**Three Phases:**

**1. Analysis Phase:**
```
Goal: Determine which transactions were active at crash
Process:
  - Scan WAL from last checkpoint
  - Build Transaction Table (status of each transaction)
  - Build Dirty Page Table (which pages had unflushed changes)
Output: Know what to redo and undo
```

**2. Redo Phase (Roll Forward):**
```
Goal: Restore database to exact state at crash moment
Process:
  - Replay all logged operations from earliest dirty page
  - Apply changes even if page already flushed (idempotent)
  - Recreate exact state, including uncommitted transactions
Principle: "Repeat history"
```

**3. Undo Phase (Roll Backward):**
```
Goal: Abort incomplete transactions
Process:
  - Identify transactions without COMMIT record
  - Undo their changes using before_image
  - Process in reverse log order (LIFO)
  - Write Compensation Log Records (CLRs) to avoid re-undo
Result: Only committed transactions remain
```

**Checkpointing:**
- Periodically write checkpoint record to WAL
- Checkpoint contains: Active transactions, dirty pages, buffer state
- Recovery starts from last checkpoint (not from beginning of log)
- Fuzzy checkpoints: Allow concurrent transactions during checkpoint

---

## Concurrency Control Mechanisms

### Lock-Based Concurrency Control (2PL)

**Two-Phase Locking Protocol:**
```
Growing Phase:
  - Transaction acquires locks as needed
  - Cannot release any lock

Shrinking Phase:
  - Transaction releases locks
  - Cannot acquire new locks

Guarantee: Ensures serializability
```

**Lock Types:**
- **Shared Lock (S):** Read access, multiple transactions can hold
- **Exclusive Lock (X):** Write access, exclusive ownership
- **Intent Locks (IS, IX):** Hierarchical locking (table → page → row)

**Lock Compatibility Matrix:**
```
      S    X
  S   ✓    ✗
  X   ✗    ✗
```

**Deadlock Handling:**
```
Detection: Maintain waits-for graph, detect cycles
Prevention: 
  - Wait-Die: Older transaction waits, younger aborts
  - Wound-Wait: Older transaction forces younger to abort
```

### Multiversion Concurrency Control (MVCC)

**Core Idea:** Maintain multiple versions of each data item, readers never block writers

**Version Storage:**
```
Approach 1: Append-Only (PostgreSQL)
  - New version appended to table
  - Old versions retained with timestamps
  - Vacuum process removes old versions

Approach 2: Time-Travel (Oracle)
  - New version in main table
  - Old versions in rollback segments

Approach 3: Delta Storage
  - Base version + delta chains
```

**Transaction Timestamps:**
```
Each transaction gets:
  - Start Timestamp (TS_start): When transaction begins
  - Commit Timestamp (TS_commit): When transaction commits

Each version has:
  - Creation Timestamp: When version created
  - Expiration Timestamp: When version superseded
```

**Version Visibility Rules (Snapshot Isolation):**
```python
def is_visible(version, transaction):
    # Version created before transaction started?
    if version.created_ts > transaction.start_ts:
        return False
    
    # Version deleted/expired before transaction started?
    if version.expired_ts <= transaction.start_ts:
        return False
    
    # Version created by this transaction?
    if version.creator_txn == transaction.id:
        return True
    
    # Version created by uncommitted transaction?
    if not is_committed(version.creator_txn):
        return False
    
    return True
```

**MVCC Benefits:**
- Readers never block writers
- Writers never block readers
- Only writers block writers (on same record)
- Consistent snapshots for long-running queries

**MVCC Challenges:**
- **Garbage Collection:** Old versions must be vacuumed
- **Version Bloat:** Can consume significant space
- **Write Skew Anomaly:** Snapshot Isolation allows some anomalies (requires Serializable Snapshot Isolation)

### Optimistic Concurrency Control (OCC)

**Three Phases:**

**1. Read Phase:**
```
- Transaction reads data, stores in private workspace
- No locks acquired
- Maintains read set (items read) and write set (items to write)
```

**2. Validation Phase:**
```
- Check if transaction conflicts with concurrent transactions
- Conflict detection:
  - Did any item in read set get modified by concurrent transaction?
  - Did any item in write set get modified by concurrent transaction?
- If conflict: Abort and retry
- If no conflict: Proceed to write phase
```

**3. Write Phase:**
```
- Apply changes from private workspace to database
- Acquire short-term locks during actual writes
- Release locks immediately after
```

**When to Use OCC:**
- ✅ Low contention workloads
- ✅ Read-heavy applications
- ✅ Short transactions
- ❌ High contention → frequent aborts

### Lock-Free Concurrency Control

Modern high-performance databases increasingly use **lock-free algorithms** to eliminate latch contention and deadlocks.

**Progress Guarantees Hierarchy:**

| Guarantee | Definition | Database Example |
|-----------|------------|------------------|
| **Blocking** | May deadlock, needs detection | PostgreSQL, MySQL InnoDB (traditional) |
| **Obstruction-Free** | Progress when alone | Rare (research systems) |
| **Lock-Free** | System-wide progress (≥1 thread succeeds) | Microsoft Hekaton Bw-Trees |
| **Wait-Free** | Per-thread bounded progress | Fetch&Add counters |

**Lock-Free Data Structures in Production:**

**1. Bw-Trees (Microsoft Hekaton)**
```
Architecture:
┌──────────────────────────┐
│ Mapping Table (CAS-based)│
│  Page ID → Address       │
└────────┬─────────────────┘
         ↓
    Delta Chain:
    [Insert Delta] ← Latest
    [Delete Delta]
    [Base Page]    ← Oldest
```

**Key Properties:**
- **10x faster** than latched B-Trees under high contention
- **Lock-free reads:** Never blocked by writes
- **Delta updates:** Append-only, flash-friendly
- **CAS-based:** Atomic pointer swaps in mapping table

**2. Lock-Free Skip Lists (LevelDB/RocksDB MemTable)**
- Concurrent inserts via Compare-and-Swap
- Read-only search (wait-free)
- No rebalancing overhead (vs. AVL/Red-Black trees)
- Cache-friendly sequential access

**3. Michael-Scott Queue (Thread Pool Work Queues)**
- Lock-free FIFO queue
- Enqueue/dequeue via CAS
- Handles producer-consumer coordination
- Used in database thread pools

**When to Use Lock-Free:**

✅ **Use lock-free when:**
- High contention expected (hot pages, hot indexes)
- Maximum throughput critical
- Deadlock elimination required
- Multi-core scalability essential

❌ **Avoid lock-free when:**
- Complexity budget limited
- Team lacks concurrency expertise
- Debugging difficulty unacceptable
- Simple blocking approach sufficient

**Trade-offs:**
- **Pro:** No deadlocks, better scalability, lower tail latency
- **Con:** Complex implementation, requires memory reclamation (epoch-based GC), ABA problem risks

> 💡 **Deep Dive Available:** For detailed lock-free algorithms, progress guarantees, and implementation patterns, see **[concurrency-primitives.md](concurrency-primitives.md)**

### Consensus and Distributed Coordination

**FLP Impossibility Result** (Fischer, Lynch, Paterson, 1985)

**Theorem:** In an asynchronous distributed system with even one faulty process, no deterministic algorithm can solve consensus.

**Implications for Distributed Databases:**
- Pure asynchronous model cannot guarantee consensus termination
- Real systems assume **partial synchrony** (eventually bounded delays)
- Requires failure detectors or timing assumptions

**How Production Systems Circumvent FLP:**

**1. Raft Consensus (CockroachDB, TiDB, etcd)**
```
Assumption: Partial synchrony + leader election timeouts
Properties:
  - Leader-based coordination
  - Log replication with majority quorum
  - Strong consistency (linearizable)
```

**2. Multi-Paxos (Google Spanner, Cosmos DB)**
```
Assumption: Majority quorum availability
Properties:
  - Proposer-acceptor-learner roles
  - Two-phase protocol (prepare + accept)
  - Optimized with stable leader
```

**3. Leaderless Replication (Cassandra)**
```
Assumption: Quorum reads/writes, eventual consistency
Properties:
  - No consensus required
  - AP in CAP theorem (availability over consistency)
  - Read repair for convergence
```

**Herlihy's Consensus Hierarchy:**

**Why Hardware Primitives Matter:**

| Primitive | Consensus Number | Can Build | Used In |
|-----------|------------------|-----------|---------|
| Read/Write | 1 | Single-process only | Basic variables |
| Test&Set, Fetch&Add | 2 | Two-process coordination | Simple locks, counters |
| **Compare-and-Swap** | **∞** | **Any concurrent object** | **Lock-free data structures** |

**Key Insight:** CAS can implement any concurrent object (universal construction) → Why modern CPUs provide CAS

**Database Implications:**
- **Lock-free indexes** require CAS (Bw-Trees, skip lists)
- **Atomic counters** can use Fetch-and-Add
- **Distributed transactions** require consensus protocols (2PC, Raft)

---

## Buffer Management

### Buffer Pool Architecture

**Purpose:** Cache disk pages in memory to avoid expensive I/O

```
┌────────────────────────────────────────────────┐
│  Buffer Pool (Array of Frames)                 │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐          │
│  │Frame1│ │Frame2│ │Frame3│ │Frame4│ ...      │
│  │Page A│ │Page D│ │Page B│ │Page Z│          │
│  │Dirty │ │Clean │ │Dirty │ │Clean │          │
│  └──────┘ └──────┘ └──────┘ └──────┘          │
├────────────────────────────────────────────────┤
│  Page Table (Hash Map)                         │
│  Page ID → Frame Number + Metadata             │
├────────────────────────────────────────────────┤
│  Free List / Eviction Policy                   │
│  LRU / LRU-K / CLOCK / ARC                     │
└────────────────────────────────────────────────┘
```

### Eviction Policies

**LRU (Least Recently Used):**
```
- Evict page accessed longest time ago
- Implemented via doubly-linked list
- Move to front on access
- Evict from back
Problem: Sequential scans pollute cache
```

**LRU-K (K=2 is common):**
```
- Track last K access timestamps
- Evict page with oldest K-th access
- Protects against one-time sequential scans
- More resistant to cache pollution
```

**CLOCK Algorithm:**
```
- Circular buffer with "clock hand"
- Each page has reference bit
- Access: Set reference bit = 1
- Eviction: Sweep clock hand, clear reference bits
- Evict first page with reference bit = 0
Advantage: Approximates LRU with O(1) update
```

**ARC (Adaptive Replacement Cache):**
```
- Maintains two LRU lists:
  - T1: Pages accessed once recently
  - T2: Pages accessed twice recently
- Dynamically balances between recency and frequency
- Adapts to workload patterns
Used by: PostgreSQL, ZFS
```

### Dirty Page Handling

**Dirty Bit:**
- Set when page modified in buffer pool
- Indicates page must be written back to disk before eviction

**Write Strategies:**
```
1. Write-Through:
   - Write to disk immediately on every update
   - Slow, but simple recovery

2. Write-Back (Deferred Write):
   - Write to disk only on eviction or checkpoint
   - Fast writes, requires WAL for durability

3. Group Commit:
   - Batch multiple transactions' writes
   - Flush WAL once for batch
   - Reduces I/O, improves throughput
```

### Page Replacement Algorithm

```python
def get_page(page_id):
    # Check if page in buffer pool
    frame = page_table.get(page_id)
    
    if frame is not None:
        # Hit: Update eviction policy
        eviction_policy.mark_accessed(frame)
        return buffer_pool[frame]
    
    # Miss: Need to fetch from disk
    frame = get_free_frame()
    
    if frame is None:
        # No free frames, must evict
        victim_frame = eviction_policy.select_victim()
        
        if buffer_pool[victim_frame].is_dirty():
            # Write dirty page to disk (ensure LSN < WAL)
            flush_page_to_disk(victim_frame)
        
        page_table.remove(buffer_pool[victim_frame].page_id)
        frame = victim_frame
    
    # Load page from disk
    buffer_pool[frame] = load_page_from_disk(page_id)
    page_table[page_id] = frame
    eviction_policy.mark_accessed(frame)
    
    return buffer_pool[frame]
```

---

## Compaction Strategies

### LSM Compaction Deep Dive

**Write Amplification:**
```
WA = (Total bytes written to disk) / (User data bytes)

Example: 1 GB user data written
  - L0 flush: 1 GB
  - L0 → L1 compaction: 1 GB + overlaps
  - L1 → L2 compaction: 1 GB + overlaps
  ...
  Total: 10 GB written → WA = 10x
```

**Read Amplification:**
```
RA = Number of SSTables checked per read

Example: 6-level LSM
  - MemTable: 1
  - L0: 4 SSTables (all overlap)
  - L1-L5: 1 SSTable each (binary search)
  Total: 1 + 4 + 5 = 10 reads (worst case)

Bloom filters reduce RA significantly!
```

**Space Amplification:**
```
SA = (Total disk space used) / (Logical data size)

Causes:
  - Multiple versions during compaction
  - Fragmentation
  - Tombstones not yet compacted

Size-Tiered: SA = 2x (waiting for merge)
Leveled: SA = 1.1x (minimal duplication)
```

### Compaction Triggers

**Size-Based Triggers:**
```
Level N threshold = Level (N-1) size × Multiplier

Example (multiplier = 10):
  L0: 4 SSTables
  L1: 10 MB
  L2: 100 MB
  L3: 1 GB
  L4: 10 GB
```

**Time-Based Triggers:**
```
- Periodic compaction (every N hours)
- Ensures tombstones eventually removed
- Handles low-write-rate scenarios
```

**Manual Triggers:**
```
- Operator-initiated (major compaction)
- Compact entire level or entire database
- Expensive but maximizes space reclamation
```

### Compaction Optimizations

**Parallel Compaction:**
```
- Multiple compaction threads
- Non-overlapping key ranges compacted concurrently
- Improves throughput on multi-core systems
```

**Incremental Compaction:**
```
- Break large compaction into smaller chunks
- Reduces tail latency spikes
- Spread I/O over time
```

**Bloom Filter Tuning:**
```
- Larger Bloom filters → lower false positive rate
- Trade-off: Memory vs read I/O
- Per-level tuning: Larger filters for older levels
```

---

## Storage Engine Selection Criteria

### Decision Matrix

| Workload Type | B-Tree | LSM-Tree |
|---------------|--------|----------|
| **Read-Heavy OLTP** | ✅ Excellent | ⚠️ Acceptable with tuning |
| **Write-Heavy Ingestion** | ⚠️ Acceptable | ✅ Excellent |
| **Range Scans** | ✅ Excellent | ✅ Good (if sorted) |
| **Point Lookups** | ✅ Excellent | ⚠️ Multiple levels |
| **Random Updates** | ⚠️ Write amplification | ✅ Append-only |
| **Space Efficiency** | ✅ Minimal overhead | ⚠️ Compaction overhead |
| **Predictable Latency** | ✅ Consistent | ⚠️ Compaction spikes |
| **Flash/SSD Friendly** | ⚠️ Write amplification | ✅ Sequential writes |

### Workload-Specific Recommendations

**OLTP (Online Transaction Processing):**
```
Characteristics:
  - Short transactions
  - Mix of reads/writes
  - Point lookups and small range scans
  - Strong consistency required

Recommendation: B-Tree (InnoDB, PostgreSQL)
  - Predictable read latency
  - MVCC for concurrency
  - Mature tooling and operations
```

**OLAP (Online Analytical Processing):**
```
Characteristics:
  - Long-running queries
  - Large scans and aggregations
  - Read-heavy, batch writes
  - Column-oriented access

Recommendation: Columnar storage or LSM with column families
  - Parquet, ORC file formats
  - Compression-friendly
  - Examples: ClickHouse, Apache Druid
```

**Time-Series Data:**
```
Characteristics:
  - Write-heavy, append-only
  - Time-based partitioning
  - TTL-based expiration
  - Range queries by time

Recommendation: LSM-Tree with time-window compaction
  - Fast ingestion
  - Efficient bulk deletion
  - Examples: InfluxDB, TimescaleDB, Prometheus
```

**Key-Value Stores:**
```
Characteristics:
  - Simple get/put/delete
  - High throughput required
  - Eventual consistency acceptable

Recommendation: LSM-Tree
  - RocksDB, LevelDB
  - Used by: Cassandra, ScyllaDB, CockroachDB
```

**Logging and Event Streams:**
```
Characteristics:
  - Append-only writes
  - Sequential reads
  - Immutable records

Recommendation: Log-structured storage
  - Apache Kafka (log segments)
  - Bitcask (used by Riak KV)
  - Simple, fast, predictable
```

### Hybrid Approaches

**WiredTiger (MongoDB):**
- Supports both B-Tree and LSM-Tree storage
- User chooses per-collection
- B-Tree default for OLTP-like workloads
- LSM option for write-heavy collections

**MyRocks (MySQL with RocksDB):**
- LSM storage with MySQL compatibility
- Better write throughput than InnoDB
- Flash-friendly (reduced write amplification)
- Trade-off: Slightly higher read latency

**TiKV (TiDB storage layer):**
- RocksDB-based (LSM)
- Distributed KV store
- Raft replication
- Optimized for cloud-native deployments

---

## Key Takeaways

### When to Use B-Trees
✅ **Use B-Trees when:**
- Read performance is critical
- Workload is OLTP-style (mixed read/write, short transactions)
- Predictable latency required (avoid compaction spikes)
- Space efficiency important (minimal amplification)
- Mature ecosystem needed (PostgreSQL, MySQL)

### When to Use LSM-Trees
✅ **Use LSM-Trees when:**
- Write throughput is bottleneck
- Flash/SSD storage (minimize random writes)
- Ingestion/logging workloads (append-heavy)
- Eventual consistency acceptable
- Comfortable with compaction tuning

### Universal Principles
1. **No Silver Bullet:** Every storage engine is a trade-off
2. **Measure Your Workload:** Synthetic benchmarks ≠ production
3. **Read/Write/Space Amplification:** Understand the triangle
4. **Bloom Filters Matter:** Critical for LSM read performance
5. **Buffer Pool Tuning:** Often more important than storage engine choice
6. **WAL is Essential:** Durability without sacrificing write speed
7. **MVCC Reduces Contention:** But requires garbage collection
8. **Test Failure Scenarios:** Recovery is where bugs hide

---

## References and Further Reading

### Books
- **"Database Internals"** by Alex Petrov (O'Reilly, 2019) - Primary source
- **"Designing Data-Intensive Applications"** by Martin Kleppmann (O'Reilly, 2017)
- **"Database System Concepts"** by Silberschatz, Korth, Sudarshan (11th Edition)

### Seminal Papers
- **B-Trees:** "Organization and Maintenance of Large Ordered Indices" (Bayer & McCreight, 1972)
- **LSM-Trees:** "The Log-Structured Merge-Tree" (O'Neil et al., 1996)
- **ARIES:** "ARIES: A Transaction Recovery Method" (Mohan et al., 1992)
- **MVCC:** "Multiversion Concurrency Control" (Bernstein & Goodman, 1983)

### Video Resources
- **Computer Science Center Database Courses** (Vadim Tsesko, Ilya Teterin) - Russian language, comprehensive coverage of database internals, distributed systems, CAP theorem, replication strategies
- **CMU Database Systems Course** (Andy Pavlo) - Available on YouTube

### Production Systems Documentation
- **RocksDB Wiki:** https://github.com/facebook/rocksdb/wiki
- **PostgreSQL Internals:** https://www.postgresql.org/docs/current/storage.html
- **InnoDB Architecture:** https://dev.mysql.com/doc/refman/8.0/en/innodb-architecture.html
- **LevelDB Documentation:** https://github.com/google/leveldb/blob/main/doc/index.md

---

**Version:** 1.3.0 | **Updated:** December 2024
