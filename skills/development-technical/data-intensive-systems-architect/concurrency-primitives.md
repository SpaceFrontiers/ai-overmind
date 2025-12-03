# Concurrency Primitives and Synchronization Algorithms

> Part of [Data-Intensive Systems Architect](SKILL.md) skill

Low-level synchronization primitives, concurrent programming patterns, and lock-free algorithms essential for modern database implementations. Based on Taubenfeld and Raynal's concurrent programming research.

---

## Table of Contents

1. [Progress Guarantees and Non-Blocking Algorithms](#progress-guarantees-and-non-blocking-algorithms)
2. [Atomic Primitives and Hardware Support](#atomic-primitives-and-hardware-support)
3. [Mutual Exclusion Algorithms](#mutual-exclusion-algorithms)
4. [Lock-Free Data Structures](#lock-free-data-structures)
5. [Synchronization Patterns](#synchronization-patterns)
6. [Consensus and Impossibility Results](#consensus-and-impossibility-results)
7. [Software Transactional Memory](#software-transactional-memory)
8. [Distributed Atomic Registers](#distributed-atomic-registers)
9. [Production Database Applications](#production-database-applications)

---

## Progress Guarantees and Non-Blocking Algorithms

### The Hierarchy of Progress Guarantees

Modern concurrent systems can be classified by the progress guarantees they provide:

```
┌─────────────────────────────────────────────────┐
│  BLOCKING (Traditional)                         │
│  - May deadlock                                 │
│  - Requires deadlock detection/prevention       │
│  - Examples: Locks, semaphores, monitors        │
├─────────────────────────────────────────────────┤
│  OBSTRUCTION-FREE (Weakest Non-Blocking)        │
│  - Progress only when running alone             │
│  - No progress guarantee under contention       │
│  - Rarely used in practice                      │
├─────────────────────────────────────────────────┤
│  LOCK-FREE (System-Wide Progress)               │
│  - At least one thread makes progress          │
│  - System never stalls                          │
│  - Individual threads may starve                │
│  - Examples: Michael-Scott queue, Bw-Trees     │
├─────────────────────────────────────────────────┤
│  WAIT-FREE (Strongest Guarantee)                │
│  - Every thread makes progress in bounded steps │
│  - No starvation possible                       │
│  - Hardest to implement, highest overhead       │
│  - Examples: Wait-free snapshot, fetch&add     │
└─────────────────────────────────────────────────┘
```

### Definitions

**Blocking Algorithm:**
- Uses locks or other blocking synchronization
- A thread may be indefinitely delayed by other threads
- May deadlock if locks acquired in wrong order
- **Trade-off:** Simple to reason about, but poor worst-case behavior

**Obstruction-Free Algorithm:**
- Guarantees progress for a thread running in isolation
- If other threads stop interfering, any thread will complete
- Typical pattern: Retry loop with exponential backoff
- **Trade-off:** Weak guarantee, useful mainly as theoretical concept

**Lock-Free Algorithm:**
- Guarantees system-wide progress
- At least one thread makes progress in every execution step
- Some threads may be starved, but system as a whole advances
- **Trade-off:** Good throughput, potential individual thread starvation

**Wait-Free Algorithm:**
- Guarantees per-thread progress in bounded steps
- Every thread completes its operation in finite time
- Strongest possible guarantee
- **Trade-off:** Complex implementation, often with "helping" mechanism

### When to Use Each

| Progress Guarantee | Use When | Avoid When | Database Example |
|-------------------|----------|------------|------------------|
| **Blocking** | Simplicity preferred, low contention | Real-time requirements, high contention | PostgreSQL buffer manager |
| **Obstruction-Free** | Research/theory | Production systems | (Rare in databases) |
| **Lock-Free** | High contention, maximum throughput | Complexity budget limited | Microsoft Hekaton Bw-Trees |
| **Wait-Free** | Real-time, bounded latency critical | Performance overhead unacceptable | Fetch&Add for counters |

### Boosting Techniques

**From Obstruction-Free to Lock-Free:**
- Add global timestamp or version counter
- Threads help each other complete operations
- Example: Transform obstruction-free snapshot to lock-free

**From Lock-Free to Wait-Free:**
- Implement helping mechanism
- Fast path for uncontended case
- Slow path with helping for contended case
- Example: Universal construction with helping

---

## Atomic Primitives and Hardware Support

### Hardware Atomic Operations

Modern CPUs provide atomic read-modify-write operations that are the foundation of concurrent programming.

#### Compare-and-Swap (CAS)

**Pseudocode:**
```c
bool compare_and_swap(int *addr, int expected, int new_value) {
    atomic {
        if (*addr == expected) {
            *addr = new_value;
            return true;
        }
        return false;
    }
}
```

**Hardware Support:**
- **x86/x64:** `CMPXCHG` instruction
- **ARM:** `LDREX`/`STREX` (Load-Linked/Store-Conditional)
- **POWER:** `LWARX`/`STWCX`

**Example Usage (Lock-Free Increment):**
```c
void lock_free_increment(atomic_int *counter) {
    int old_value, new_value;
    do {
        old_value = atomic_load(counter);
        new_value = old_value + 1;
    } while (!compare_and_swap(counter, old_value, new_value));
}
```

**Why CAS is Powerful:**
- Can implement any concurrent object (universal construction)
- Consensus number = ∞ (Herlihy's hierarchy)
- Foundation for lock-free algorithms

#### Load-Linked/Store-Conditional (LL/SC)

**Semantics:**
```c
int load_linked(int *addr) {
    // Mark address for monitoring
    return *addr;
}

bool store_conditional(int *addr, int value) {
    // Succeeds only if no other store to addr since LL
    if (addr_not_modified_since_LL) {
        *addr = value;
        return true;
    }
    return false;
}
```

**Advantages over CAS:**
- No ABA problem (detects any intermediate modification)
- More flexible than CAS

**ABA Problem with CAS:**
```
Thread 1: Reads A from location
Thread 2: Changes A → B → A
Thread 1: CAS succeeds (thinks nothing changed!)
```

**Solution:** Use LL/SC or add version tags

#### Fetch-and-Add

**Semantics:**
```c
int fetch_and_add(int *addr, int delta) {
    atomic {
        int old_value = *addr;
        *addr = old_value + delta;
        return old_value;
    }
}
```

**Properties:**
- Wait-free (bounded time)
- Consensus number = 2 (can only coordinate 2 threads for consensus)
- Used in: Ticket locks, counters, barriers

#### Test-and-Set

**Semantics:**
```c
bool test_and_set(bool *addr) {
    atomic {
        bool old_value = *addr;
        *addr = true;
        return old_value;
    }
}
```

**Properties:**
- Simplest atomic primitive
- Consensus number = 2
- Used in: Basic spinlocks

### Herlihy's Consensus Hierarchy

**Consensus Number:** Maximum number of threads for which an object can solve consensus

```
┌────────────────────────────────────────────┐
│  Level ∞ (Universal)                       │
│  - Compare-and-Swap (CAS)                  │
│  - Load-Linked/Store-Conditional (LL/SC)   │
│  → Can build ANY concurrent object         │
├────────────────────────────────────────────┤
│  Level 2                                   │
│  - Test-and-Set                            │
│  - Fetch-and-Add                           │
│  - Swap                                    │
│  → Can coordinate 2 threads                │
├────────────────────────────────────────────┤
│  Level 1                                   │
│  - Read/Write registers                    │
│  - Queues, Stacks                          │
│  → Cannot solve consensus for even 2       │
└────────────────────────────────────────────┘
```

**Key Insight:** Cannot build Level n+1 object from only Level n objects (in wait-free manner)

**Practical Implications:**
- Why CAS is ubiquitous in concurrent programming
- Why databases need CAS for lock-free indexes
- Why hardware provides CAS/LL/SC

**Database Relevance:**
- **Lock-free indexes:** Require CAS (Bw-Trees, skip lists)
- **Atomic counters:** Can use Fetch-and-Add
- **Distributed consensus:** Requires higher-level protocols (Paxos, Raft)

---

## Mutual Exclusion Algorithms

### Classic Algorithms (Educational)

#### Peterson's Algorithm (2 Threads)

**Properties:**
- Uses only read/write registers
- Deadlock-free, starvation-free
- Demonstrates fundamental synchronization concepts

**Code:**
```c
bool flag[2] = {false, false};  // Interest flags
int turn = 0;                    // Tiebreaker

// Thread i (where i ∈ {0, 1})
void lock(int i) {
    int j = 1 - i;
    flag[i] = true;      // Announce interest
    turn = j;            // Give priority to other
    while (flag[j] && turn == j) {
        // Spin while other interested and their turn
    }
}

void unlock(int i) {
    flag[i] = false;
}
```

**Why It Works:**
- If both threads want entry, one wins on `turn` tiebreaker
- If only one thread wants entry, it enters immediately
- Mutual exclusion proven by case analysis

#### Lamport's Bakery Algorithm (n Threads)

**Intuition:** Like taking a number at a deli counter

**Properties:**
- First-come, first-served (FIFO fairness)
- Works with safe registers (weakest consistency)
- No bound on ticket numbers (unbounded space)

**Code:**
```c
int ticket[n] = {0, ...};      // Ticket numbers
bool choosing[n] = {false, ...}; // Taking ticket flag

void lock(int i) {
    choosing[i] = true;
    // Take maximum ticket + 1
    ticket[i] = 1 + max(ticket[0], ..., ticket[n-1]);
    choosing[i] = false;
    
    // Wait for smaller tickets
    for (int j = 0; j < n; j++) {
        while (choosing[j]) { /* wait */ }
        while (ticket[j] != 0 && 
               (ticket[j] < ticket[i] || 
                (ticket[j] == ticket[i] && j < i))) {
            // Wait if j has smaller ticket or wins tiebreaker
        }
    }
}

void unlock(int i) {
    ticket[i] = 0;
}
```

**Key Insight:** Can build atomic operations from non-atomic operations!

### Production-Grade Scalable Locks

#### MCS Lock (Mellor-Crummey and Scott)

**Design Goals:**
- **Local spinning:** Each thread spins on its own memory location
- **Cache-friendly:** Minimal cache line bouncing
- **FIFO fairness:** Request order preserved
- **Scalable:** Performance improves with more cores

**Structure:**
```c
struct qnode {
    struct qnode *next;
    bool waiting;
};

struct mcs_lock {
    atomic_ptr tail;  // Tail of waiting queue
};
```

**Algorithm:**
```c
void mcs_acquire(struct mcs_lock *lock, struct qnode *mynode) {
    mynode->next = NULL;
    mynode->waiting = true;
    
    // Atomically add self to tail
    struct qnode *prev = atomic_swap(&lock->tail, mynode);
    
    if (prev != NULL) {
        // Queue was non-empty, link in
        prev->next = mynode;
        // Spin on local variable (cache-friendly!)
        while (mynode->waiting) { /* spin */ }
    }
    // Lock acquired
}

void mcs_release(struct mcs_lock *lock, struct qnode *mynode) {
    if (mynode->next == NULL) {
        // Check if we're last in queue
        if (compare_and_swap(&lock->tail, mynode, NULL)) {
            return; // Queue now empty
        }
        // Wait for next to link in
        while (mynode->next == NULL) { /* spin */ }
    }
    // Wake up next waiter
    mynode->next->waiting = false;
}
```

**Advantages:**
- Each thread spins on its own cache line
- No cache line bouncing (unlike simple spinlock)
- Scales to hundreds of cores

**Database Usage:**
- **PostgreSQL:** Considered for LWLocks on high-core systems
- **MySQL:** Evaluated for InnoDB latch manager
- **Research systems:** Common in experimental databases

#### CLH Lock (Craig, Landin, Hagersten)

**Similar to MCS but simpler:**
- Implicit queue (linked by predecessors)
- Spin on predecessor's node
- Slightly less fair than MCS but simpler

**Key Difference from MCS:**
- MCS: Explicit queue with next pointers
- CLH: Implicit queue, spin on predecessor

### Readers-Writer Locks

**Problem:** Allow multiple concurrent readers OR one exclusive writer

**Variants:**
1. **Reader preference:** Readers have priority (may starve writers)
2. **Writer preference:** Writers have priority (may starve readers)
3. **Fair:** FIFO order regardless of type

**Simple Reader-Preference Implementation:**
```c
struct rwlock {
    int readers;        // Number of active readers
    mutex lock;         // Protects readers count
    semaphore resource; // Controls access to resource
};

void read_lock(struct rwlock *rw) {
    mutex_lock(&rw->lock);
    rw->readers++;
    if (rw->readers == 1) {
        semaphore_wait(&rw->resource); // First reader locks
    }
    mutex_unlock(&rw->lock);
}

void read_unlock(struct rwlock *rw) {
    mutex_lock(&rw->lock);
    rw->readers--;
    if (rw->readers == 0) {
        semaphore_signal(&rw->resource); // Last reader unlocks
    }
    mutex_unlock(&rw->lock);
}

void write_lock(struct rwlock *rw) {
    semaphore_wait(&rw->resource); // Exclusive access
}

void write_unlock(struct rwlock *rw) {
    semaphore_signal(&rw->resource);
}
```

**Database Usage:**
- **Buffer pool pages:** Multiple readers, exclusive writer
- **Catalog cache:** Read-heavy workload
- **Lock-free alternatives:** RCU (Read-Copy-Update) in Linux kernel, used in some databases

---

## Lock-Free Data Structures

### Michael-Scott Lock-Free Queue

**Most famous lock-free data structure** - Used in production systems worldwide

**Structure:**
```c
struct node {
    void *data;
    atomic_ptr next;
};

struct queue {
    atomic_ptr head;
    atomic_ptr tail;
    node *dummy;  // Sentinel node
};
```

**Enqueue (Producer):**
```c
void enqueue(struct queue *q, void *data) {
    node *new_node = malloc(sizeof(node));
    new_node->data = data;
    new_node->next = NULL;
    
    while (true) {
        node *last = atomic_load(&q->tail);
        node *next = atomic_load(&last->next);
        
        // Check tail still valid
        if (last == atomic_load(&q->tail)) {
            if (next == NULL) {
                // Tail is really last, try to link in
                if (compare_and_swap(&last->next, NULL, new_node)) {
                    // Success! Try to swing tail
                    compare_and_swap(&q->tail, last, new_node);
                    return;
                }
            } else {
                // Tail behind, help advance it
                compare_and_swap(&q->tail, last, next);
            }
        }
    }
}
```

**Dequeue (Consumer):**
```c
void *dequeue(struct queue *q) {
    while (true) {
        node *first = atomic_load(&q->head);
        node *last = atomic_load(&q->tail);
        node *next = atomic_load(&first->next);
        
        if (first == atomic_load(&q->head)) {
            if (first == last) {
                if (next == NULL) {
                    return NULL; // Queue empty
                }
                // Tail behind, help advance it
                compare_and_swap(&q->tail, last, next);
            } else {
                void *data = next->data;
                // Try to swing head to next
                if (compare_and_swap(&q->head, first, next)) {
                    free(first); // Reclaim old dummy
                    return data;
                }
            }
        }
    }
}
```

**Key Features:**
- **Helping mechanism:** Threads help each other advance tail
- **Lock-free:** Always at least one thread makes progress
- **ABA-safe:** Careful pointer manipulation avoids ABA
- **Linearizable:** Strong consistency guarantee

**Database Applications:**
- **Thread pool work queues:** Dispatch tasks to workers
- **Log flush queue:** Batch log records
- **Network I/O queues:** Buffer incoming requests

### Lock-Free Skip List (MemTable)

**Used in:** LevelDB, RocksDB MemTable

**Structure:**
```
Level 3:  Head -----------------> 40 -----------------> NULL
Level 2:  Head -------> 20 -----> 40 -------> 60 ----> NULL
Level 1:  Head -> 10 -> 20 -> 30->40 -> 50 -> 60 -----> NULL
Level 0:  Head -> 10 -> 20 -> 30->40 -> 50 -> 60 -> 70-> NULL
```

**Properties:**
- **Probabilistic balancing:** No rebalancing needed (unlike B-Trees)
- **Lock-free search:** Read-only traversal
- **CAS-based insertion:** Atomic pointer updates
- **Multiple levels:** Logarithmic search time

**Search (Lock-Free, Wait-Free):**
```c
node *find(skiplist *list, int key) {
    node *curr = list->head;
    for (int level = MAX_LEVEL; level >= 0; level--) {
        while (curr->next[level] != NULL && 
               curr->next[level]->key < key) {
            curr = curr->next[level];
        }
    }
    curr = curr->next[0];
    if (curr != NULL && curr->key == key) {
        return curr;
    }
    return NULL;
}
```

**Insert (Lock-Free via CAS):**
```c
void insert(skiplist *list, int key, void *value) {
    node *update[MAX_LEVEL + 1];
    node *curr = list->head;
    
    // Find insertion point at each level
    for (int level = MAX_LEVEL; level >= 0; level--) {
        while (curr->next[level] != NULL && 
               curr->next[level]->key < key) {
            curr = curr->next[level];
        }
        update[level] = curr;
    }
    
    // Create new node with random height
    int height = random_level();
    node *new_node = create_node(key, value, height);
    
    // Link in new node from bottom up
    for (int level = 0; level <= height; level++) {
        do {
            new_node->next[level] = update[level]->next[level];
        } while (!compare_and_swap(&update[level]->next[level],
                                   new_node->next[level],
                                   new_node));
    }
}
```

**Why Skip Lists for MemTable:**
- ✅ Lock-free reads during writes
- ✅ No rebalancing overhead (vs. AVL/Red-Black trees)
- ✅ Cache-friendly sequential access
- ✅ Simple to implement correctly

### Bw-Trees (Lock-Free B-Trees)

**Developed by:** Microsoft Research (Levandoski, Lomet, Sengupta, 2013)  
**Used in:** Microsoft Hekaton (SQL Server In-Memory OLTP)

**Key Innovation:** Delta updates instead of in-place modifications

**Architecture:**
```
Mapping Table (CAS-based):
┌─────────────────────────────┐
│ Page ID → Physical Address  │
│  1000   →  0x7f8b4c00       │
│  1001   →  0x7f8b5000       │
└─────────────────────────────┘
         ↓
Delta Chain:
┌──────────────┐
│ Insert Delta │ ← Latest
├──────────────┤
│ Delete Delta │
├──────────────┤
│  Base Page   │ ← Oldest
└──────────────┘
```

**Operations:**

**Search:**
```c
value *search(bwtree *tree, int key) {
    page_id pid = tree->root_pid;
    
    while (true) {
        page *p = mapping_table_lookup(pid);
        
        // Traverse delta chain
        while (p->type == DELTA) {
            if (p->delta_type == INSERT && p->key == key) {
                return p->value;
            }
            if (p->delta_type == DELETE && p->key == key) {
                return NULL;
            }
            p = p->base;
        }
        
        // Search base page
        if (p->is_leaf) {
            return base_page_search(p, key);
        }
        
        // Navigate to child
        pid = base_page_find_child(p, key);
    }
}
```

**Insert (Lock-Free):**
```c
void insert(bwtree *tree, page_id pid, int key, value *val) {
    while (true) {
        page *old_page = mapping_table_lookup(pid);
        
        // Create insert delta
        page *delta = create_insert_delta(key, val, old_page);
        
        // Atomically swing mapping table pointer
        if (compare_and_swap(&mapping_table[pid], old_page, delta)) {
            // Success!
            
            // Check if consolidation needed
            if (delta_chain_too_long(delta)) {
                consolidate_page(pid);
            }
            return;
        }
        // CAS failed, retry
    }
}
```

**Consolidation (Background):**
```c
void consolidate_page(page_id pid) {
    page *old_page = mapping_table_lookup(pid);
    
    // Create new base page from delta chain
    page *new_base = apply_all_deltas(old_page);
    
    // Atomically replace
    compare_and_swap(&mapping_table[pid], old_page, new_base);
    
    // Old pages will be reclaimed by epoch-based GC
}
```

**Performance Benefits:**
- **10x faster** than latched B-Trees under high contention
- **Lock-free reads:** Never blocked by writes
- **Append-only:** Flash-friendly (no in-place updates)
- **Cache-friendly:** Delta chains often fit in cache

**Trade-offs:**
- More complex than traditional B-Trees
- Garbage collection required (epoch-based reclamation)
- Delta chains add read overhead if too long

---

## Synchronization Patterns

### Barrier Synchronization

**Problem:** Coordinate multiple threads to reach a common point before proceeding

**Applications in Databases:**
- Parallel query execution (synchronize between pipeline stages)
- Parallel sort (synchronize between merge phases)
- Snapshot coordination (all threads see consistent state)

#### Centralized Barrier (Simple)

```c
struct barrier {
    int count;              // Threads arrived
    int total;              // Total threads
    mutex lock;
    condition_variable cv;
};

void barrier_wait(struct barrier *b) {
    mutex_lock(&b->lock);
    b->count++;
    
    if (b->count == b->total) {
        // Last thread, release all
        b->count = 0;
        condition_broadcast(&b->cv);
    } else {
        // Wait for others
        while (b->count != 0) {
            condition_wait(&b->cv, &b->lock);
        }
    }
    mutex_unlock(&b->lock);
}
```

**Problem:** Contention on central counter

#### Tree-Based Barrier (Scalable)

**Structure:** Binary tree of sub-barriers

```
      Root (4 threads)
       /    \
    L1a(2) L1b(2)
    / \    / \
  T0 T1  T2 T3
```

**Algorithm:**
1. **Arrival phase:** Threads arrive at leaves, propagate up tree
2. **Wakeup phase:** Root signals, wakeups propagate down tree

**Benefits:**
- Logarithmic contention (vs. linear for centralized)
- Cache-friendly (local communication)

#### Combining Tree Barrier

**Key Idea:** Combine operations while traversing tree

**Applications:**
- Parallel aggregation (SUM, COUNT)
- Distributed reduction
- Used in parallel databases for aggregate queries

### Producer-Consumer Pattern

**Classic synchronization problem** - Multiple producers, multiple consumers

**Bounded Buffer Implementation (Semaphores):**
```c
struct bounded_buffer {
    void *items[N];
    int in, out;
    semaphore empty;  // Counts empty slots
    semaphore full;   // Counts full slots
    mutex lock;       // Protects buffer
};

void init_buffer(struct bounded_buffer *buf) {
    buf->in = buf->out = 0;
    semaphore_init(&buf->empty, N);  // N empty slots
    semaphore_init(&buf->full, 0);   // 0 full slots
    mutex_init(&buf->lock);
}

void produce(struct bounded_buffer *buf, void *item) {
    semaphore_wait(&buf->empty);  // Wait for empty slot
    mutex_lock(&buf->lock);
    
    buf->items[buf->in] = item;
    buf->in = (buf->in + 1) % N;
    
    mutex_unlock(&buf->lock);
    semaphore_signal(&buf->full);  // Signal full slot
}

void *consume(struct bounded_buffer *buf) {
    semaphore_wait(&buf->full);  // Wait for full slot
    mutex_lock(&buf->lock);
    
    void *item = buf->items[buf->out];
    buf->out = (buf->out + 1) % N;
    
    mutex_unlock(&buf->lock);
    semaphore_signal(&buf->empty);  // Signal empty slot
    return item;
}
```

**Database Applications:**
- Log writer (producers: transactions, consumer: log flusher)
- Network I/O (producers: connections, consumers: worker threads)
- Batch processing (producers: data ingest, consumers: processors)

### Dining Philosophers Problem

**Classic deadlock scenario** (Dijkstra, 1965)

**Problem Setup:**
- 5 philosophers sitting at round table
- 5 chopsticks (one between each pair)
- Need 2 chopsticks to eat
- Can only pick up adjacent chopsticks

**Naive Solution (Deadlocks!):**
```c
void philosopher(int i) {
    while (true) {
        think();
        pickup_left_chopstick(i);
        pickup_right_chopstick(i);
        eat();
        putdown_chopsticks(i);
    }
}
```

**Problem:** All philosophers pick up left chopstick simultaneously → deadlock!

**Solution 1: Asymmetry**
```c
void philosopher(int i) {
    while (true) {
        think();
        if (i % 2 == 0) {
            pickup_left(i);
            pickup_right(i);
        } else {
            pickup_right(i);
            pickup_left(i);
        }
        eat();
        putdown_chopsticks(i);
    }
}
```

**Solution 2: Resource Hierarchy**
- Number chopsticks 1-5
- Always pick up lower-numbered chopstick first
- Breaks circular wait condition

**Database Relevance:**
- **Multiple lock acquisition:** Must acquire in consistent order
- **Deadlock prevention:** Lock ordering strategies
- **Transaction management:** Wound-wait, wait-die protocols

---

## Consensus and Impossibility Results

### FLP Impossibility Theorem

**Statement:** In an asynchronous distributed system with even one faulty process, no deterministic algorithm can solve consensus.

**Formal Definition:**
- **Asynchronous system:** No bounds on message delays or processing speeds
- **Faulty process:** May crash (fail-stop)
- **Consensus:** All correct processes must agree on same value

**Proof Sketch:**
1. Show there exists an initial bivalent configuration (can decide either 0 or 1)
2. Prove system can remain bivalent indefinitely
3. Therefore, no guarantee of termination

**Implications for Databases:**

❌ **What FLP Means:**
- Cannot build perfectly reliable distributed database in pure asynchronous model
- Distributed transactions cannot guarantee termination
- Replication protocols must make assumptions

✅ **How Real Systems Work:**
- **Paxos/Raft:** Assume partial synchrony (eventually bounded delays)
- **Timeouts:** Assume failure detector (may be inaccurate)
- **TrueTime (Spanner):** Assume bounded clock skew

**Practical Workarounds:**

**1. Partial Synchrony:**
- System eventually becomes synchronous
- Allows algorithms like Raft, Multi-Paxos

**2. Failure Detectors:**
- **Omega (Ω):** Eventually accurate leader election
- Sufficient to solve consensus
- Implemented via heartbeats + timeouts

**3. Randomization:**
- Ben-Or's randomized consensus
- Terminates with probability 1
- Not used in production (non-determinism problematic)

### Herlihy's Universal Construction

**Theorem:** Any concurrent object can be built from consensus + read/write registers in a wait-free manner.

**Key Insight:** Consensus is "universal" - it's the most powerful synchronization primitive

**Construction Sketch:**
```c
struct universal_object {
    consensus *decide;      // Consensus object
    operation *pending[n];  // Pending operations
    state *current_state;   // Current object state
};

result apply_operation(struct universal_object *obj, operation *op) {
    int slot = thread_id;
    obj->pending[slot] = op;
    
    // Propose to execute my operation next
    int winner = consensus_propose(obj->decide, slot);
    
    // Execute winner's operation
    operation *winning_op = obj->pending[winner];
    result res = execute(obj->current_state, winning_op);
    
    // Update state
    obj->current_state = compute_new_state(obj->current_state, winning_op);
    
    return res;
}
```

**Why This Matters:**
- Compare-and-Swap has consensus number ∞
- Therefore, CAS + registers can build any concurrent object
- Foundation for lock-free data structures

**Helping Mechanism:**
```c
// If my CAS fails, help winner complete their operation
if (!compare_and_swap(&obj->state, old, new)) {
    // Someone else won, help them
    help_complete_operation(obj);
}
```

### Consensus in Distributed Databases

#### Raft Consensus (CockroachDB, TiDB, etcd)

**Key Properties:**
- **Leader-based:** Single leader coordinates
- **Log replication:** Leader replicates log to followers
- **Majority quorum:** Requires n/2 + 1 nodes for progress

**How Raft Circumvents FLP:**
- **Leader election timeout:** Assumes bounds on message delays
- **Heartbeats:** Failure detection mechanism
- **Partial synchrony:** Eventually bounded delays

**Phases:**
1. **Leader Election:** Elect single leader via voting
2. **Log Replication:** Leader appends entries, waits for majority
3. **Safety:** Committed entries never lost

#### Multi-Paxos (Google Spanner, Cosmos DB)

**Key Properties:**
- **Proposer-Acceptor-Learner roles**
- **Majority quorum** for acceptance
- **Two-phase protocol** (prepare + accept)

**Optimizations:**
- **Multi-Paxos:** Elect stable leader, skip prepare phase
- **Fast Paxos:** Skip leader for uncontended operations

---

## Software Transactional Memory

### STM Fundamentals

**Goal:** Make concurrent programming easier by providing transactional semantics

**API:**
```c
atomic {
    // Code here executes atomically
    x = read(shared_var1);
    write(shared_var2, x + 1);
}
```

**Compiler Transforms To:**
```c
stm_begin();
x = stm_read(&shared_var1);
stm_write(&shared_var2, x + 1);
if (!stm_commit()) {
    // Conflict detected, retry
    goto retry;
}
```

### STM Consistency: Opacity

**Opacity:** Stronger than serializability for databases

**Requirements:**
1. **Committed transactions:** Appear to execute serially
2. **Aborted transactions:** Never see inconsistent state (even internally)

**Why Opacity Matters:**
```c
// Without opacity, aborted transaction might crash:
atomic {
    int x = read(var1);
    int y = read(var2);
    int result = 1000 / (x - y);  // May divide by zero!
    write(var3, result);
}
```

If aborted transaction sees inconsistent state (x == y when they should differ), crash occurs even though transaction will abort.

### TL2 Algorithm (Dice, Shalev, Shavit)

**Time-based locking** - Most influential STM algorithm

**Global Version Clock:**
```c
atomic_int global_clock = 0;
```

**Per-Location Versioned Lock:**
```c
struct versioned_lock {
    int version;      // Even: unlocked, Odd: locked
    mutex lock;
};
```

**Read Phase:**
```c
int stm_read(struct location *loc) {
    int start_time = atomic_load(&global_clock);
    int value, version;
    
    do {
        version = atomic_load(&loc->version);
        if (version & 1) continue;  // Locked, retry
        value = loc->data;
        memory_fence();
    } while (version != atomic_load(&loc->version));
    
    // Log read for validation
    add_to_read_set(loc, version);
    return value;
}
```

**Write Phase (Deferred):**
```c
void stm_write(struct location *loc, int value) {
    // Buffer write locally
    add_to_write_set(loc, value);
}
```

**Commit Phase:**
```c
bool stm_commit() {
    // 1. Lock all locations in write set
    for (each loc in write_set) {
        lock(loc);  // Set version to odd
    }
    
    // 2. Increment global clock
    int commit_time = fetch_and_add(&global_clock, 1);
    
    // 3. Validate read set
    for (each loc in read_set) {
        if (loc->version > start_time && 
            loc not in write_set) {
            unlock_all();
            return false;  // Conflict!
        }
    }
    
    // 4. Write and unlock
    for (each loc in write_set) {
        loc->data = buffered_value;
        loc->version = commit_time;  // Even, unlocked
    }
    
    return true;
}
```

**Key Features:**
- **Read invisibility:** Reads don't modify shared state
- **Optimistic:** Assume no conflicts, validate at commit
- **Timestamp-based:** Global clock orders transactions

### STM vs. Database Transactions

| Property | STM | Database ACID |
|----------|-----|---------------|
| **Durability** | No (in-memory) | Yes (WAL, disk) |
| **Atomicity** | Yes | Yes |
| **Isolation** | Opacity | Serializability |
| **Consistency** | Application-defined | Schema + constraints |
| **Persistence** | Volatile | Durable |
| **Scale** | Single machine | Distributed |

**STM in Databases:**
- **Hekaton:** Uses optimistic MVCC similar to STM
- **HyPer:** STM-inspired transaction management
- **Research systems:** Many experiments with STM

---

## Distributed Atomic Registers

### Lamport's Register Hierarchy

**Safe Register:**
- Weakest guarantee
- Concurrent read may return any value in register's domain
- Only useful as building block

**Regular Register:**
- Concurrent read returns old value or value being written
- No "new-old inversion" (time travel)
- Stronger than safe, weaker than atomic

**Atomic Register (Linearizable):**
- Strongest guarantee
- All operations appear to execute at single point in time
- Total order consistent with real-time ordering

### ABD Algorithm (Attiya, Bar-Noy, Dolev)

**Problem:** Build atomic read/write register from unreliable message-passing network

**System Model:**
- n nodes, f < n/2 may crash
- Asynchronous network (unbounded delays)
- Majority quorum: ⌈(n+1)/2⌉

**Data Structure:**
```c
struct replicated_register {
    int value;
    int timestamp;
};
```

**Write Protocol:**
```c
void abd_write(int new_value) {
    // Phase 1: Get latest timestamp
    int max_ts = query_majority_for_max_timestamp();
    
    // Phase 2: Write with incremented timestamp
    broadcast_to_majority(new_value, max_ts + 1);
}
```

**Read Protocol:**
```c
int abd_read() {
    // Phase 1: Query majority for value with max timestamp
    (value, ts) = query_majority_for_max();
    
    // Phase 2: Write back to majority (read repair)
    broadcast_to_majority(value, ts);
    
    return value;
}
```

**Why It Works:**
- Majority quorums always intersect
- Timestamps provide total order
- Read writeback ensures future reads see latest

**Database Applications:**
- **Quorum reads/writes** (Cassandra, Dynamo)
- **Chain replication** (stronger than ABD, linearizable)
- **Distributed consensus** (Paxos uses similar quorum ideas)

---

## Production Database Applications

### Microsoft Hekaton (SQL Server In-Memory OLTP)

**Architecture:**
- **Bw-Trees** for indexes (lock-free)
- **Optimistic MVCC** for transactions (STM-inspired)
- **Lock-free data structures** throughout

**Performance:**
- **10-100x faster** than disk-based SQL Server
- Scales linearly to 100+ cores
- No latch contention

**Key Techniques:**
- Bw-Trees with delta updates
- Compile transactions to native code
- Epoch-based garbage collection

### RocksDB MemTable (LevelDB/RocksDB)

**Implementation:**
- **Lock-free skip list** for MemTable
- Concurrent inserts without locks
- Read-only searches (wait-free)

**Code:** Based on Google's open-source LevelDB

**Performance Characteristics:**
- Millions of inserts/second per core
- Scales to many threads
- Cache-friendly access patterns

### CockroachDB (Raft Consensus)

**Architecture:**
- **Raft replication** per range (shard)
- **Multi-Paxos-like** coordination
- **Timestamp-based MVCC**

**How Raft Used:**
1. Every write proposed to Raft leader
2. Leader replicates to majority
3. Committed writes applied to state machine

**Scalability:**
- Thousands of Raft groups per cluster
- Horizontal scaling via sharding
- Strong consistency (linearizable)

### PostgreSQL (Lightweight Locks)

**Latch Implementation:**
- **LWLocks** (Lightweight Locks) for internal synchronization
- **Semaphores** (POSIX) for waiting
- **Spinlocks** for very short critical sections

**Buffer Manager:**
- **Pin/Unpin** protocol for pages
- **Shared/Exclusive** latches on buffer descriptors
- Considered MCS locks for high-core scalability

---

## Key Takeaways

### When to Use Each Technique

**Lock-Based (Traditional):**
- ✅ Simplicity critical
- ✅ Low to moderate contention
- ✅ Well-understood by team
- ✅ Deadlock detection acceptable
- **Examples:** PostgreSQL, MySQL InnoDB

**Lock-Free (Modern):**
- ✅ High contention expected
- ✅ Maximum throughput critical
- ✅ Deadlock elimination required
- ✅ Team has concurrency expertise
- **Examples:** Hekaton, RocksDB

**Wait-Free (Specialized):**
- ✅ Real-time guarantees needed
- ✅ Bounded latency critical
- ✅ Individual thread starvation unacceptable
- ❌ High implementation complexity
- **Examples:** Fetch&Add counters, some STM

### Progress Guarantee Selection

| Workload | Contention | Recommended | Rationale |
|----------|------------|-------------|-----------|
| OLTP transactions | Low-Medium | 2PL or MVCC (blocking) | Simple, well-tested |
| In-memory indexes | High | Lock-free (Bw-Trees, skip lists) | Eliminate latch bottleneck |
| Counters | Any | Wait-free (Fetch&Add) | Simple, bounded time |
| Thread pools | Medium-High | Lock-free queues | Avoid coordination overhead |
| Distributed coordination | N/A | Consensus (Raft/Paxos) | Require agreement despite failures |

### Fundamental Principles

1. **Hardware Matters:** CAS is universal, Test&Set is not
2. **FLP is Real:** Distributed consensus needs assumptions
3. **Helping Helps:** Lock-free algorithms often need helping mechanisms
4. **Garbage Collection:** Lock-free requires memory reclamation strategy
5. **Validation Costs:** Optimistic approaches pay at commit time
6. **Local Spinning:** Cache-friendly spinning (MCS, CLH) scales better
7. **ABA Problems:** Be aware of pointer reuse issues with CAS

---

## References and Further Reading

### Books
- **"Synchronization Algorithms and Concurrent Programming"** by Gadi Taubenfeld (Pearson, 2006)
- **"Concurrent Programming: Algorithms, Principles, and Foundations"** by Michel Raynal (Springer, 2013)
- **"The Art of Multiprocessor Programming"** by Maurice Herlihy and Nir Shavit (Morgan Kaufmann, 2012)

### Seminal Papers

**Impossibility and Theory:**
- Fischer, Lynch, Paterson (1985) "Impossibility of Distributed Consensus with One Faulty Process"
- Herlihy (1991) "Wait-Free Synchronization"
- Herlihy (1991) "Wait-Free Hierarchy"

**Lock-Free Data Structures:**
- Michael, Scott (1996) "Simple, Fast, and Practical Non-Blocking and Blocking Concurrent Queue Algorithms"
- Harris (2001) "A Pragmatic Implementation of Non-Blocking Linked-Lists"
- Levandoski, Lomet, Sengupta (2013) "The Bw-Tree: A B-tree for New Hardware Platforms"

**Scalable Locking:**
- Mellor-Crummey, Scott (1991) "Algorithms for Scalable Synchronization on Shared-Memory Multiprocessors"
- Craig (1993) "Building FIFO and Priority-Queuing Spin Locks from Atomic Swap"

**Software Transactional Memory:**
- Dice, Shalev, Shavit (2006) "Transactional Locking II"
- Shavit, Touitou (1995) "Software Transactional Memory"

**Distributed Atomic Registers:**
- Attiya, Bar-Noy, Dolev (1995) "Sharing Memory Robustly in Message-Passing Systems"
- Lamport (1986) "On Interprocess Communication"

**Consensus:**
- Lamport (1998) "The Part-Time Parliament" (Paxos)
- Ongaro, Ousterhout (2014) "In Search of an Understandable Consensus Algorithm" (Raft)

### Video Lectures
- **MIT 6.824 (Distributed Systems)** - Robert Morris, Frans Kaashoek
- **CMU 15-418 (Parallel Architecture)** - Kayvon Fatahalian
- **Computer Science Center (Parallel Programming)** - Evgeny Kalishenko (Russian)

---

**Version:** 1.3.0 | **Updated:** December 2024
