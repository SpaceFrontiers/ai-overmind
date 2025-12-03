# Distributed Systems Algorithms

> Part of [Data-Intensive Systems Architect](SKILL.md) skill

Essential algorithms for building data-intensive distributed systems. Covers consensus, data structures, partitioning, and probabilistic algorithms.

---

## Table of Contents

1. [Distributed Commit Protocols](#distributed-commit-protocols)
2. [Consensus Algorithms](#consensus-algorithms)
3. [Leader Election Algorithms](#leader-election-algorithms)
4. [Partitioning Algorithms](#partitioning-algorithms)
5. [Probabilistic Data Structures](#probabilistic-data-structures)
6. [Data Integrity Algorithms](#data-integrity-algorithms)
7. [Communication Algorithms](#communication-algorithms)
8. [Clock and Ordering Algorithms](#clock-and-ordering-algorithms)
9. [Load Balancing Algorithms](#load-balancing-algorithms)
10. [Quorum Systems](#quorum-systems)
11. [Failure Detection](#failure-detection)
12. [Anti-Entropy Synchronization](#anti-entropy-synchronization)

---

## Distributed Commit Protocols

Protocols for atomic commits across multiple nodes.

### Two-Phase Commit (2PC)

**Purpose:** Ensure all nodes commit or abort a transaction atomically.

**Roles:**
- **Coordinator:** Manages the commit protocol
- **Participants:** Execute transaction locally

**Protocol:**
```
Phase 1: Voting (Prepare)
┌─────────────┐                      ┌──────────────┐
│ Coordinator │──prepare───────────►│ Participants │
│             │◄─vote(yes/no)────────│              │
└─────────────┘                      └──────────────┘

Phase 2: Decision (Commit/Abort)
┌─────────────┐                      ┌──────────────┐
│ Coordinator │──commit/abort──────►│ Participants │
│             │◄─ack─────────────────│              │
└─────────────┘                      └──────────────┘
```

**Algorithm:**
```python
class TwoPhaseCommit:
    def coordinate(self, participants, transaction):
        # Phase 1: Prepare
        votes = []
        for p in participants:
            p.write_to_log("PREPARE", transaction)
            vote = p.prepare(transaction)
            votes.append(vote)
        
        # Decision
        if all(v == "YES" for v in votes):
            decision = "COMMIT"
        else:
            decision = "ABORT"
        
        self.write_to_log(decision, transaction)
        
        # Phase 2: Commit/Abort
        for p in participants:
            if decision == "COMMIT":
                p.commit(transaction)
            else:
                p.abort(transaction)
            p.write_to_log(decision, transaction)
```

**Blocking Problem:** If coordinator fails after sending prepare but before decision, participants are blocked holding locks.

### Three-Phase Commit (3PC)

**Purpose:** Non-blocking commit protocol (under certain failure assumptions).

**Phases:**
1. **CanCommit:** Coordinator asks if participants can commit
2. **PreCommit:** Coordinator tells participants to prepare
3. **DoCommit:** Coordinator tells participants to commit

```
Phase 1: CanCommit
Coordinator ──canCommit?──► Participants
            ◄──yes/no──────

Phase 2: PreCommit (if all yes)
Coordinator ──preCommit───► Participants
            ◄──ack─────────

Phase 3: DoCommit
Coordinator ──doCommit────► Participants
            ◄──haveCommitted─
```

**Key Insight:** PreCommit phase ensures participants know the decision before committing, allowing recovery without coordinator.

**Trade-off:** More messages (3 phases vs 2) but non-blocking under crash failures.

### 2PC vs 3PC

| Aspect | 2PC | 3PC |
|--------|-----|-----|
| Messages | 4N | 6N |
| Blocking | Yes (coordinator failure) | No (crash failures only) |
| Network partitions | Problematic | Still problematic |
| Complexity | Simpler | More complex |
| Use in practice | Common | Rare |

**Note:** In practice, 2PC with timeouts and recovery procedures is more common than 3PC.

---

## Consensus Algorithms

Consensus algorithms enable distributed nodes to agree on a single value despite failures.

### Paxos

**Purpose:** Achieve consensus in asynchronous networks with crash failures.

**Roles:**
- **Proposer:** Proposes values
- **Acceptor:** Votes on proposals (majority required)
- **Learner:** Learns the chosen value

**Two-Phase Protocol:**

```
Phase 1: Prepare
┌──────────┐                    ┌──────────┐
│ Proposer │──prepare(n)──────►│ Acceptors│
│          │◄─promise(n,v)─────│ (majority)│
└──────────┘                    └──────────┘

Phase 2: Accept
┌──────────┐                    ┌──────────┐
│ Proposer │──accept(n,v)─────►│ Acceptors│
│          │◄─accepted(n,v)────│ (majority)│
└──────────┘                    └──────────┘
```

**Key Properties:**
- **Safety:** Only one value can be chosen
- **Liveness:** Eventually a value is chosen (with stable leader)
- **Fault tolerance:** Survives f failures with 2f+1 nodes

**Complexity:** 2 round trips in normal case (can be optimized to 1 with Multi-Paxos)

### Raft

**Purpose:** Understandable consensus algorithm (equivalent to Multi-Paxos).

**Key Insight:** Decompose consensus into:
1. **Leader Election:** Elect a single leader
2. **Log Replication:** Leader replicates log entries
3. **Safety:** Ensure consistency

**Node States:**
```
┌──────────┐  timeout  ┌───────────┐  majority  ┌────────┐
│ Follower │─────────►│ Candidate │──────────►│ Leader │
└──────────┘          └───────────┘            └────────┘
     ▲                      │                       │
     │      higher term     │    higher term        │
     └──────────────────────┴───────────────────────┘
```

**Terms:** Time divided into terms with consecutive integers. Each term has at most one leader.

**Log Replication:**
```python
class RaftNode:
    def append_entries(self, term, leader_id, prev_log_index, 
                       prev_log_term, entries, leader_commit):
        # 1. Reply false if term < currentTerm
        if term < self.current_term:
            return False
        
        # 2. Reply false if log doesn't contain entry at prevLogIndex
        if not self.log_matches(prev_log_index, prev_log_term):
            return False
        
        # 3. Delete conflicting entries and append new ones
        self.log = self.log[:prev_log_index + 1] + entries
        
        # 4. Update commit index
        if leader_commit > self.commit_index:
            self.commit_index = min(leader_commit, len(self.log) - 1)
        
        return True
```

**Comparison with Paxos:**

| Aspect | Paxos | Raft |
|--------|-------|------|
| Understandability | Complex | Simple |
| Leader | Optional | Required |
| Log ordering | Flexible | Strict |
| Implementation | Harder | Easier |
| Performance | Similar | Similar |

### Fast Paxos

**Purpose:** Reduce latency from 2 round trips to 1 in common case.

**Key Idea:** Proposer sends directly to acceptors, bypassing leader.

**Trade-off:** Requires larger quorums (3f+1 instead of 2f+1) to handle collisions.

```
Normal Paxos:  Client → Leader → Acceptors → Learners  (2 RTT)
Fast Paxos:    Client → Acceptors → Learners           (1 RTT)
```

### Byzantine Fault Tolerance (PBFT)

**Purpose:** Consensus with malicious (Byzantine) nodes.

**Requirement:** 3f+1 nodes to tolerate f Byzantine failures.

**Three-Phase Protocol:**
1. **Pre-prepare:** Leader broadcasts request
2. **Prepare:** Nodes broadcast prepare messages
3. **Commit:** Nodes broadcast commit messages

**Use Cases:**
- Blockchain consensus
- Financial systems
- Critical infrastructure

---

## Partitioning Algorithms

### Consistent Hashing

**Purpose:** Distribute data across nodes with minimal redistribution on node changes.

**Key Insight:** Map both keys and nodes to a ring; key belongs to first node clockwise.

```
        Node A (hash=10)
           ╱
          ○
         ╱ ╲
    ○───○   ○───○
   ╱         ╲
Node D        Node B
(hash=80)     (hash=30)
   ╲         ╱
    ○───○───○
         ╲
        Node C (hash=50)

Key with hash=25 → Node B (first node ≥ 25)
Key with hash=55 → Node D (first node ≥ 55, wrapping)
```

**Virtual Nodes:** Each physical node maps to multiple positions for better distribution.

```python
class ConsistentHash:
    def __init__(self, nodes, virtual_nodes=150):
        self.ring = {}
        self.sorted_keys = []
        
        for node in nodes:
            for i in range(virtual_nodes):
                key = self.hash(f"{node}:{i}")
                self.ring[key] = node
                self.sorted_keys.append(key)
        
        self.sorted_keys.sort()
    
    def get_node(self, key):
        if not self.ring:
            return None
        
        hash_key = self.hash(key)
        
        # Binary search for first node >= hash_key
        for ring_key in self.sorted_keys:
            if ring_key >= hash_key:
                return self.ring[ring_key]
        
        # Wrap around to first node
        return self.ring[self.sorted_keys[0]]
    
    def hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
```

**Properties:**
- Adding/removing node affects only K/N keys (K=total keys, N=nodes)
- Virtual nodes improve load balance
- Used by: DynamoDB, Cassandra, Riak, Memcached

### Rendezvous Hashing (HRW)

**Purpose:** Alternative to consistent hashing with simpler implementation.

**Algorithm:** For each key, compute hash with each node; choose highest.

```python
def get_node(key, nodes):
    return max(nodes, key=lambda node: hash(f"{key}:{node}"))
```

**Advantages over Consistent Hashing:**
- Simpler implementation
- No ring maintenance
- Deterministic node selection

**Disadvantage:** O(N) per lookup vs O(log N) for consistent hashing.

### Range Partitioning

**Purpose:** Partition data by key ranges for range queries.

```
Partition 1: A-G    → Node 1
Partition 2: H-N    → Node 2
Partition 3: O-T    → Node 3
Partition 4: U-Z    → Node 4
```

**Challenge:** Hot spots if data distribution is skewed.

**Solution:** Dynamic splitting based on size/load.

---

## Probabilistic Data Structures

### Bloom Filter

**Purpose:** Space-efficient set membership testing with false positives.

**Structure:** Bit array + k hash functions.

```
Insert "apple":
  h1("apple") = 2  →  set bit 2
  h2("apple") = 5  →  set bit 5
  h3("apple") = 9  →  set bit 9

Bit array: [0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0]
                ↑       ↑         ↑
```

**Query:** Check if ALL k bits are set.

```python
class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = [False] * size
    
    def add(self, item):
        for seed in range(self.num_hashes):
            index = self._hash(item, seed) % self.size
            self.bits[index] = True
    
    def might_contain(self, item):
        for seed in range(self.num_hashes):
            index = self._hash(item, seed) % self.size
            if not self.bits[index]:
                return False  # Definitely not in set
        return True  # Possibly in set (may be false positive)
    
    def _hash(self, item, seed):
        return hash(f"{item}:{seed}")
```

**False Positive Rate:**

```
FPR ≈ (1 - e^(-kn/m))^k

Where:
  k = number of hash functions
  n = number of elements
  m = number of bits

Optimal k = (m/n) * ln(2) ≈ 0.7 * (m/n)
```

**Use Cases:**
- Database query optimization (avoid disk reads)
- Distributed caches (check before network call)
- Spell checkers
- Network routers (packet filtering)

**Variants:**
- **Counting Bloom Filter:** Support deletions (use counters instead of bits)
- **Cuckoo Filter:** Better space efficiency, supports deletion
- **Quotient Filter:** Cache-friendly, supports merging

### HyperLogLog

**Purpose:** Estimate cardinality (count distinct) with minimal memory.

**Key Insight:** The position of the leftmost 1-bit in random hashes indicates log₂(n).

**Algorithm:**
1. Hash each element
2. Use first few bits to select a register (bucket)
3. Count leading zeros in remaining bits
4. Store maximum leading zeros per register
5. Estimate: harmonic mean of 2^(register values)

```python
class HyperLogLog:
    def __init__(self, precision=14):  # 2^14 = 16384 registers
        self.p = precision
        self.m = 1 << precision  # number of registers
        self.registers = [0] * self.m
        self.alpha = self._get_alpha()
    
    def add(self, item):
        h = hash(item)
        # First p bits determine register
        register = h & (self.m - 1)
        # Remaining bits: count leading zeros + 1
        remaining = h >> self.p
        self.registers[register] = max(
            self.registers[register],
            self._leading_zeros(remaining) + 1
        )
    
    def count(self):
        # Harmonic mean of 2^register values
        indicator = sum(2.0 ** -r for r in self.registers)
        estimate = self.alpha * self.m * self.m / indicator
        return int(estimate)
    
    def _leading_zeros(self, value):
        if value == 0:
            return 64 - self.p
        zeros = 0
        while (value & 1) == 0:
            zeros += 1
            value >>= 1
        return zeros
```

**Properties:**
- Memory: ~1.5 bytes per register (12 KB for 1.6% error)
- Error: 1.04 / √m (m = number of registers)
- Mergeable: Union of HLLs by taking max of each register

**Use Cases:**
- Unique visitor counting
- Database query planning
- Network traffic analysis

### Count-Min Sketch

**Purpose:** Estimate frequency of items in a stream.

**Structure:** 2D array with d rows (hash functions) and w columns.

```python
class CountMinSketch:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]
    
    def add(self, item, count=1):
        for row in range(self.depth):
            col = self._hash(item, row) % self.width
            self.table[row][col] += count
    
    def estimate(self, item):
        # Return minimum across all rows (least overestimated)
        return min(
            self.table[row][self._hash(item, row) % self.width]
            for row in range(self.depth)
        )
```

**Properties:**
- Always overestimates (never underestimates)
- Error decreases with width, probability decreases with depth
- Space: O(d × w)

**Use Cases:**
- Heavy hitters detection
- Network traffic monitoring
- Database query frequency

---

## Data Integrity Algorithms

### Merkle Trees

**Purpose:** Efficient verification of data integrity in distributed systems.

**Structure:** Binary tree where each non-leaf node is hash of its children.

```
                    Root Hash
                   /         \
              Hash(0-1)    Hash(2-3)
              /     \      /     \
          Hash0   Hash1  Hash2   Hash3
            |       |      |       |
          Data0   Data1  Data2   Data3
```

**Verification:** To verify Data2, only need Hash3 and Hash(0-1).

```python
class MerkleTree:
    def __init__(self, data_blocks):
        self.leaves = [self.hash(block) for block in data_blocks]
        self.tree = self._build_tree(self.leaves)
    
    def _build_tree(self, nodes):
        if len(nodes) == 1:
            return nodes
        
        # Pad if odd number of nodes
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        
        # Build parent level
        parents = []
        for i in range(0, len(nodes), 2):
            parent = self.hash(nodes[i] + nodes[i+1])
            parents.append(parent)
        
        return self._build_tree(parents) + nodes
    
    def get_root(self):
        return self.tree[0]
    
    def get_proof(self, index):
        """Return sibling hashes needed to verify leaf at index."""
        proof = []
        n = len(self.leaves)
        offset = len(self.tree) - n
        
        while n > 1:
            sibling = index ^ 1  # XOR to get sibling
            proof.append((self.tree[offset + sibling], index % 2))
            index //= 2
            n //= 2
            offset -= n
        
        return proof
    
    def verify(self, leaf_hash, proof, root):
        current = leaf_hash
        for sibling_hash, is_right in proof:
            if is_right:
                current = self.hash(sibling_hash + current)
            else:
                current = self.hash(current + sibling_hash)
        return current == root
    
    def hash(self, data):
        return hashlib.sha256(data.encode()).hexdigest()
```

**Properties:**
- Verification: O(log n) hashes
- Proof size: O(log n) hashes
- Detect which blocks differ between two trees

**Use Cases:**
- Git (content-addressable storage)
- Blockchain (transaction verification)
- Certificate Transparency
- IPFS, BitTorrent

---

## Communication Algorithms

### Gossip Protocol (Epidemic Protocol)

**Purpose:** Disseminate information in large-scale distributed systems.

**Mechanism:** Each node periodically exchanges state with random peers.

```python
class GossipNode:
    def __init__(self, node_id, peers):
        self.node_id = node_id
        self.peers = peers
        self.state = {}
        self.version = {}
    
    def update(self, key, value):
        self.state[key] = value
        self.version[key] = self.version.get(key, 0) + 1
    
    def gossip_round(self):
        # Select random peer
        peer = random.choice(self.peers)
        
        # Exchange state
        peer_state, peer_version = peer.receive_gossip(
            self.state, self.version
        )
        
        # Merge: keep higher versions
        for key in set(self.state) | set(peer_state):
            if peer_version.get(key, 0) > self.version.get(key, 0):
                self.state[key] = peer_state[key]
                self.version[key] = peer_version[key]
    
    def receive_gossip(self, sender_state, sender_version):
        # Merge incoming state
        for key in sender_state:
            if sender_version.get(key, 0) > self.version.get(key, 0):
                self.state[key] = sender_state[key]
                self.version[key] = sender_version[key]
        
        return self.state, self.version
```

**Convergence:** Information reaches all N nodes in O(log N) rounds.

**Variants:**
- **Push:** Send updates to random peers
- **Pull:** Request updates from random peers
- **Push-Pull:** Exchange in both directions (fastest)

**Properties:**
- Scalable: O(log N) convergence
- Fault-tolerant: Works with node failures
- Eventually consistent
- Simple to implement

**Use Cases:**
- Failure detection (heartbeats)
- Membership management
- Database replication (Cassandra, Riak)
- Cluster state synchronization

---

## Leader Election Algorithms

Leader election designates one node as coordinator for distributed tasks.

### Bully Algorithm

**Purpose:** Elect the node with highest ID as leader.

**Message Types:**
- **Election:** Starts election process
- **Answer:** Acknowledges election message
- **Leader:** Declares winner

**Algorithm:**
```python
class BullyNode:
    def __init__(self, node_id, all_nodes):
        self.node_id = node_id
        self.all_nodes = all_nodes
        self.leader = max(all_nodes)  # Initially highest ID
    
    def detect_leader_failure(self):
        """Called when leader doesn't respond to heartbeat."""
        self.start_election()
    
    def start_election(self):
        higher_nodes = [n for n in self.all_nodes if n > self.node_id]
        
        if not higher_nodes:
            # I am the highest, declare myself leader
            self.declare_leader()
            return
        
        # Send election message to all higher nodes
        responses = []
        for node in higher_nodes:
            response = self.send_election(node)
            if response:
                responses.append(response)
        
        if not responses:
            # No higher node responded, I become leader
            self.declare_leader()
        else:
            # Wait for leader announcement from higher node
            self.wait_for_leader()
    
    def receive_election(self, from_node):
        if from_node < self.node_id:
            # Send answer and start my own election
            self.send_answer(from_node)
            self.start_election()
    
    def declare_leader(self):
        self.leader = self.node_id
        for node in self.all_nodes:
            if node != self.node_id:
                self.send_leader(node, self.node_id)
```

**Complexity:**
- Best case: O(N) messages (highest node detects failure)
- Worst case: O(N²) messages (lowest node detects failure)

### Ring Algorithm

**Purpose:** Elect leader using logical ring topology.

**Algorithm:**
1. Node detecting failure sends Election message with its ID around ring
2. Each node adds its ID to message if higher, forwards message
3. When message returns to initiator, highest ID is leader
4. Initiator sends Leader message around ring

```
Node 3 detects failure:
  → [3] → Node 5 adds ID → [3,5] → Node 2 → [3,5] → Node 7 → [3,5,7]
  → back to Node 3: Leader is 7
  → Leader announcement travels around ring
```

**Complexity:** O(3N - 1) messages (always)

### Comparison

| Algorithm | Best Case | Worst Case | Topology |
|-----------|-----------|------------|----------|
| Bully | O(N) | O(N²) | Full mesh |
| Ring | O(3N) | O(3N) | Logical ring |
| Raft Election | O(N) | O(N) | Full mesh |

**When to Use:**
- **Bully:** Simple systems, infrequent elections
- **Ring:** Predictable message count needed
- **Raft:** Production systems, integrated with consensus

---

## Clock and Ordering Algorithms

### Lamport Clocks

**Purpose:** Establish partial ordering of events in distributed systems.

**Rules:**
1. Before each event, increment local clock
2. When sending message, include clock value
3. When receiving, set clock to max(local, received) + 1

```python
class LamportClock:
    def __init__(self):
        self.time = 0
    
    def tick(self):
        """Local event occurred."""
        self.time += 1
        return self.time
    
    def send(self):
        """Prepare to send message."""
        self.time += 1
        return self.time
    
    def receive(self, sender_time):
        """Received message with sender's timestamp."""
        self.time = max(self.time, sender_time) + 1
        return self.time
```

**Property:** If a → b (a happened before b), then C(a) < C(b).

**Limitation:** C(a) < C(b) does NOT imply a → b (could be concurrent).

### Vector Clocks

**Purpose:** Detect causality and concurrency between events.

**Structure:** Each node maintains a vector of clocks, one per node.

```python
class VectorClock:
    def __init__(self, node_id, num_nodes):
        self.node_id = node_id
        self.clock = [0] * num_nodes
    
    def tick(self):
        """Local event occurred."""
        self.clock[self.node_id] += 1
        return self.clock.copy()
    
    def send(self):
        """Prepare to send message."""
        self.clock[self.node_id] += 1
        return self.clock.copy()
    
    def receive(self, sender_clock):
        """Received message with sender's vector clock."""
        for i in range(len(self.clock)):
            self.clock[i] = max(self.clock[i], sender_clock[i])
        self.clock[self.node_id] += 1
        return self.clock.copy()
    
    @staticmethod
    def compare(vc1, vc2):
        """
        Returns:
          -1 if vc1 < vc2 (vc1 happened before vc2)
           1 if vc1 > vc2 (vc2 happened before vc1)
           0 if concurrent
        """
        less = all(a <= b for a, b in zip(vc1, vc2))
        greater = all(a >= b for a, b in zip(vc1, vc2))
        
        if less and not greater:
            return -1
        elif greater and not less:
            return 1
        else:
            return 0  # Concurrent
```

**Properties:**
- VC(a) < VC(b) ⟺ a → b (happened before)
- VC(a) || VC(b) ⟺ a and b are concurrent

**Use Cases:**
- Conflict detection in replicated databases
- Causal ordering of messages
- Version vectors in distributed storage

---

## Load Balancing Algorithms

Distribute workload across multiple servers or resources.

### Static Algorithms

**Round Robin (RR):**
- Distribute requests sequentially to each server
- Simple but ignores server capacity and current load

**Weighted Round Robin (WRR):**
- Assign weights based on server capacity
- Higher weight = more requests
- Requires prior knowledge of server capabilities

```python
class WeightedRoundRobin:
    def __init__(self, servers):
        # servers = [("server1", 5), ("server2", 3), ("server3", 2)]
        self.servers = servers
        self.weights = [w for _, w in servers]
        self.current = 0
        self.current_weight = 0
        self.max_weight = max(self.weights)
        self.gcd_weight = self._gcd_list(self.weights)
    
    def next_server(self):
        while True:
            self.current = (self.current + 1) % len(self.servers)
            if self.current == 0:
                self.current_weight -= self.gcd_weight
                if self.current_weight <= 0:
                    self.current_weight = self.max_weight
            
            if self.weights[self.current] >= self.current_weight:
                return self.servers[self.current][0]
```

### Dynamic Algorithms

**Least Connections:**
- Route to server with fewest active connections
- Adapts to varying request durations

**Weighted Least Connections (WLC):**
- Combines connection count with server capacity
- Score = connections / weight
- Route to lowest score

```python
class WeightedLeastConnections:
    def __init__(self, servers):
        self.servers = {name: {"weight": w, "connections": 0} 
                       for name, w in servers}
    
    def next_server(self):
        best = min(self.servers.items(), 
                   key=lambda x: x[1]["connections"] / x[1]["weight"])
        best[1]["connections"] += 1
        return best[0]
    
    def release(self, server):
        self.servers[server]["connections"] -= 1
```

### Comparison

| Algorithm | Adapts to Load | Server Awareness | Complexity |
|-----------|----------------|------------------|------------|
| Round Robin | No | No | O(1) |
| Weighted RR | No | Yes (static) | O(1) |
| Least Connections | Yes | No | O(N) |
| Weighted LC | Yes | Yes | O(N) |

---

## Quorum Systems

Ensure consistency in replicated data stores.

### Basic Quorum Rules

For N replicas with R read quorum and W write quorum:

**Strict Quorum (Strong Consistency):**
```
R + W > N    (read and write quorums overlap)
W > N/2      (prevents concurrent conflicting writes)
```

**Partial Quorum (Eventual Consistency):**
```
R + W ≤ N    (quorums may not overlap)
```

### Common Configurations

| Config | R | W | N | Consistency | Availability |
|--------|---|---|---|-------------|--------------|
| Read-optimized | 1 | N | N | Strong | Low write availability |
| Write-optimized | N | 1 | N | Strong | Low read availability |
| Balanced | ⌈(N+1)/2⌉ | ⌈(N+1)/2⌉ | N | Strong | Balanced |
| Eventual | 1 | 1 | N | Eventual | High |

### Probabilistically Bounded Staleness (PBS)

For partial quorums, PBS quantifies staleness probability:

```
Given: N replicas, R read quorum, W write quorum
If R + W ≤ N, reads may return stale data

PBS shows: Even with R=1, W=1, N=3:
- 99.9% of reads return data < 10ms stale
- Significant latency benefits over strict quorums
```

**Key Insight:** Eventual consistency is often "good enough" because:
- Writes propagate quickly in practice
- Most reads hit recently-written data
- Staleness window is typically milliseconds

### Implementation Example

```python
class QuorumStore:
    def __init__(self, replicas, read_quorum, write_quorum):
        self.replicas = replicas
        self.R = read_quorum
        self.W = write_quorum
        self.N = len(replicas)
    
    def write(self, key, value, version):
        # Send to all replicas, wait for W acknowledgments
        acks = 0
        for replica in self.replicas:
            if replica.write(key, value, version):
                acks += 1
                if acks >= self.W:
                    return True
        return False
    
    def read(self, key):
        # Read from R replicas, return highest version
        responses = []
        for replica in self.replicas:
            response = replica.read(key)
            if response:
                responses.append(response)
                if len(responses) >= self.R:
                    break
        
        # Return value with highest version
        return max(responses, key=lambda x: x.version)
```

---

## Failure Detection

Detect node failures in distributed systems.

### Heartbeat-Based Detection

**Simple Heartbeat:**
- Nodes send periodic "I'm alive" messages
- Missing heartbeats trigger failure suspicion

```python
class HeartbeatDetector:
    def __init__(self, timeout_ms=5000):
        self.timeout = timeout_ms
        self.last_heartbeat = {}
    
    def receive_heartbeat(self, node_id):
        self.last_heartbeat[node_id] = time.now()
    
    def is_alive(self, node_id):
        if node_id not in self.last_heartbeat:
            return False
        elapsed = time.now() - self.last_heartbeat[node_id]
        return elapsed < self.timeout
```

**Challenge:** Fixed timeout is problematic:
- Too short → false positives (slow network)
- Too long → slow failure detection

### Phi Accrual Failure Detector

**Purpose:** Adaptive failure detection based on heartbeat history.

**Key Idea:** Instead of binary alive/dead, output a suspicion level (φ).

```python
class PhiAccrualDetector:
    def __init__(self, threshold=8, window_size=1000):
        self.threshold = threshold
        self.window_size = window_size
        self.heartbeat_intervals = []
        self.last_heartbeat = None
    
    def receive_heartbeat(self):
        now = time.now()
        if self.last_heartbeat:
            interval = now - self.last_heartbeat
            self.heartbeat_intervals.append(interval)
            if len(self.heartbeat_intervals) > self.window_size:
                self.heartbeat_intervals.pop(0)
        self.last_heartbeat = now
    
    def phi(self):
        """Calculate suspicion level."""
        if not self.heartbeat_intervals:
            return 0
        
        now = time.now()
        time_since_last = now - self.last_heartbeat
        
        # Assume normal distribution of intervals
        mean = statistics.mean(self.heartbeat_intervals)
        std = statistics.stdev(self.heartbeat_intervals)
        
        # Probability that we should have received heartbeat by now
        # φ = -log10(1 - CDF(time_since_last))
        z = (time_since_last - mean) / std
        p = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        
        if p >= 1:
            return float('inf')
        return -math.log10(1 - p)
    
    def is_alive(self):
        return self.phi() < self.threshold
```

**Properties:**
- φ = 1 → 10% chance of false positive
- φ = 2 → 1% chance
- φ = 3 → 0.1% chance
- Adapts to network conditions automatically

**Used by:** Cassandra, Akka

---

## Anti-Entropy Synchronization

Repair data inconsistencies between replicas.

### Read Repair

Fix inconsistencies during read operations:

```python
def read_with_repair(key, replicas, quorum):
    responses = []
    for replica in replicas[:quorum]:
        responses.append(replica.read(key))
    
    # Find most recent version
    latest = max(responses, key=lambda x: x.version)
    
    # Repair stale replicas (async)
    for i, response in enumerate(responses):
        if response.version < latest.version:
            replicas[i].write_async(key, latest.value, latest.version)
    
    return latest
```

### Active Anti-Entropy (Merkle Trees)

Periodically compare and sync entire datasets:

```
Replica A                    Replica B
    │                            │
    ▼                            ▼
Build Merkle Tree           Build Merkle Tree
    │                            │
    └──── Exchange Roots ────────┘
              │
              ▼
         Roots Match?
         ├── Yes: Done
         └── No: Compare subtrees
                    │
                    ▼
              Find differing leaves
                    │
                    ▼
              Sync differing keys
```

**Advantages:**
- Efficient: Only exchange O(log N) hashes to find differences
- Complete: Catches all inconsistencies
- Background: Doesn't impact foreground operations

**Used by:** Riak, Cassandra, DynamoDB

---

## Algorithm Selection Guide

| Problem | Algorithm | Trade-off |
|---------|-----------|-----------|
| Consensus (crash failures) | Raft | Simplicity over flexibility |
| Consensus (Byzantine) | PBFT | Security over performance |
| Fast consensus | Fast Paxos | Latency over quorum size |
| Data partitioning | Consistent Hashing | Minimal redistribution |
| Set membership | Bloom Filter | Space vs false positives |
| Cardinality estimation | HyperLogLog | Accuracy vs memory |
| Frequency estimation | Count-Min Sketch | Space vs overestimation |
| Data integrity | Merkle Tree | Efficient verification |
| Information dissemination | Gossip | Scalability vs latency |
| Event ordering | Vector Clocks | Causality detection |
| Load distribution | Weighted Least Conn | Adapts to server load |
| Replica consistency | Quorum (R+W>N) | Consistency vs availability |
| Failure detection | Phi Accrual | Adaptive to network |
| Data sync | Anti-Entropy | Background consistency |

---

## References

- Lamport, L. (1998). "The Part-Time Parliament" (Paxos)
- Lamport, L. (2006). "Fast Paxos"
- Ongaro, D., Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm" (Raft)
- Karger, D. et al. (1997). "Consistent Hashing and Random Trees"
- Bloom, B. (1970). "Space/Time Trade-offs in Hash Coding with Allowable Errors"
- Flajolet, P. et al. (2007). "HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm"
- Merkle, R. (1987). "A Digital Signature Based on a Conventional Encryption Function"
- Demers, A. et al. (1987). "Epidemic Algorithms for Replicated Database Maintenance"
- Fidge, C. (1988). "Timestamps in Message-Passing Systems That Preserve the Partial Ordering"
- Garcia-Molina, H. (1982). "Elections in a Distributed Computing System" (Bully Algorithm)
- Bailis, P. et al. (2012). "Probabilistically Bounded Staleness for Practical Partial Quorums"
- Hayashibara, N. et al. (2004). "The φ Accrual Failure Detector"
- Jiménez-Peris, R. et al. (2003). "Are Quorums an Alternative for Data Replication?"

### Video Lectures

- **Computer Science Center** - Distributed Databases (Vadim Tsesko) - 2PC, Raft, Paxos
- **Yandex for ML** - Bloom Filter and Count-Min Sketch (probabilistic data structures)
- **Computer Science Seminar** - Raft Algorithm (Maxim Babenko)
- **Lektorium** - Object Replication to Database Replication (Fernando Pedone)

**Version:** 1.5.0 | **Updated:** December 2024
