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
    - [Read Repair](#read-repair)
    - [Hinted Handoff](#hinted-handoff)
    - [Active Anti-Entropy (Merkle Trees)](#active-anti-entropy-merkle-trees)
    - [Gossip-Based Anti-Entropy](#gossip-based-anti-entropy)
    - [Version Vectors and Conflict Detection](#version-vectors-and-conflict-detection)
    - [Conflict-Free Replicated Data Types (CRDTs)](#conflict-free-replicated-data-types-crdts)

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

Distribute workload across multiple servers or resources. Load balancing is fundamental to building scalable, highly available systems.

### Load Balancer Types by OSI Layer

#### Layer 4 (L4) Load Balancing - Transport Layer

**Operates at:** TCP/UDP level using IP addresses and port numbers.

**Characteristics:**
- Fast processing (no packet inspection)
- Protocol-agnostic (works with any TCP/UDP application)
- Performs Network Address Translation (NAT)
- Cannot make content-based routing decisions

**Use cases:**
- High-throughput scenarios where content inspection not needed
- Database connection pooling
- Gaming servers, VoIP

```
Client ──► L4 Load Balancer ──► Server
           (IP:Port routing)
```

#### Layer 7 (L7) Load Balancing - Application Layer

**Operates at:** HTTP/HTTPS level, inspecting request content.

**Characteristics:**
- Content-based routing (URL, headers, cookies)
- SSL/TLS termination
- Request manipulation (header injection, URL rewriting)
- Higher latency due to packet inspection

**Use cases:**
- Microservices routing by path/header
- A/B testing, canary deployments
- API gateway functionality
- Session affinity via cookies

```
Client ──► L7 Load Balancer ──► Service A (/api/users)
           (content routing)  ──► Service B (/api/orders)
                              ──► Service C (/api/products)
```

#### Global Server Load Balancing (GSLB)

**Operates at:** DNS level across multiple data centers/regions.

**Characteristics:**
- Geographic traffic distribution
- Disaster recovery and failover
- Latency-based routing (nearest data center)
- Health monitoring across regions

**Use cases:**
- Multi-region deployments
- CDN routing
- Disaster recovery
- Compliance (data residency requirements)

```
User (Europe) ──► DNS/GSLB ──► EU Data Center
User (US)     ──► DNS/GSLB ──► US Data Center
User (Asia)   ──► DNS/GSLB ──► Asia Data Center
```

### L4 vs L7 vs GSLB Comparison

| Aspect | L4 | L7 | GSLB |
|--------|----|----|------|
| OSI Layer | Transport | Application | DNS |
| Routing basis | IP, Port | Content, Headers | Geography, Health |
| Performance | Fastest | Moderate | Varies |
| SSL termination | No | Yes | No |
| Content inspection | No | Yes | No |
| Scope | Single site | Single site | Multi-site |

---

### Static Algorithms

Algorithms that don't consider real-time server state.

#### Round Robin (RR)

**Mechanism:** Distribute requests sequentially to each server in rotation.

```python
class RoundRobin:
    def __init__(self, servers):
        self.servers = servers
        self.current = 0
    
    def next_server(self):
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server
```

**Pros:** Simple, O(1), fair distribution for homogeneous servers
**Cons:** Ignores server capacity and current load

#### Weighted Round Robin (WRR)

**Mechanism:** Assign weights based on server capacity; higher weight = more requests.

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
    
    def _gcd_list(self, numbers):
        from math import gcd
        from functools import reduce
        return reduce(gcd, numbers)
```

**Use case:** Heterogeneous server fleet with known capacities.

#### IP Hash (Source Hash)

**Mechanism:** Hash client IP to deterministically select server.

```python
class IPHash:
    def __init__(self, servers):
        self.servers = servers
    
    def get_server(self, client_ip):
        hash_value = hash(client_ip)
        return self.servers[hash_value % len(self.servers)]
```

**Pros:** Session affinity without cookies, deterministic
**Cons:** Uneven distribution if IP distribution skewed, problematic with NAT

#### URL Hash

**Mechanism:** Hash request URL to route to consistent server (good for caching).

```python
class URLHash:
    def __init__(self, servers):
        self.servers = servers
    
    def get_server(self, url):
        hash_value = hash(url)
        return self.servers[hash_value % len(self.servers)]
```

**Use case:** Maximize cache hit rates by routing same URLs to same servers.

---

### Dynamic Algorithms

Algorithms that adapt to real-time server state.

#### Least Connections

**Mechanism:** Route to server with fewest active connections.

```python
class LeastConnections:
    def __init__(self, servers):
        self.connections = {server: 0 for server in servers}
    
    def next_server(self):
        server = min(self.connections, key=self.connections.get)
        self.connections[server] += 1
        return server
    
    def release(self, server):
        self.connections[server] -= 1
```

**Pros:** Adapts to varying request durations
**Cons:** O(N) lookup, doesn't account for server capacity

#### Weighted Least Connections (WLC)

**Mechanism:** Combines connection count with server capacity.

```python
class WeightedLeastConnections:
    def __init__(self, servers):
        # servers = [("server1", 5), ("server2", 3), ("server3", 2)]
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

**Score formula:** `connections / weight` (lower is better)

#### Least Response Time

**Mechanism:** Route to server with lowest average response time + fewest connections.

```python
class LeastResponseTime:
    def __init__(self, servers):
        self.servers = {server: {
            "connections": 0,
            "avg_response_time": 0.0,
            "request_count": 0
        } for server in servers}
    
    def next_server(self):
        def score(item):
            name, stats = item
            # Combine response time and connection count
            return stats["avg_response_time"] * (stats["connections"] + 1)
        
        best = min(self.servers.items(), key=score)
        best[1]["connections"] += 1
        return best[0]
    
    def record_response(self, server, response_time):
        stats = self.servers[server]
        stats["connections"] -= 1
        # Exponential moving average
        alpha = 0.3
        stats["avg_response_time"] = (
            alpha * response_time + 
            (1 - alpha) * stats["avg_response_time"]
        )
        stats["request_count"] += 1
```

**Use case:** Heterogeneous backends with varying performance characteristics.

#### Resource-Based (Adaptive)

**Mechanism:** Route based on real-time server resource metrics (CPU, memory, etc.).

```python
class ResourceBasedLoadBalancer:
    def __init__(self, servers):
        self.servers = {server: {
            "cpu_load": 0.0,
            "memory_used": 0.0,
            "connections": 0
        } for server in servers}
    
    def update_metrics(self, server, cpu_load, memory_used):
        self.servers[server]["cpu_load"] = cpu_load
        self.servers[server]["memory_used"] = memory_used
    
    def next_server(self):
        def health_score(item):
            name, metrics = item
            # Lower score = healthier server
            return (
                metrics["cpu_load"] * 0.5 +
                metrics["memory_used"] * 0.3 +
                metrics["connections"] * 0.2
            )
        
        best = min(self.servers.items(), key=health_score)
        best[1]["connections"] += 1
        return best[0]
```

**Requires:** Agent on each server reporting metrics to load balancer.

---

### Advanced Algorithms

#### Power of Two Random Choices (P2C)

**Mechanism:** Pick two random servers, choose the one with fewer connections.

**Key insight:** Exponential improvement over pure random with minimal overhead.

```python
import random

class PowerOfTwoChoices:
    def __init__(self, servers):
        self.servers = servers
        self.connections = {server: 0 for server in servers}
    
    def next_server(self):
        # Pick two random servers
        if len(self.servers) < 2:
            choice = self.servers[0]
        else:
            candidates = random.sample(self.servers, 2)
            # Choose the one with fewer connections
            choice = min(candidates, key=lambda s: self.connections[s])
        
        self.connections[choice] += 1
        return choice
    
    def release(self, server):
        self.connections[server] -= 1
```

**Mathematical property:**
- Random allocation: max load = O(log N / log log N)
- Power of Two Choices: max load = O(log log N) — exponential improvement!

**When to use:**
- Distributed load balancers (sidecars) without shared state
- Service mesh architectures
- When Least Connections is impractical

**Trade-off:** Slightly worse than Least Connections but works in distributed settings without coordination.

#### Consistent Hashing with Bounded Loads

**Mechanism:** Consistent hashing + load limits to prevent hotspots.

```python
class BoundedLoadConsistentHash:
    def __init__(self, servers, load_factor=1.25):
        self.servers = servers
        self.load_factor = load_factor
        self.connections = {server: 0 for server in servers}
        self.ring = self._build_ring()
    
    def _build_ring(self):
        # Simplified: use consistent hashing ring
        ring = {}
        for i, server in enumerate(self.servers):
            ring[hash(server) % 360] = server
        return dict(sorted(ring.items()))
    
    def _max_load(self):
        total = sum(self.connections.values())
        avg = total / len(self.servers) if self.servers else 0
        return max(1, int(avg * self.load_factor))
    
    def get_server(self, key):
        hash_val = hash(key) % 360
        max_load = self._max_load()
        
        # Find first server on ring that isn't overloaded
        for ring_pos in sorted(self.ring.keys()):
            if ring_pos >= hash_val:
                server = self.ring[ring_pos]
                if self.connections[server] < max_load:
                    self.connections[server] += 1
                    return server
        
        # Wrap around
        for ring_pos in sorted(self.ring.keys()):
            server = self.ring[ring_pos]
            if self.connections[server] < max_load:
                self.connections[server] += 1
                return server
        
        # All overloaded, pick least loaded
        return min(self.connections, key=self.connections.get)
```

**Use case:** Caching systems where you want cache locality but need to prevent hotspots.

---

### Session Affinity (Sticky Sessions)

Ensure requests from the same client go to the same server.

#### Affinity vs Persistence

| Term | Layer | Mechanism | Accuracy |
|------|-------|-----------|----------|
| **Affinity** | L3/L4 | IP address | ~Approximate |
| **Persistence** | L7 | Cookies, headers | 100% accurate |

#### Implementation Methods

**1. Source IP Affinity:**
```python
class SourceIPAffinity:
    def __init__(self, servers):
        self.servers = servers
        self.mapping = {}  # IP -> server
    
    def get_server(self, client_ip):
        if client_ip not in self.mapping:
            # First request: use round-robin or other algorithm
            self.mapping[client_ip] = self.servers[
                hash(client_ip) % len(self.servers)
            ]
        return self.mapping[client_ip]
```

**Limitations:** NAT, mobile users changing IPs, proxy servers.

**2. Cookie-Based Persistence:**
```python
class CookiePersistence:
    def __init__(self, servers, cookie_name="SERVERID"):
        self.servers = servers
        self.cookie_name = cookie_name
        self.current = 0
    
    def get_server(self, request_cookies):
        # Check for existing session cookie
        server_id = request_cookies.get(self.cookie_name)
        if server_id and server_id in self.servers:
            return server_id, None  # No new cookie needed
        
        # New session: assign server and create cookie
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server, {self.cookie_name: server}
```

**3. Application Cookie Persistence:**
- Use existing session cookie (e.g., JSESSIONID)
- Load balancer learns server from Set-Cookie response
- Routes future requests with that cookie to same server

---

### Health Checking

Detect and remove unhealthy servers from the pool.

#### Active Health Checks

Load balancer proactively probes servers.

```python
import asyncio
import aiohttp

class ActiveHealthChecker:
    def __init__(self, servers, check_interval=10, 
                 healthy_threshold=2, unhealthy_threshold=3):
        self.servers = {server: {
            "healthy": True,
            "consecutive_successes": 0,
            "consecutive_failures": 0
        } for server in servers}
        self.check_interval = check_interval
        self.healthy_threshold = healthy_threshold
        self.unhealthy_threshold = unhealthy_threshold
    
    async def check_server(self, server):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{server}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except:
            return False
    
    async def run_health_checks(self):
        while True:
            for server in self.servers:
                is_healthy = await self.check_server(server)
                stats = self.servers[server]
                
                if is_healthy:
                    stats["consecutive_successes"] += 1
                    stats["consecutive_failures"] = 0
                    if stats["consecutive_successes"] >= self.healthy_threshold:
                        stats["healthy"] = True
                else:
                    stats["consecutive_failures"] += 1
                    stats["consecutive_successes"] = 0
                    if stats["consecutive_failures"] >= self.unhealthy_threshold:
                        stats["healthy"] = False
            
            await asyncio.sleep(self.check_interval)
    
    def get_healthy_servers(self):
        return [s for s, stats in self.servers.items() if stats["healthy"]]
```

**Health check types:**
- **TCP:** Connection succeeds
- **HTTP:** GET /health returns 2xx
- **gRPC:** Health checking protocol
- **Custom:** Application-specific logic

#### Passive Health Checks

Monitor actual traffic for failures.

```python
class PassiveHealthChecker:
    def __init__(self, servers, failure_threshold=5, 
                 recovery_time=30, window_size=60):
        self.servers = {server: {
            "healthy": True,
            "failures": [],  # timestamps of recent failures
            "ejected_until": 0
        } for server in servers}
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.window_size = window_size
    
    def record_failure(self, server):
        import time
        now = time.time()
        stats = self.servers[server]
        
        # Add failure, remove old ones outside window
        stats["failures"].append(now)
        stats["failures"] = [
            t for t in stats["failures"] 
            if now - t < self.window_size
        ]
        
        # Check if threshold exceeded
        if len(stats["failures"]) >= self.failure_threshold:
            stats["healthy"] = False
            stats["ejected_until"] = now + self.recovery_time
            stats["failures"] = []
    
    def record_success(self, server):
        # Optionally clear some failures on success
        pass
    
    def get_healthy_servers(self):
        import time
        now = time.time()
        healthy = []
        for server, stats in self.servers.items():
            # Check if recovery time passed
            if not stats["healthy"] and now > stats["ejected_until"]:
                stats["healthy"] = True
            if stats["healthy"]:
                healthy.append(server)
        return healthy
```

---

### Load Balancer High Availability

#### Active-Passive (Failover)

```
                    ┌─────────────────┐
                    │  Virtual IP     │
                    │  (Floating IP)  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
       ┌──────▼──────┐              ┌───────▼─────┐
       │   Active    │◄─Heartbeat──►│   Passive   │
       │     LB      │              │     LB      │
       └──────┬──────┘              └─────────────┘
              │                     (takes over on
              │                      Active failure)
    ┌─────────┼─────────┐
    │         │         │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Server1│ │Server2│ │Server3│
└───────┘ └───────┘ └───────┘
```

**Mechanism:** VRRP (Virtual Router Redundancy Protocol) or keepalived.

#### Active-Active

```
                    ┌─────────────────┐
                    │   DNS/Anycast   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
       ┌──────▼──────┐              ┌───────▼─────┐
       │     LB 1    │              │     LB 2    │
       │   (Active)  │              │   (Active)  │
       └──────┬──────┘              └──────┬──────┘
              │                            │
              └────────────┬───────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
          ┌───▼───┐    ┌───▼───┐    ┌───▼───┐
          │Server1│    │Server2│    │Server3│
          └───────┘    └───────┘    └───────┘
```

**Benefits:** Better resource utilization, higher throughput.
**Challenge:** Session synchronization between load balancers.

---

### Distributed Load Balancing (Service Mesh)

In service mesh architectures, each service has a sidecar proxy doing client-side load balancing.

```
┌─────────────────────┐     ┌─────────────────────┐
│     Service A       │     │     Service B       │
│  ┌───────────────┐  │     │  ┌───────────────┐  │
│  │   App Code    │  │     │  │   App Code    │  │
│  └───────┬───────┘  │     │  └───────────────┘  │
│          │          │     │          ▲          │
│  ┌───────▼───────┐  │     │  ┌───────┴───────┐  │
│  │ Sidecar Proxy │──┼─────┼──► Sidecar Proxy │  │
│  │  (Envoy/etc)  │  │     │  │  (Envoy/etc)  │  │
│  └───────────────┘  │     │  └───────────────┘  │
└─────────────────────┘     └─────────────────────┘
```

**Advantages:**
- No single point of failure
- Decentralized decision making
- Service-specific load balancing policies

**Challenges:**
- No global view of server load
- Power of Two Choices works well here
- Need service discovery (Consul, etcd, Kubernetes)

---

### Algorithm Comparison

| Algorithm | Adapts to Load | Server Awareness | Complexity | Best For |
|-----------|----------------|------------------|------------|----------|
| Round Robin | No | No | O(1) | Homogeneous servers |
| Weighted RR | No | Yes (static) | O(1) | Known capacity differences |
| IP Hash | No | No | O(1) | Session affinity (L4) |
| Least Connections | Yes | No | O(N) or O(log N)* | Varying request durations |
| Weighted LC | Yes | Yes | O(N) | Heterogeneous + varying load |
| Least Response Time | Yes | Yes | O(N) | Latency-sensitive apps |
| Power of Two | Yes | No | O(1) | Distributed/sidecar LBs |
| Resource-Based | Yes | Yes | O(N) | Complex workloads |

*O(log N) with binary tree implementation (HAProxy).

### Algorithm Selection Guide

```
                        Start
                          │
                          ▼
              ┌───────────────────────┐
              │ Servers homogeneous?  │
              └───────────┬───────────┘
                    │           │
                   Yes          No
                    │           │
                    ▼           ▼
            ┌───────────┐  ┌───────────────┐
            │Need sticky│  │Known capacity?│
            │ sessions? │  └───────┬───────┘
            └─────┬─────┘      │       │
              │       │       Yes      No
             Yes      No       │       │
              │       │        ▼       ▼
              ▼       ▼   Weighted  Least Conn
          IP Hash  Round    RR     or Weighted LC
                   Robin
                          
              ┌───────────────────────┐
              │ Distributed LBs?      │
              │ (sidecars/mesh)       │
              └───────────┬───────────┘
                    │           │
                   Yes          No
                    │           │
                    ▼           ▼
              Power of Two   Least Connections
              Random Choices  (centralized)
```

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

Anti-entropy protocols detect and repair data inconsistencies between replicas in distributed systems. The term originates from thermodynamics—entropy represents disorder (inconsistency), and anti-entropy mechanisms work to reduce it.

### Core Concepts

**Entropy Sources in Distributed Systems:**
- Network partitions causing divergent writes
- Node failures during replication
- Message loss or reordering
- Clock skew affecting version ordering
- Concurrent updates to the same key

**Anti-Entropy Approaches:**

| Approach | Trigger | Scope | Latency Impact |
|----------|---------|-------|----------------|
| Read Repair | On read | Single key | Foreground |
| Hinted Handoff | On write failure | Single key | Background |
| Active Anti-Entropy | Periodic | Full dataset | Background |
| Passive Anti-Entropy | On access | Accessed data | Foreground |

---

### Read Repair

Fix inconsistencies opportunistically during read operations.

```python
class ReadRepairCoordinator:
    def read_with_repair(self, key, replicas, quorum):
        responses = []
        for replica in replicas[:quorum]:
            responses.append(replica.read(key))
        
        # Find most recent version using version vector comparison
        latest = self.resolve_versions(responses)
        
        # Repair stale replicas asynchronously
        for i, response in enumerate(responses):
            if self.is_stale(response, latest):
                self.schedule_repair(replicas[i], key, latest)
        
        return latest.value
    
    def resolve_versions(self, responses):
        """Resolve using vector clocks or timestamps."""
        valid = [r for r in responses if r is not None]
        if not valid:
            return None
        
        # Compare version vectors
        latest = valid[0]
        for response in valid[1:]:
            cmp = self.compare_versions(response.version, latest.version)
            if cmp > 0:  # response is newer
                latest = response
            elif cmp == 0:  # concurrent - need conflict resolution
                latest = self.resolve_conflict(latest, response)
        
        return latest
    
    def schedule_repair(self, replica, key, latest):
        """Queue async repair to avoid blocking reads."""
        self.repair_queue.put((replica, key, latest.value, latest.version))
```

**Read Repair Variants:**
- **Blocking Read Repair:** Wait for repair before returning (stronger consistency)
- **Async Read Repair:** Return immediately, repair in background (lower latency)
- **Probabilistic Read Repair:** Repair only a percentage of reads (reduced overhead)

**Trade-offs:**
- Pros: No additional background processes, repairs hot data
- Cons: Increases read latency, cold data may remain inconsistent

---

### Hinted Handoff

Temporarily store writes for unavailable nodes, replay when they recover.

```python
class HintedHandoffCoordinator:
    def __init__(self, hint_window_hours=3):
        self.hint_window = hint_window_hours * 3600
        self.hints_store = HintsStore()
    
    def write(self, key, value, version, target_replicas):
        successful = []
        hints_created = []
        
        for replica in target_replicas:
            if replica.is_available():
                try:
                    replica.write(key, value, version)
                    successful.append(replica)
                except WriteFailure:
                    hint = self.create_hint(replica, key, value, version)
                    hints_created.append(hint)
            else:
                # Node unavailable - create hint
                hint = self.create_hint(replica, key, value, version)
                hints_created.append(hint)
        
        # Store hints for later delivery
        for hint in hints_created:
            self.hints_store.save(hint)
        
        return len(successful) >= self.write_quorum
    
    def create_hint(self, target_replica, key, value, version):
        return Hint(
            id=uuid4(),
            target_node_id=target_replica.id,
            key=key,
            value=value,
            version=version,
            timestamp=time.now(),
            ttl=self.hint_window
        )
    
    def replay_hints(self, recovered_node_id):
        """Called when gossip detects node recovery."""
        hints = self.hints_store.get_hints_for(recovered_node_id)
        
        for hint in hints:
            if hint.is_expired():
                self.hints_store.delete(hint)
                continue
            
            try:
                target = self.get_node(hint.target_node_id)
                target.write(hint.key, hint.value, hint.version)
                self.hints_store.delete(hint)
            except WriteFailure:
                # Node went down again, keep hint
                pass
```

**Hint Storage Schema:**
```sql
CREATE TABLE hints (
    hint_id         UUID PRIMARY KEY,
    target_node_id  UUID NOT NULL,
    key             BLOB NOT NULL,
    value           BLOB NOT NULL,
    version         BLOB NOT NULL,  -- Version vector or timestamp
    created_at      TIMESTAMP NOT NULL,
    ttl_seconds     INT NOT NULL,
    INDEX (target_node_id, created_at)
);
```

**Sloppy Quorum with Hinted Handoff:**

In strict quorum, writes fail if W nodes are unavailable. Sloppy quorum allows writes to any N healthy nodes in the hash ring:

```
Strict Quorum (N=3, W=2):
  Node A (down) ──X──► Write fails if only 1 of 3 available
  Node B ────────────► 
  Node C ────────────►

Sloppy Quorum (N=3, W=2):
  Node A (down) ──X──► Hint stored at Node D
  Node B ────────────► Write succeeds
  Node C ────────────► Write succeeds
  Node D (backup) ───► Stores hint for Node A
```

**Hinted Handoff Limitations:**
- Hints consume storage on backup nodes
- Long outages exhaust hint window
- Backup node failure loses hints
- Read consistency not guaranteed during hint window

---

### Active Anti-Entropy (Merkle Trees)

Periodically compare and synchronize entire datasets using hash trees.

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
         ├── Yes: Done (O(1) comparison)
         └── No: Compare subtrees recursively
                    │
                    ▼
              Find differing leaves (O(log N) hashes)
                    │
                    ▼
              Sync only differing keys
```

**Merkle Tree Anti-Entropy Implementation:**

```python
class MerkleAntiEntropy:
    def __init__(self, data_store, tree_depth=10):
        self.store = data_store
        self.depth = tree_depth
        self.tree = None
        self.rebuild_interval = 3600  # 1 hour
    
    def build_tree(self):
        """Build Merkle tree over key ranges."""
        leaves = []
        for key_range in self.partition_keyspace():
            range_hash = self.hash_key_range(key_range)
            leaves.append(range_hash)
        
        self.tree = self.build_from_leaves(leaves)
        return self.tree
    
    def hash_key_range(self, key_range):
        """Hash all key-value pairs in range."""
        h = hashlib.sha256()
        for key, value, version in self.store.scan(key_range):
            h.update(key.encode())
            h.update(value)
            h.update(str(version).encode())
        return h.digest()
    
    def compare_with_peer(self, peer):
        """Find differing key ranges with peer."""
        differences = []
        self._compare_subtree(peer, 0, 0, differences)
        return differences
    
    def _compare_subtree(self, peer, node_idx, depth, differences):
        my_hash = self.tree.get_hash(node_idx)
        peer_hash = peer.get_tree_hash(node_idx)
        
        if my_hash == peer_hash:
            return  # Subtree matches
        
        if depth == self.depth:
            # Leaf node - this key range differs
            key_range = self.tree.get_key_range(node_idx)
            differences.append(key_range)
            return
        
        # Recurse into children
        left_child = 2 * node_idx + 1
        right_child = 2 * node_idx + 2
        self._compare_subtree(peer, left_child, depth + 1, differences)
        self._compare_subtree(peer, right_child, depth + 1, differences)
    
    def sync_differences(self, peer, differences):
        """Synchronize differing key ranges."""
        for key_range in differences:
            my_data = list(self.store.scan(key_range))
            peer_data = peer.scan(key_range)
            
            # Merge using version vectors
            merged = self.merge_data(my_data, peer_data)
            
            # Apply merged data to both replicas
            for key, value, version in merged:
                self.store.write(key, value, version)
                peer.write(key, value, version)
```

**Merkle Tree Optimizations:**
- **Incremental Updates:** Update tree on writes instead of full rebuild
- **Bloom Filter Pre-check:** Skip comparison if Bloom filters match
- **Adaptive Depth:** Deeper trees for hot key ranges
- **Streaming Comparison:** Compare while building to reduce memory

**Cassandra Anti-Entropy (nodetool repair):**
```bash
# Full repair - compares all data
nodetool repair keyspace_name

# Incremental repair - only unrepaired data
nodetool repair -inc keyspace_name

# Parallel repair - multiple ranges simultaneously
nodetool repair -pr keyspace_name
```

---

### Gossip-Based Anti-Entropy

Epidemic protocols for probabilistic consistency in large clusters.

**Push, Pull, and Push-Pull Variants:**

```python
class GossipAntiEntropy:
    def __init__(self, node_id, peers, fanout=3):
        self.node_id = node_id
        self.peers = peers
        self.fanout = fanout  # Number of peers per round
        self.state = {}       # key -> (value, version_vector)
    
    def gossip_round(self):
        """Execute one gossip round."""
        targets = random.sample(self.peers, min(self.fanout, len(self.peers)))
        
        for peer in targets:
            # Push-Pull: exchange state bidirectionally
            self.push_pull(peer)
    
    def push_pull(self, peer):
        """Most efficient: exchange in both directions."""
        # Send digest of local state
        my_digest = self.create_digest()
        peer_digest = peer.exchange_digest(my_digest)
        
        # Determine what to send and request
        to_send = self.find_newer_local(peer_digest)
        to_request = self.find_newer_remote(peer_digest)
        
        # Exchange actual data
        peer_data = peer.exchange_data(to_send, to_request)
        self.merge_data(peer_data)
    
    def create_digest(self):
        """Create compact summary of local state."""
        return {key: version for key, (_, version) in self.state.items()}
    
    def find_newer_local(self, peer_digest):
        """Find keys where local version is newer."""
        newer = []
        for key, (value, version) in self.state.items():
            peer_version = peer_digest.get(key)
            if peer_version is None or version > peer_version:
                newer.append((key, value, version))
        return newer
```

**Convergence Analysis:**

For N nodes with push-pull gossip:
- Expected rounds to infect all nodes: O(log N)
- Message complexity per round: O(N × fanout)
- Total messages to convergence: O(N × fanout × log N)

```
Round 1: 1 node infected
Round 2: ~fanout nodes infected  
Round 3: ~fanout² nodes infected
...
Round log_fanout(N): All N nodes infected
```

**Scuttlebutt Protocol:**

Efficient gossip that only transmits deltas:

```python
class ScuttlebuttNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.state = {}  # key -> (value, version, source_node)
        self.version_vector = defaultdict(int)  # node_id -> max_version
    
    def update(self, key, value):
        """Local update."""
        self.version_vector[self.node_id] += 1
        version = self.version_vector[self.node_id]
        self.state[key] = (value, version, self.node_id)
    
    def exchange(self, peer):
        """Scuttlebutt exchange."""
        # Send our version vector
        peer_vv = peer.get_version_vector()
        
        # Find deltas: entries where we have newer versions
        deltas = []
        for key, (value, version, source) in self.state.items():
            if version > peer_vv.get(source, 0):
                deltas.append((key, value, version, source))
        
        # Send deltas, receive peer's deltas
        peer_deltas = peer.receive_deltas(deltas, self.version_vector)
        
        # Merge peer's deltas
        for key, value, version, source in peer_deltas:
            if version > self.version_vector.get(source, 0):
                self.state[key] = (value, version, source)
                self.version_vector[source] = max(
                    self.version_vector[source], version
                )
```

---

### Version Vectors and Conflict Detection

Track causality to detect concurrent updates.

```python
class VersionVector:
    def __init__(self, node_id):
        self.node_id = node_id
        self.vector = defaultdict(int)
    
    def increment(self):
        """Increment on local write."""
        self.vector[self.node_id] += 1
        return dict(self.vector)
    
    def merge(self, other_vector):
        """Merge on receiving update."""
        for node, version in other_vector.items():
            self.vector[node] = max(self.vector[node], version)
    
    @staticmethod
    def compare(vv1, vv2):
        """
        Compare two version vectors.
        Returns:
          'before' if vv1 happened before vv2
          'after' if vv1 happened after vv2
          'concurrent' if neither dominates
          'equal' if identical
        """
        all_keys = set(vv1.keys()) | set(vv2.keys())
        
        vv1_dominates = False
        vv2_dominates = False
        
        for key in all_keys:
            v1 = vv1.get(key, 0)
            v2 = vv2.get(key, 0)
            
            if v1 > v2:
                vv1_dominates = True
            elif v2 > v1:
                vv2_dominates = True
        
        if vv1_dominates and not vv2_dominates:
            return 'after'
        elif vv2_dominates and not vv1_dominates:
            return 'before'
        elif not vv1_dominates and not vv2_dominates:
            return 'equal'
        else:
            return 'concurrent'
```

**Dotted Version Vectors (DVV):**

Optimization for client-server systems to reduce vector size:

```python
class DottedVersionVector:
    """
    Compact version vector for systems where clients
    don't maintain persistent identity.
    
    Structure: (dot, version_vector)
    - dot: (node_id, counter) for the latest update
    - version_vector: causal context
    """
    def __init__(self):
        self.dot = None  # (node_id, counter)
        self.vector = {}  # node_id -> counter
    
    def update(self, node_id, counter):
        """Record new update."""
        self.dot = (node_id, counter)
    
    def sync(self):
        """Move dot into vector."""
        if self.dot:
            node_id, counter = self.dot
            self.vector[node_id] = max(self.vector.get(node_id, 0), counter)
            self.dot = None
    
    def descends(self, other):
        """Check if self causally descends from other."""
        # Check dot
        if other.dot:
            node_id, counter = other.dot
            if self.vector.get(node_id, 0) < counter:
                if self.dot != other.dot:
                    return False
        
        # Check vector
        for node_id, counter in other.vector.items():
            if self.vector.get(node_id, 0) < counter:
                return False
        
        return True
```

---

### Conflict-Free Replicated Data Types (CRDTs)

Data structures that automatically resolve conflicts through mathematical properties.

**CRDT Properties:**
- **Commutativity:** Order of operations doesn't matter
- **Associativity:** Grouping of operations doesn't matter
- **Idempotency:** Duplicate operations have no effect

**State-Based CRDTs (CvRDTs):**

```python
class GCounter:
    """Grow-only counter - state-based CRDT."""
    
    def __init__(self, node_id):
        self.node_id = node_id
        self.counts = defaultdict(int)
    
    def increment(self, amount=1):
        self.counts[self.node_id] += amount
    
    def value(self):
        return sum(self.counts.values())
    
    def merge(self, other):
        """Merge by taking max of each node's count."""
        for node_id, count in other.counts.items():
            self.counts[node_id] = max(self.counts[node_id], count)


class PNCounter:
    """Positive-Negative counter - supports decrement."""
    
    def __init__(self, node_id):
        self.p = GCounter(node_id)  # Increments
        self.n = GCounter(node_id)  # Decrements
    
    def increment(self, amount=1):
        self.p.increment(amount)
    
    def decrement(self, amount=1):
        self.n.increment(amount)
    
    def value(self):
        return self.p.value() - self.n.value()
    
    def merge(self, other):
        self.p.merge(other.p)
        self.n.merge(other.n)


class LWWRegister:
    """Last-Writer-Wins Register."""
    
    def __init__(self):
        self.value = None
        self.timestamp = 0
    
    def write(self, value, timestamp):
        if timestamp > self.timestamp:
            self.value = value
            self.timestamp = timestamp
    
    def merge(self, other):
        if other.timestamp > self.timestamp:
            self.value = other.value
            self.timestamp = other.timestamp


class ORSet:
    """Observed-Remove Set - add-wins semantics."""
    
    def __init__(self, node_id):
        self.node_id = node_id
        self.elements = {}  # element -> set of (node_id, counter) tags
        self.counter = 0
    
    def add(self, element):
        self.counter += 1
        tag = (self.node_id, self.counter)
        if element not in self.elements:
            self.elements[element] = set()
        self.elements[element].add(tag)
    
    def remove(self, element):
        """Remove all observed tags for element."""
        if element in self.elements:
            self.elements[element] = set()
    
    def contains(self, element):
        return element in self.elements and len(self.elements[element]) > 0
    
    def merge(self, other):
        """Union of elements, union of tags."""
        all_elements = set(self.elements.keys()) | set(other.elements.keys())
        for element in all_elements:
            my_tags = self.elements.get(element, set())
            other_tags = other.elements.get(element, set())
            self.elements[element] = my_tags | other_tags
```

**Operation-Based CRDTs (CmRDTs):**

```python
class OpBasedCounter:
    """Operation-based counter - requires reliable causal broadcast."""
    
    def __init__(self):
        self.value = 0
    
    def increment(self):
        # Generate operation
        op = ('increment', 1)
        self.apply(op)
        return op  # Broadcast to other replicas
    
    def apply(self, op):
        """Apply operation - must be commutative."""
        op_type, amount = op
        if op_type == 'increment':
            self.value += amount
        elif op_type == 'decrement':
            self.value -= amount
```

**CRDT Selection Guide:**

| Use Case | CRDT Type | Conflict Resolution |
|----------|-----------|---------------------|
| Counters | G-Counter, PN-Counter | Sum of node counts |
| Registers | LWW-Register, MV-Register | Timestamp or multi-value |
| Sets | G-Set, 2P-Set, OR-Set | Add-wins or remove-wins |
| Maps | OR-Map, LWW-Map | Per-key resolution |
| Sequences | RGA, WOOT, LSEQ | Position-based ordering |

---

### Anti-Entropy Selection Guide

| Scenario | Recommended Approach | Rationale |
|----------|---------------------|-----------|
| Hot data, read-heavy | Read Repair | Repairs most-accessed data |
| Temporary failures | Hinted Handoff | Fast recovery, bounded storage |
| Long-term consistency | Merkle Tree AAE | Complete, efficient comparison |
| Large clusters | Gossip-based | Scalable, probabilistic |
| Conflict-prone data | CRDTs | Automatic resolution |
| Mixed workload | Layered approach | Combine multiple techniques |

**Layered Anti-Entropy (Production Pattern):**

```
Layer 1: Read Repair (immediate, per-request)
    ↓
Layer 2: Hinted Handoff (short-term, per-failure)
    ↓
Layer 3: Active Anti-Entropy (periodic, full-scan)
    ↓
Layer 4: Manual Repair (on-demand, operator-triggered)
```

**Used by:**
- **Amazon DynamoDB:** Merkle trees + hinted handoff + read repair
- **Apache Cassandra:** All four layers
- **Riak:** Merkle trees + read repair + CRDTs
- **CockroachDB:** Raft-based replication + MVCC

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

### Consensus and Replication
- Lamport, L. (1998). "The Part-Time Parliament" (Paxos)
- Lamport, L. (2006). "Fast Paxos"
- Ongaro, D., Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm" (Raft)
- Kemme, B., Jimenez-Peris, R., Patino-Martinez, M. (2010). "Database Replication" (Synthesis Lectures)

### Partitioning and Hashing
- Karger, D. et al. (1997). "Consistent Hashing and Random Trees"
- Mirrokni, V. et al. (2018). "Consistent Hashing with Bounded Loads" (Google Research)

### Probabilistic Data Structures
- Bloom, B. (1970). "Space/Time Trade-offs in Hash Coding with Allowable Errors"
- Flajolet, P. et al. (2007). "HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm"
- Merkle, R. (1987). "A Digital Signature Based on a Conventional Encryption Function"

### Anti-Entropy and Epidemic Protocols
- Demers, A. et al. (1987). "Epidemic Algorithms for Replicated Database Maintenance"
- Özkasap, Ö. et al. (2010). "An analytical framework for self-organizing peer-to-peer anti-entropy algorithms" (Performance Evaluation)
- DeCandia, G. et al. (2007). "Dynamo: Amazon's Highly Available Key-value Store" (SOSP)
- Lakshman, A., Malik, P. (2010). "Cassandra: A Decentralized Structured Storage System" (LADIS)

### CRDTs and Eventual Consistency
- Shapiro, M. et al. (2011). "Conflict-free Replicated Data Types" (SSS)
- Shapiro, M. et al. (2011). "A comprehensive study of Convergent and Commutative Replicated Data Types" (INRIA)
- Karayel, E., Gonzàlez, E. (2022). "Strong eventual consistency of the collaborative editing framework WOOT" (Distributed Computing)
- Viotti, P., Vukolić, M. (2016). "Consistency in Non-Transactional Distributed Storage Systems" (ACM Computing Surveys)
- Almeida, P. et al. (2018). "Delta State Replicated Data Types" (Journal of Parallel and Distributed Computing)

### Clocks and Ordering
- Fidge, C. (1988). "Timestamps in Message-Passing Systems That Preserve the Partial Ordering"
- Lamport, L. (1978). "Time, Clocks, and the Ordering of Events in a Distributed System"
- Preguiça, N. et al. (2012). "Dotted Version Vectors: Logical Clocks for Optimistic Replication"

### Failure Detection and Leader Election
- Garcia-Molina, H. (1982). "Elections in a Distributed Computing System" (Bully Algorithm)
- Hayashibara, N. et al. (2004). "The φ Accrual Failure Detector"

### Quorums and Consistency
- Bailis, P. et al. (2012). "Probabilistically Bounded Staleness for Practical Partial Quorums"
- Jiménez-Peris, R. et al. (2003). "Are Quorums an Alternative for Data Replication?"

### Load Balancing
- Mitzenmacher, M. (1996). "The Power of Two Choices in Randomized Load Balancing" (PhD Thesis, Harvard)
- Mitzenmacher, M., Richa, A., Sitaraman, R. (2001). "The Power of Two Random Choices: A Survey of Techniques and Results"

### Video Lectures

- **Computer Science Center** - Distributed Databases (Vadim Tsesko) - 2PC, Raft, Paxos
- **Yandex for ML** - Bloom Filter and Count-Min Sketch (probabilistic data structures)
- **Computer Science Seminar** - Raft Algorithm (Maxim Babenko)
- **Lektorium** - Object Replication to Database Replication (Fernando Pedone)

**Version:** 1.7.0 | **Updated:** December 2024
