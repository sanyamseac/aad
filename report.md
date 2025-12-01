# Social Network and Friend Recommendation Analysis Report

**A Comprehensive Study of Graph Traversals, Centrality Measures, Community Detection, and Friend Recommendation Systems on Facebook Social Networks**

---

## Codebase
GitHub Repository: [github.com/sanyamseac/aad](https://github.com/sanyamseac/aad)

---

## Authors

Team aad.js with following members:
- [Sanyam Gandhi](sanyam.gandhi@research.iiit.ac.in)
- [Dhyey Thummar](dhyey.thummar@students.iiit.ac.in)
- [Jainam Modi](jainam.modi@students.iiit.ac.in)
- [Ayush Kanani](ayush.kanani@students.iiit.ac.in)
- [Akshat Sharma](akshat.sharma@research.iiit.ac.in)

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Algorithm Descriptions](#algorithm-descriptions)
   - 3.1 [Graph Traversal and Connectivity](#31-graph-traversal-and-connectivity)
   - 3.2 [Centrality Measures](#32-centrality-measures)
   - 3.3 [Community Detection](#33-community-detection)
   - 3.4 [Friend Recommendation Systems](#34-friend-recommendation-systems)
4. [Implementation Details](#implementation-details)
5. [Experimental Setup](#experimental-setup)
6. [Conclusion](#conclusion)
7. [Bonus Disclosure](#bonus-disclosure)
8. [References](#references)

---

## Abstract

This report presents a comprehensive analysis of fundamental graph algorithms applied to social network analysis using the Stanford Network Analysis Platform (SNAP) Facebook ego network dataset. We implement and evaluate twelve algorithms across four categories: graph traversal and connectivity (Breadth-First Search, Depth-First Search, Union-Find by Rank, Union-Find by Size), centrality measures (Degree, Closeness, Betweenness, Eigenvector), community detection (Girvan-Newman, Louvain, Leiden), and friend recommendation systems (Common Neighbors, Jaccard Coefficient, Adamic-Adar, Preferential Attachment, Resource Allocation, and Node2Vec). 

Our empirical analysis validates theoretical complexity bounds across all algorithms on real-world social network data containing 4,039 nodes and 88,234 edges. Key findings include: (1) BFS and DFS demonstrate linear O(V+E) scaling with minimal order sensitivity, (2) Union-Find achieves near-constant amortized operations with O(E·α(V)) complexity, (3) Resource Allocation outperforms classical heuristics for link prediction with F1=0.0425, and (4) Leiden algorithm provides optimal community detection with guaranteed connectivity while maintaining O(N log N) complexity. The study demonstrates the practical trade-offs between algorithmic sophistication, computational efficiency, and solution quality for real-world social network applications.

---

## Introduction

### Problem Definition

Social networks are complex systems where understanding connectivity patterns, identifying influential nodes, detecting communities, and predicting future connections are fundamental problems with significant real-world applications. This project addresses four interconnected problems:

1. **Connectivity Analysis**: Determining connected components and traversing social graphs efficiently
2. **Influence Identification**: Quantifying node importance through multiple centrality perspectives
3. **Community Structure**: Partitioning networks into meaningful clusters of densely connected users
4. **Link Prediction**: Recommending potential friendships based on network topology

### Real-World Relevance

These problems have direct applications across multiple domains:

**Social Media and Marketing**: Platforms like Facebook, LinkedIn, and Twitter utilize these algorithms for friend suggestions ("People You May Know"), influencer identification for targeted marketing campaigns, content feed optimization through community detection, and viral marketing strategy development through understanding network structure.

**Public Health and Epidemiology**: Network analysis enables super-spreader identification for disease control, contact tracing optimization during pandemics, vaccination strategy planning based on centrality measures, and epidemic modeling through connectivity analysis.

**Cybersecurity and Infrastructure**: Organizations employ these methods for vulnerability assessment in computer networks, identifying critical infrastructure nodes, detecting anomalous community patterns indicating coordinated attacks, and network resilience analysis.

**Knowledge Management**: Academic and corporate environments use graph algorithms for research collaboration prediction, citation network analysis, expert discovery through centrality measures, and knowledge cluster identification via community detection.

### Project Objectives

This project aims to:

1. **Implement** twelve fundamental graph algorithms from scratch with rigorous correctness guarantees
2. **Validate** theoretical time and space complexity through extensive empirical analysis
3. **Compare** algorithmic performance on real-world social network data across multiple metrics
4. **Analyze** network structural properties and their impact on algorithm behavior
5. **Provide** practical guidance for algorithm selection based on application requirements and constraints

---

## Algorithm Descriptions

### 3.1 Graph Traversal and Connectivity

#### 3.1.1 Overview

This section analyzes the runtime behaviour, invariance properties, and connectivity metrics for four fundamental graph algorithms applied to the SNAP-EGO Facebook social network datasets. These algorithms are foundational for network analysis, social graph processing, and systems problems including connectivity discovery, component labelling, shortest-path layering, and dynamic connectivity.

#### 3.1.2 Breadth-First Search (BFS)

**Algorithm Description**

**Background:** BFS is a fundamental graph traversal algorithm first formalized for computing shortest paths in unweighted graphs. It systematically explores vertices level-by-level from a source node using a FIFO queue.

**Correctness:** BFS guarantees that nodes are visited in non-decreasing order of distance from the source. The FIFO queue ensures the first time a node is reached represents the shortest path in an unweighted graph.

### Complexity Analysis

**Theoretical Complexity:**

**Time:** $O(V + E)$
- Each vertex is enqueued/dequeued exactly once: $O(V)$
- Each edge is examined at most twice (once per endpoint): $O(E)$
- Total: $O(V + E)$

**Space:** $O(V)$
- Queue storage: $O(V)$ in worst case (star graph)
- Visited set/array: $O(V)$
- Level tracking: $O(V)$

### Empirical Performance

#### Complexity Grid (Runtime vs Graph Size)
![BFS Complexity Grid](./3.1.1cca/results/plots/bfs/B_Complexity_Grid.png)

**Graph Description:** This plot shows BFS execution time versus the combined input size (V+E) across 10 dataset steps ranging from 348 nodes to 4,039 nodes.

**Interpretation:** The plot exhibits a clear linear trend confirming the theoretical $O(V+E)$ complexity. The growth rate is consistent across small (348 nodes) to medium-large (4,039 nodes) graphs. Minor deviations from perfect linearity are due to cache effects and system noise.

**Key Observations:**
- Linear scaling confirmed: doubling V+E approximately doubles runtime
- No superlinear growth detected
- Execution times range from 0.0008s (smallest) to 0.0199s (largest)

#### Start-Node Invariance
![BFS Start-Node Invariance](./3.1.1cca/results/plots/bfs/A_Invariance.png)

**Graph Description:** Statistical summary of BFS runtime variation when starting from different nodes in the same graph. Metrics include mean, standard deviation, and coefficient of variation.

**Interpretation:** The low coefficient of variation (18.90%) demonstrates that BFS performance is relatively independent of start node selection. This is expected because BFS explores the entire connected component regardless of entry point.

**Key Observations:**
- Mean runtime: 0.01843s
- Standard deviation: 0.00097s  
- Coefficient of variation: 5.28% (excellent consistency)
- Performance is highly consistent across different start nodes

#### Order Invariance
![BFS Order Invariance](./3.1.1cca/results/plots/bfs/C_Order_Invariance.png)

**Graph Description:** Comparison of BFS runtime on normal (original adjacency list order) vs shuffled (randomized adjacency list order) edge orderings.

**Interpretation:** The small difference (0.00403s) between normal and shuffled orderings indicates BFS is relatively insensitive to adjacency list order. However, the slight increase with shuffling suggests cache locality benefits from the original ordering.

**Key Observations:**
- Normal ordering: 0.01512s
- Shuffled ordering: 0.01568s
- Difference: 0.00056s (3.7% increase)
- Minimal order sensitivity; cache effects present but very small

### Measured Performance Data

*Raw performance data available in: `3.1.1cca/results/data/bfs_performance.csv`*

### Strengths, Weaknesses, and Use Cases

**Strengths:**
- Simple, intuitive implementation
- Guaranteed shortest paths in unweighted graphs
- Optimal $O(V+E)$ time complexity
- Low memory overhead for sparse graphs
- Predictable performance characteristics

**Weaknesses:**
- Memory-intensive for very dense graphs (large queue)
- Not suitable for weighted shortest paths
- Full-graph exploration may be wasteful if target is near source
- Queue overhead can impact cache performance

**Ideal Use Cases:**
- Connected component discovery
- Shortest path in unweighted graphs
- Level-order network analysis
- Social network distance computation
- Web crawling with bounded depth

#### 3.1.3 Depth-First Search (DFS)

**Algorithm Description**

**Background:** DFS is a fundamental traversal that explores as deeply as possible along each branch before backtracking. It's widely used for structural graph properties, ordering, and cycle detection. The algorithm uses either recursion or an explicit stack to track the exploration path.

**Correctness:** DFS visits every reachable vertex exactly once. The recursion/stack mechanism ensures complete exploration: when all neighbors are processed, control returns to the previous vertex, guaranteeing eventual completion.

### Complexity Analysis

**Theoretical Complexity:**

**Time:** $O(V + E)$
- Each vertex is visited once: $O(V)$
- Each edge is examined at most twice: $O(E)$
- Total: $O(V + E)$

**Space:** $O(V)$
- Recursion stack depth: $O(V)$ worst case (path graph)
- Visited set/array: $O(V)$
- For iterative implementation: explicit stack $O(V)$

### Empirical Performance

#### Complexity Grid (Runtime vs Graph Size)
![DFS Complexity Grid](./3.1.1cca/results/plots/dfs/B_Complexity_Grid.png)

**Graph Description:** This plot displays DFS execution time as a function of graph size (V+E) across the 10-step dataset progression.

**Interpretation:** DFS exhibits linear scaling nearly identical to BFS, confirming $O(V+E)$ complexity. The consistent linear trend across all graph sizes demonstrates efficient implementation with minimal overhead.

**Key Observations:**
- Linear relationship between runtime and V+E
- Performance comparable to BFS (sometimes slightly faster, sometimes slower)
- No evidence of quadratic or superlinear behavior
- Execution times: 0.0008s (smallest) to 0.0192s (largest)

#### Start-Node Invariance
![DFS Start-Node Invariance](./3.1.1cca/results/plots/dfs/A_Invariance.png)

**Graph Description:** Statistical analysis of DFS runtime variability across different starting nodes.

**Interpretation:** DFS shows higher variance (34.56%) than BFS (5.28%), indicating more variability in performance across different start nodes. This is due to the depth-first nature which can vary significantly depending on the structure explored first.

**Key Observations:**
- Mean runtime: 0.02235s
- Standard deviation: 0.00773s
- Coefficient of variation: 34.56% (higher variance than BFS)
- More variability than BFS across different start points

#### Order Invariance
![DFS Order Invariance](./3.1.1cca/results/plots/dfs/C_Order_Invariance.png)

**Graph Description:** Comparison of DFS performance between original and shuffled adjacency list orderings.

**Interpretation:** DFS shows excellent order invariance with only 0.00059s difference (3.9% change). This minimal sensitivity suggests DFS is highly robust to edge ordering, likely due to depth-first exploration reducing cache sensitivity.

**Key Observations:**
- Normal ordering: 0.01522s
- Shuffled ordering: 0.01581s
- Difference: 0.00059s (3.9% change)
- Essentially order-independent
- Similar order invariance to BFS

### Measured Performance Data

*Raw performance data available in: `3.1.1cca/results/data/dfs_performance.csv`*

### Strengths, Weaknesses, and Use Cases

**Strengths:**
- Minimal implementation complexity
- Natural recursive implementation
- Excellent for structural analysis (cycles, ordering)
- Better cache locality than BFS in some cases
- Superior order invariance
- Low constant factors

**Weaknesses:**
- Recursion depth limitations (can overflow stack on deep graphs)
- Does not find shortest paths
- May explore very deep before exploring nearby nodes
- Not suitable for shortest-path queries

**Ideal Use Cases:**
- Topological sorting
- Cycle detection
- Connected component labelling
- Maze solving
- Dependency resolution
- Tree/forest traversal

#### 3.1.4 Union-Find (Disjoint Set Union)

We analyze both Union by Rank and Union by Size together as they implement the same abstraction with minor heuristic differences.

**Algorithm Description**

**Background:** Union-Find (Disjoint Set Union, DSU) is a data structure for maintaining a partition of elements under union and find operations. With path compression and union heuristics, it achieves near-constant time per operation. This structure dates to early work on Kruskal's MST and dynamic connectivity algorithms.

**How it Works:**
- Each element has a parent pointer; roots represent set representatives
### Algorithm Description

**Background:** Union-Find (Disjoint Set Union, DSU) is a data structure for maintaining a partition of elements under union and find operations. With path compression and union heuristics (rank or size), it achieves near-constant time per operation. This structure dates to early work on Kruskal's MST and dynamic connectivity algorithms. Each element has a parent pointer with roots representing set representatives. The **find** operation follows parent pointers to the root while applying path compression to flatten the tree. **Union by Rank** attaches the tree with smaller rank under the tree with larger rank, while **Union by Size** attaches the tree with fewer nodes under the tree with more nodes.

**Correctness:** The structure maintains valid partitions with these invariants: every element's parent chain leads to a root, roots uniquely represent disjoint sets, union correctly merges two sets by linking roots, and path compression preserves set membership while improving performance.
- $\alpha(n)$ is the inverse Ackermann function
- For all practical values ($n \leq 10^{80}$): $\alpha(n) \leq 4$
- Effective complexity: $O(m)$ (nearly linear)
- For graph connectivity: $O(E \cdot \alpha(V))$ ≈ $O(E)$

**The Inverse Ackermann Function ($\alpha$):**
The inverse Ackermann function grows extraordinarily slowly:
- $\alpha(1) = 1$
- $\alpha(10) = 2$
- $\alpha(1000) = 3$
- $\alpha(10^{80}) = 4$ (exceeds atoms in universe)

This means Union-Find achieves virtually constant-time operations in practice.

**Space:** $O(V)$
- Parent array: $O(V)$
- Rank/Size array: $O(V)$

### Empirical Performance

#### Complexity Grid - Union by Rank
![UFA Rank Complexity Grid](./3.1.1cca/results/plots/ufa_rank/B_Complexity_Grid.png)

**Graph Description:** Runtime of Union-Find by Rank versus graph size (V+E).

**Interpretation:** UFA-Rank shows near-linear scaling with V+E, consistent with $O(E \cdot \alpha(V))$ theory. The slope is steeper than BFS/DFS due to higher constant factors from find/union operations, but asymptotic behavior is excellent.

**Key Observations:**
- Near-linear scaling confirmed
- ~3× slower than BFS/DFS in absolute time
- Higher constant factors visible
- Execution times: 0.0021s (smallest) to 0.0646s (largest)

#### Complexity Grid - Union by Size
![UFA Size Complexity Grid](./3.1.1cca/results/plots/ufa_size/B_Complexity_Grid.png)

**Graph Description:** Runtime of Union-Find by Size versus graph size (V+E).

**Interpretation:** UFA-Size exhibits nearly identical scaling to UFA-Rank, confirming that both heuristics achieve the same asymptotic complexity. Minor runtime differences are due to implementation details rather than algorithmic differences.

**Key Observations:**
- Virtually identical to UFA-Rank performance
- Near-linear scaling maintained
- Execution times: 0.0021s (smallest) to 0.0648s (largest)
- Comparable constant factors to UFA-Rank

#### Order Invariance - Union by Rank
![UFA Rank Order Invariance](./3.1.1cca/results/plots/ufa_rank/C_Order_Invariance.png)

**Graph Description:** Comparison of UFA-Rank runtime on normal vs shuffled edge orderings.

**Interpretation:** UFA-Rank shows significant sensitivity to edge ordering (44.3% increase with shuffling). This is because edge order affects the sequence of union operations, which influences tree structure even with path compression. The original ordering likely has beneficial locality patterns.

**Key Observations:**
- Normal ordering: 0.05170s
- Shuffled ordering: 0.05919s
- Difference: 0.00748s (14.5% increase)
- Moderate order sensitivity
- Edge order impacts tree structure and cache behavior

#### Order Invariance - Union by Size
![UFA Size Order Invariance](./3.1.1cca/results/plots/ufa_size/C_Order_Invariance.png)

**Graph Description:** Comparison of UFA-Size runtime on normal vs shuffled edge orderings.

**Interpretation:** UFA-Size shows similar order sensitivity (13.9% increase) to UFA-Rank. Both Union-Find variants show moderate sensitivity to edge ordering due to their incremental set-merging nature.

**Key Observations:**
- Normal ordering: 0.05156s
- Shuffled ordering: 0.05873s
- Difference: 0.00717s (13.9% increase)
- Similar to UFA-Rank order sensitivity
- Edge ordering matters for Union-Find performance

### Measured Performance Data

*Raw performance data available in: `3.1.1cca/results/data/ufa_rank_performance.csv` and `3.1.1cca/results/data/ufa_size_performance.csv`*

### Strengths, Weaknesses, and Use Cases

**Strengths:**
- Near-constant time operations in practice
- Ideal for dynamic connectivity queries
- Minimal memory footprint
- Simple implementation
- Optimal for offline component labelling
- Essential for Kruskal's MST algorithm

**Weaknesses:**
- Higher constant factors than traversal algorithms
- Sensitive to edge ordering
- Does not provide path information
- Cannot answer "find shortest path" queries
- No level/distance information
- Requires all edges upfront (offline algorithm)

**Ideal Use Cases:**
- Connected component labelling from edge lists
- Kruskal's Minimum Spanning Tree
- Dynamic connectivity in incremental graphs
- Detecting cycles in undirected graphs
- Image segmentation (pixel connectivity)
- Network reliability analysis

#### 3.1.5 Algorithm Comparison and Network Analysis

**Complete Performance Table**

| Files | V | E | BFS(s) | DFS(s) | UFA Rank(s) | UFA Size(s) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 348 | 2,866 | 0.0008 | 0.0008 | 0.0021 | 0.0021 |
| 2 | 1,390 | 30,654 | 0.0059 | 0.0060 | 0.0200 | 0.0202 |
| 3 | 2,167 | 45,414 | 0.0098 | 0.0092 | 0.0313 | 0.0306 |
| 4 | 2,916 | 76,190 | 0.0155 | 0.0156 | 0.0577 | 0.0562 |
| 5 | 3,463 | 81,550 | 0.0177 | 0.0179 | 0.0581 | 0.0575 |
| 6 | 3,670 | 84,875 | 0.0184 | 0.0180 | 0.0639 | 0.0593 |
| 7 | 3,730 | 85,080 | 0.0197 | 0.0179 | 0.0594 | 0.0596 |
| 8 | 3,833 | 86,243 | 0.0183 | 0.0190 | 0.0615 | 0.0597 |
| 9 | 4,003 | 88,069 | 0.0199 | 0.0185 | 0.0618 | 0.0602 |
| 10 | 4,039 | 88,234 | 0.0194 | 0.0192 | 0.0646 | 0.0648 |

*Raw data also available in: `3.1.1cca/results/data/complete_performance.csv`*

### Comparative Analysis

**Runtime Performance:**
- BFS and DFS demonstrate the fastest absolute runtimes
- BFS and DFS have nearly identical performance
- UFA variants are ~3× slower than traversal algorithms
- All algorithms scale linearly with V+E

**Order Sensitivity:**
- BFS: Minimal sensitivity (3.7% variation)
- DFS: Minimal sensitivity (3.9% variation)
- UFA-Rank: Moderate sensitivity (14.5% variation)
- UFA-Size: Moderate sensitivity (13.9% variation)

**Start-Node Variance:**
- BFS: 5.28% coefficient of variation (excellent consistency)
- DFS: 34.56% coefficient of variation (higher variability)
- BFS shows superior start-node consistency

### Visual Comparisons

#### Global Network Metrics

![Nodes](./3.1.1cca/results/plots/Metric_nodes.png)
![Edges](./3.1.1cca/results/plots/Metric_edges.png)
![Components](./3.1.1cca/results/plots/Metric_num_components.png)
![GC Coverage](./3.1.1cca/results/plots/Metric_gc_coverage.png)

#### Algorithm Complexity Grids
![BFS Complexity](./3.1.1cca/results/plots/bfs/B_Complexity_Grid.png)
![DFS Complexity](./3.1.1cca/results/plots/dfs/B_Complexity_Grid.png)
![UFA Rank Complexity](./3.1.1cca/results/plots/ufa_rank/B_Complexity_Grid.png)
![UFA Size Complexity](./3.1.1cca/results/plots/ufa_size/B_Complexity_Grid.png)

**Network Metrics and Their Implications**

The following metrics characterize the structural properties of the analyzed networks and their impact on algorithm performance.

**1. Nodes (Graph Size)**
![Nodes Metric](./3.1.1cca/results/plots/Metric_nodes.png)

**Definition:** Total unique users/vertices in the network. Formally: $V = |\text{vertices}|$

**Observed Trend:** Grows from 348 to 4,039 nodes across 10 steps as ego-networks are merged.

**Interpretation:** This metric directly measures network scale. Linear growth indicates steady accumulation of ego-networks.

**Algorithmic Impact:**
- Directly affects traversal time (BFS/DFS visit every node)
- Determines memory requirements for visited sets
- Union-Find parent array scales with V
- Mid-sized graphs (hundreds to thousands) favor linear algorithms

**Practical Consequence:** These graphs are small enough for in-memory processing but large enough to benefit from algorithmic optimization. Scale-aware batching can improve throughput when processing multiple ego-networks.

**2. Edges (Connectivity)**
![Edges Metric](./3.1.1cca/results/plots/Metric_edges.png)

**Definition:** Total friendship connections in the network. Formally: $E = |\text{edges}|$

**Observed Trend:** Grows from 2,866 to 88,234 edges, increasing faster than nodes.

**Interpretation:** Superlinear edge growth (relative to nodes) indicates increasing connectivity as networks merge. The E/V ratio grows from ~8 to ~22.

**Algorithmic Impact:**
- Dominates runtime for all algorithms (all are $O(V+E)$ or $O(E \cdot \alpha(V))$)
- Each edge requires processing in traversals
- Each edge triggers union/find operations in UFA
- Edge-heavy graphs favor algorithms with low per-edge overhead

**Practical Consequence:** Edge processing costs dominate CPU time. Filtering, pre-aggregation, or sampling edges can significantly reduce computation. For analytics requiring full edge exploration, choose algorithms with minimal per-edge overhead (BFS/DFS over UFA for traversal tasks).

**3. Density**
![Density Metric](./3.1.1cca/results/plots/Metric_density.png)

**Definition:** Ratio of existing edges to maximum possible edges. For undirected graphs: $D = \frac{2E}{V(V-1)}$

**Observed Trend:** Decreases from 0.047 to 0.011 as network grows.is characteristic of real-world social networks where individuals maintain bounded friend counts (Dunbar's number) even as the network grows.

**Algorithmic Impact:**
- Lower density reduces memory for adjacency matrix representation
- Sparse graphs favor adjacency list representations
- BFS queue sizes and DFS stack depths remain manageable
- Algorithms with $O(V+E)$ complexity benefit from sparse graphs (E ≈ V rather than E ≈ V²)

**Practical Consequence:** Sparsity enables efficient storage and processing. Avoid algorithms requiring dense matrices ($O(V^2)$ space). Use sparse representations (adjacency lists) and algorithms that scale with actual edges rather than possible edges.

**4. Average Degree**
![Average Degree Metric](./3.1.1cca/results/plots/Metric_avg_degree.png)

**Definition:** Average number of connections per user. Calculated as: $\bar{k} = \frac{2E}{V}$

**Observed Trend:** Ranges from 16.47 to 52.26, stabilizing around 43-47 in later steps.40-50) shows these are well-connected communities. The relative stability suggests bounded individual connectivity despite network growth.

**Algorithmic Impact:**
- Determines BFS queue growth rate (higher degree = larger frontier)
- Affects DFS stack depth and exploration pattern
- High degree increases neighbor iteration overhead
- Cache performance degrades with higher branching factors

**Practical Consequence:** High average degree can cause memory pressure in BFS (large queues) and increase cache misses. Consider neighbor sampling for expensive analytics on high-degree nodes. Hub nodes (very high degree) may require special handling.

**5. Clustering Coefficient**
![Clustering Metric](./3.1.1cca/results/plots/Metric_clustering.png)

**Definition:** Probability that two neighbors of a node are also neighbors (triangle density). Computed as: $C = \frac{3 \times \text{number of triangles}}{\text{number of connected triples}}$

For individual nodes, local clustering is: $C_i = \frac{2|\{e_{jk}: v_j, v_k \in N(v_i), e_{jk} \in E\}|}{k_i(k_i-1)}$

**Observed Trend:** Consistently high (0.574 to 0.606), stable across all steps.icating strong triadic closure (friends of friends are friends). This confirms these are real social communities, not random graphs (which would have C ≈ 0).

**Algorithmic Impact:**
- Creates redundant path exploration in traversals
- BFS/DFS may revisit neighborhoods through triangles
- Doesn't change asymptotic complexity but increases constant factors
- Union-Find unaffected (only cares about connectivity, not local structure)

**Practical Consequence:** Strong local community structure makes triangle-based heuristics (common neighbors, Jaccard coefficient) highly effective for link prediction and friend recommendation. Clustering also enables community detection algorithms to find coherent groups.

**6. Number of Components**
![Components Metric](./3.1.1cca/results/plots/Metric_num_components.png)

**Definition:** Count of disjoint maximal connected subgraphs (separate connected components).

**Observed Trend:** Typically 1, briefly jumps to 2 at step 7, returns to 1.

**Interpretation:** The network is almost always fully connected. The temporary jump to 2 components at step 7 indicates a new ego-network was added without immediate bridges to the main component; subsequent steps reconnected it.

**Algorithmic Impact:**
- Single component means traversals reach all nodes from any start
- No benefit from per-component parallelization
- Union-Find processes entire network as one structure
- Multi-component graphs would enable parallel per-component processing

**Practical Consequence:** Optimize for single large-component performance rather than multi-component scenarios. Single-traversal analyses will cover nearly the entire network. For applications requiring isolation detection, note that fragmentation is rare in these datasets.

**7. Giant Component Coverage**
![GC Coverage Metric](./3.1.1cca/results/plots/Metric_gc_coverage.png)

**Definition:** Percentage of nodes in the largest connected component. Calculated as: $\text{GC Coverage} = \frac{|GC|}{V} \times 100\%$

**Observed Trend:** 98.39% to 100%, typically 100%. virtually all nodes. The brief dip to 98.39% at step 7 corresponds to the temporary second component. Near-perfect coverage confirms network cohesion.

**Algorithmic Impact:**
- Single-source traversals reach nearly all nodes
- No isolated subgraphs require separate handling
- Path queries almost always have solutions
- Network is conducive to global diffusion processes

**Practical Consequence:** Information, influence, or epidemics can reach essentially everyone from any starting point. Global network properties (diameter, average path length) are meaningful since they cover nearly all nodes.

**8. Giant Component Size (Nodes)**
![GC Size Nodes](./3.1.1cca/results/plots/Metric_gc_size_nodes.png)

**Definition:** Number of nodes in the largest connected component. Formally: $|GC| = \max_i |C_i|$ where $C_i$ are components.

**Observed Trend:** Grows from 348 to 4,039, mirroring total node count (due to 100% coverage).
- Traversal workload determined by GC size
- Algorithms must handle full-network-scale operations
- Memory requirements scale with GC size

**Practical Consequence:** No optimization available from ignoring small disconnected components. All algorithms must scale to full network size.

**9. Diameter**
![Diameter Metric](./3.1.1cca/results/plots/Metric_diameter.png)

**Definition:** Longest shortest path between any two nodes in the giant component. Defined as: $\text{diam}(G) = \max_{u,v \in V} d(u,v)$ where $d(u,v)$ is the shortest path length.

**Observed Trend:** Ranges from 2 to 8, growing slowly with network size.
- BFS reaches maximum depth quickly (8 levels vs 4,000 nodes)
- Limited BFS queue depth despite large networks
- DFS recursion depth bounded by diameter in well-connected graphs
- Path-based algorithms benefit from short distances

**Practical Consequence:** Everyone is at most 8 steps away from everyone else. This makes the network excellent for rapid information diffusion, viral marketing, and influence propagation. Short paths enable efficient routing and recommendation.

**10. Average Path Length**
![Average Path Metric](./3.1.1cca/results/plots/Metric_avg_path.png)

**Definition:** Mean shortest-path distance between all pairs of nodes in the giant component. Computed as: $\bar{d} = \frac{1}{n(n-1)} \sum_{u \neq v} d(u,v)$ where $n = |GC|$

**Observed Trend:** Ranges from 1.95 to 3.72 hops, growing slowly.
- Local neighborhood analysis (1-2 hops) captures significant network context

**Practical Consequence:** Features based on 1-2 hop neighborhoods capture most network structure for machine learning and recommendation tasks. Influence or recommendations propagate quickly across the network. Short paths enable efficient friend-of-friend suggestions and collaborative filtering.

**Integrated Analysis: Network Structure and Algorithm Performance**

*Small-World Property*

The SNAP-EGO Facebook networks exhibit the classic **small-world** characteristics:
- High clustering coefficient (0.57-0.61): Strong local communities
- Short average path length (1.95-3.72): Global reachability
- Low diameter (2-8): Rapid information diffusion

**Implications:**
- BFS reaches most nodes within 2-4 levels despite thousands of nodes
- Traversal algorithms achieve near-complete exploration efficiently
- Local neighborhood features are highly predictive for link prediction
- Network is robust to node removal (multiple short paths exist)

*Scale and Sparsity*

The graphs grow from 348 to 4,039 nodes and 2,866 to 88,234 edges:
- Density decreases (0.047 → 0.011): Network becomes sparser
- Average degree stabilizes (43-47): Bounded individual connectivity
- E ≈ 22V in final step: Linear edge-node relationship

**Implications:**
- All tested algorithms maintain $O(V+E)$ or near-linear scaling
- Sparse representation (adjacency lists) is optimal
- Memory footprint remains manageable for all algorithms
- Linear algorithms (BFS/DFS) are ideal for these scales

*Component Structure*

The network maintains a single giant component covering 98-100% of nodes:
- Typically 1 component (fully connected)
- Brief fragmentation at step 7 (2 components) quickly resolves
- High GC coverage ensures global reachability

**Implications:**
- No benefit from per-component parallelization
- Single-source traversals reach entire network
- Union-Find efficiently labels one large component
- Path queries almost always have solutions

*Algorithm Selection Guide*

**Choose BFS when:**
- Shortest paths needed
- Level-based analysis required
- Guaranteed optimal paths desired
- Order robustness important

**Choose DFS when:**
- Structural properties needed (cycles, ordering)
- Memory constraints exist (better cache locality)
- Order invariance critical (0.6% variation)
- Recursive implementation preferred

**Choose Union-Find when:**
- Only connectivity queries needed (no paths)
- Processing edge lists offline
- Building Kruskal's MST
- Dynamic connectivity required
- Minimal memory footprint needed

**Performance Summary:**
- BFS/DFS: Fastest absolute runtime (0.002-0.05s range)
- UFA: 2-3× slower but still near-linear
- DFS: Best order invariance (0.6%)
- BFS: Best for distance queries
- All: Scale efficiently to 4k nodes, 88k edges

---

## Detailed Network Connectivity Data

*Comprehensive connectivity metrics for each dataset step are available in: `3.1.1cca/results/data/detailed_connectivity.csv`*

The data shows how network structure evolves as ego-networks are progressively merged.

**Key Observations:**
- Network grows 11.6× in nodes and 30.8× in edges
- Density decreases 4.4× indicating increasing sparsity
- Average degree stabilizes around 43-47 after initial growth
- Clustering remains remarkably stable (0.57-0.61)
- Connectivity remains robust (1 component, 100% GC coverage typical)
- Diameter grows slowly (2→8) confirming small-world property
- Average path length grows gradually (1.95→3.74 hops)




### 3.2 Centrality Measures

#### 3.2.1 Overview

This section implements and analyzes four fundamental centrality measures on Facebook ego networks: Degree Centrality (DC), Closeness Centrality (CC), Betweenness Centrality (BC), and Eigenvector Centrality (EC). These measures quantify node importance in social networks using different perspectives to identify influential nodes, information brokers, and network hubs.

#### 3.2.2 Degree Centrality (DC)

**Description**

Introduced by Freeman (1978). Measures importance based on number of direct connections. Simplest and most intuitive centrality metric.

### Algorithm

**Formula**: `DC(v) = deg(v) / (n - 1)`

**Steps**:
1. Count neighbors for each node
2. Normalize by maximum possible degree (n-1)

### Proof of Correctness

- Adjacency list stores all neighbors → correct degree count
- Normalization by (n-1) ensures range [0,1]
- Maximum degree (n-1) yields DC=1, minimum (0) yields DC=0

### Complexity

**Theoretical**:
- Time: O(V + E) - single pass through all edges
- Space: O(V) - stores centrality scores

**Empirical**: See combined complexity analysis graph in Comparative Analysis section.

Linear growth confirms O(V+E) complexity. Algorithm scales efficiently even for large graphs.

### Results & Insights

**Top 5 Nodes**:
```
Node 107: DC = 0.2588
Node 1684: DC = 0.1961
Node 1912: DC = 0.1870
Node 3437: DC = 0.1355
Node 0: DC = 0.0859
```

**Insights**: Ego nodes dominate due to direct connections with their entire ego network.

### Strengths & Weaknesses

**Strengths**: Fastest computation, intuitive, works on any graph
**Weaknesses**: Only local view, ignores connection quality
**Use Cases**: Quick hub identification, real-time analysis, initial filtering

#### 3.2.3 Closeness Centrality (CC)

**Description**

Introduced by Bavelas (1950), formalized by Freeman (1978). Measures average distance to all other nodes. High closeness = efficient information broadcaster.

### Algorithm

**Formula**: `CC(v) = [(n-1) / (N-1)] × [(n-1) / Σd(v,u)]`

**Steps**:
1. BFS from each node to find shortest paths
2. Sum distances to all reachable nodes
3. Compute reciprocal of average distance
4. Apply Wasserman-Faust normalization for disconnected graphs

### Proof of Correctness

- BFS guarantees shortest paths in unweighted graphs
- Reciprocal gives higher scores for shorter distances
- Normalization handles disconnected components correctly
- Isolated nodes receive CC=0

### Complexity

**Theoretical**:
- Time: O(V × (V + E)) = O(V²) for sparse, O(V³) for dense
- Space: O(V) - BFS queue and distance arrays

**Empirical**: See combined complexity analysis graph in Comparative Analysis section.

Quadratic growth for sparse social networks confirms theoretical bounds.

### Results & Insights

**Top 5 Nodes**:
```
Node 107: CC = 0.4597
Node 58: CC = 0.3974
Node 428: CC = 0.3948
Node 563: CC = 0.3939
Node 1684: CC = 0.3936
```

**Insights**: Nodes with high closeness can efficiently reach entire network, ideal for broadcasting.

### Strengths & Weaknesses

**Strengths**: Global perspective, meaningful interpretation
**Weaknesses**: Expensive for large networks, sensitive to disconnection
**Use Cases**: Communication networks, epidemic modeling, supply chains (<10K nodes)

#### 3.2.4 Betweenness Centrality (BC)

**Description**

Freeman (1977), optimized by Brandes (2001). Measures how often a node lies on shortest paths between others. Identifies information brokers and bridges.

### Algorithm

**Formula**: `BC(v) = Σ(σ(s,t|v) / σ(s,t))`

**Steps** (Brandes' Algorithm):
1. For each source node: Run BFS to find shortest paths
2. Track predecessors and path counts
3. Backtrack from furthest nodes, accumulating dependency
4. Normalize by (n-1)(n-2)/2

### Proof of Correctness

- BFS finds all shortest paths
- Path counting: σ[w] = Σσ[v] for predecessors v
- Pair Dependency Theorem (Brandes 2001): BC(v) = Σδₛ(v)
- Algorithm correctly computes and accumulates dependencies

### Complexity

**Theoretical**:
- Time: O(V × E) for sparse, O(V³) for dense
- Space: O(V + E) - predecessor lists and arrays

**Empirical**: See combined complexity analysis graph in Comparative Analysis section.

Growth between O(V²) and O(V³), closer to O(V²) for sparse networks. Brandes' optimization provides 10-100× speedup over naive O(V³).

### Results & Insights

**Top 5 Nodes**:
```
Node 107: BC = 0.4805
Node 1684: BC = 0.3378
Node 3437: BC = 0.2361
Node 1912: BC = 0.2293
Node 1085: BC = 0.1490
```

**Insights**: Identifies bridge nodes connecting communities. Some moderate-degree nodes have high BC due to structural position.

### Strengths & Weaknesses

**Strengths**: Reveals bottlenecks, good for community detection
**Weaknesses**: Computationally expensive, assumes equal path likelihood
**Use Cases**: Bridge identification, vulnerability analysis, community detection (<50K nodes)

#### 3.2.5 Eigenvector Centrality (EC)

**Description**

Bonacich (1972). Recursive measure where importance depends on neighbors' importance. Foundation for Google's PageRank.

### Algorithm

**Formula**: Eigenvector **x** satisfies **Ax = λx**

**Steps** (Power Iteration):
1. Initialize all nodes with score 1/√n
2. Iterate until convergence:
   - New score = Σ(neighbor scores)
   - Normalize to unit length (L2 norm)
   - Check convergence (|x^(k+1) - x^k| < ε)
3. Return converged scores

### Proof of Correctness

- Perron-Frobenius Theorem: Connected graph → unique dominant eigenvector
- Power iteration: x^k = A^k x^0 / ||A^k x^0|| → v₁ (dominant eigenvector)
- Convergence rate: O(|λ₂/λ₁|^k)
- L2 normalization ensures numerical stability

### Complexity

**Theoretical**:
- Time: O(k × E), where k = iterations (typically 20-100)
- Space: O(V) - current and previous vectors

**Empirical**: See combined complexity analysis graph in Comparative Analysis section.

Linear growth with edges, converges in 25-40 iterations typically.

### Results & Insights

**Top 5 Nodes**:
```
Node 1912: EC = 0.0954
Node 2266: EC = 0.0870
Node 2206: EC = 0.0861
Node 2233: EC = 0.0852
Node 2464: EC = 0.0843
```

**Insights**: High EC nodes are connected to other important nodes, indicating network core membership.

### Strengths & Weaknesses

**Strengths**: Quality over quantity, fast via power iteration, strong theory
**Weaknesses**: Requires connectivity, sensitive to dense clusters
**Use Cases**: Status hierarchies, citation networks, web ranking, influence propagation

#### 3.2.6 Comparative Analysis

**Centrality Score Distributions**

![Score Distributions for All Centralities](<./3.1.2ca/results/centrality_distributions.png>)

**Graph Description**: Four histograms (2×2 grid) showing the frequency distribution of centrality scores on a logarithmic scale.

**Interpretation**:
- **Power-law distribution**: All measures show right-skewed distributions where most nodes have low scores and few nodes have very high scores
- **Degree Centrality**: Smoothest distribution, indicating varied connectivity levels across the network
- **Betweenness Centrality**: Highly concentrated near zero, showing most nodes are not bridges; few nodes control information flow
- **Closeness Centrality**: More uniform distribution suggests many nodes have similar average distances to others
- **Eigenvector Centrality**: Sharp peak at low values indicates most nodes are not connected to influential neighbors
- **Implication**: Network exhibits hierarchical structure with clear distinction between ordinary nodes and influential hubs

**Correlation Matrix**

![Correlation Heatmap](<./3.1.2ca/results/correlation_heatmap.png>)

**Graph Description**: Symmetric 4×4 heatmap with color-coded Pearson correlation coefficients between all centrality measure pairs. Colors range from blue (low correlation) to red (high correlation).

**Interpretation of Correlation Coefficients**:
- **Strong correlation (≥0.7)**: Measures capture very similar aspects of node importance
- **Moderate correlation (0.4-0.7)**: Measures share some common patterns but also capture unique aspects
- **Weak correlation (<0.4)**: Measures capture largely independent aspects of node importance

**Key Insights**:
- **DC vs CC**: Well-connected nodes tend to be closer to all others (more connections = shorter average distances)
- **DC vs BC**: High-degree nodes often lie on shortest paths (hubs naturally become bridges)
- **DC vs EC**: Popular nodes tend to connect to other popular nodes (homophily in social networks)
- **CC vs BC**: Nodes close to others often bridge communities (central position enables bridging)
- **CC vs EC**: Centrally located nodes connect to important neighbors (proximity to influential nodes)
- **BC vs EC**: Bridge nodes may or may not connect to influential nodes (structural vs. recursive importance)
- **All positive correlations**: Different centrality aspects tend to reinforce each other in this network
- **No perfect correlation**: Each measure captures unique aspects of node importance

**Top Nodes Across Measures**

![Top 20 Comparison](<./3.1.2ca/results/top_nodes_comparison.png>)

**Graph Description**: Four horizontal bar charts (2×2 grid) showing the top 20 nodes for each centrality measure with their scores.

**Interpretation**:
- **Consistent leaders**: Ego nodes (107, 1684, 1912, 0, 3437) dominate top positions across almost all measures
- **Node 107**: Highest scorer in almost all measures, indicating this node is the most influential across all dimensions
- **Score variance**: Degree and Eigenvector show more gradual decline; Betweenness and Closeness show steeper drop-offs
- **Rank stability**: Top nodes maintain similar relative positions across measures, confirming robust importance
- **Mid-tier differences**: Ranks 10-20 vary significantly between measures, revealing nodes with specialized roles
- **Implication**: Network has a stable core of super-influential nodes (ego nodes) with clear hierarchical structure

**Complexity Comparison**

![Time Complexity for All Centralities](<./3.1.2ca/results/scalability_vs_complexity.png>)

**Graph Description**: Four scatter plots (2×2 grid) showing runtime vs. theoretical complexity for each algorithm as graph size increases.

**Interpretation**:
- **Degree Centrality (top-left)**: Linear relationship with O(V+E) shows runtime grows proportionally with graph size
- **Closeness Centrality (top-right)**: Quadratic curve with O(V×(V+E)) confirms O(V²) for sparse networks; O(V³) for dense
- **Betweenness Centrality (bottom-left)**: Near-quadratic growth with O(V×E) validates theoretical complexity for sparse graphs
- **Eigenvector Centrality (bottom-right)**: Linear growth with O(k×E) confirms constant iterations (k≈20-100) dominate runtime
- **Scalability verdict**: DC and EC scale well (linear); CC and BC struggle with large networks (quadratic+)
- **Practical implications**: For graphs with >10K nodes, prefer DC or EC; avoid CC and BC unless absolutely necessary

| Algorithm | Time | Space | Speed (1000 nodes) |
|-----------|------|-------|-------------------|
| DC | O(V+E) | O(V) | 0.01s (baseline) |
| EC | O(k×E) | O(V) | 0.05s (5×) |
| CC | O(V²-V³) | O(V) | 5s (500×) |
| BC | O(V×E) | O(V+E) | 50s (5000×) |

**Decision Matrix**

| Need | Algorithm | Reason |
|------|-----------|--------|
| Speed | DC | Fastest |
| Influence | EC | Connection quality |
| Bridges | BC | Path control |
| Broadcasting | CC | Average distance |
| Large networks | DC/EC | Scalable |








### 3.3 Community Detection

#### 3.3.1 Overview

This section implements community detection algorithms that partition the graph into groups (communities) where nodes are densely connected internally but sparsely connected to nodes in other groups. This analysis is crucial for understanding the latent social circles within the Facebook dataset.

**Objectives**

We are implementing and strictly comparing three distinct algorithmic approaches to partition the Facebook social graph into meaningful clusters .

* **Algorithmic Implementation:** We are deploying three specific algorithms that represent different paradigms of graph clustering:

    * **Girvan-Newman Algorithm:** A divisive, hierarchical method that focuses on identifying "bridges" between communities . We iteratively calculate edge betweenness and remove the edges with the highest values to slowly disintegrate the graph into isolated communities .
    
    * **Louvain Modularity Algorithm:** A heuristic-based, greedy optimization method . We aggregate nodes to maximize the "Modularity" score, which measures the density of links inside communities compared to links between communities .
    
    * **Leiden Algorithm:** An improved version of the Louvain method . We are implementing this to address specific connectivity defects in Louvain (such as disconnected communities) to guarantee well-connected clusters .

* **Performance Benchmarking:** We are not just finding communities; we are analyzing the *cost* of finding them. We record the runtime performance to evaluate scalability, specifically looking at how the slow, precise nature of Girvan-Newman compares to the fast, approximate nature of Louvain and Leiden .

* **Structural Evaluation:** We calculate Modularity Scores and analyze community size distributions to quantitatively determine which method produced the highest quality partition .

### 2. How's it related to the practical world?
Community detection is the computational equivalent of discovering "social circles" in real life.

* **Targeted Marketing & Influence:** By identifying dense communities, systems can identify local influencers . If a user in a specific cluster 
adopts a product or idea, it is highly probable that others in the same dense cluster will share that interest.

* **Recommendation Systems:** This directly feeds into the Friend Recommendation objective . Recommendations are often most successful when suggested within the same community. If User A and User B are in the same detected community but aren't friends yet, the algorithm suggests a high 
probability of a missing link.

* **Content Feed Optimization:** Social platforms use these algorithms to determine what content to show. Content viral in one community might not be relevant to a distant community, allowing for personalized feed curation.

### 3. How are we getting our results?
We are integrating the raw SNAP dataset with Python-based network analysis libraries to generate both visual and metric-based results .

* **Data Pipeline:**
    1.  **Input:** We load the undirected social graph from the SNAP Facebook dataset, where nodes are users and edges are friendships .

    2.  **Preprocessing:** We ensure the graph is suitable for traversal (checking for connected components) .

* **The Integration Logic:**

    * **Step A (Divisive Processing):** For **Girvan-Newman**, we calculate the shortest paths between all pairs. Edges with high "traffic" (betweenness) are identified as bottlenecks connecting different groups. We cut these edges until the modularity score peaks.

    * **Step B (Agglomerative/Optimization Processing):** For **Louvain and Leiden**, we start with every node in its own community. We then merge nodes into their neighbor's community *if and only if* it increases the total modularity of the network. Leiden adds a refinement phase to split sub-optimally merged communities .

    * **Step C (Validation):** We visualize the resulting communities to inspect if the algorithms successfully clustered dense regions .

### 3.3 Generated Outputs & Comparative Analysis

We integrate the execution of our algorithms with real-time plotting to generate four key analytical deliverables. These outputs allow us to validate the theoretical performance against our actual dataset.

#### 1. Community Structure Visualization

**Goal:** To visually inspect the quality of the graph partitioning .

**Analysis of Results:**

* **Girvan-Newman:** As seen in the visualization, this algorithm struggled to break the giant component, resulting in one massive cluster (blue nodes) with very few peripheral detachments. This visually confirms its failure to effectively partition this specific dense network.

* **Louvain:** Produced the most visually distinct and balanced "social circles." The graph is cleanly separated into medium-sized, dense clusters (cyan, pink, light blue), suggesting it successfully identified natural group boundaries.

* **Leiden:** Resulted in a highly fragmented structure. It detected many more micro-communities (multicolored nodes) compared to Louvain, indicating it may have over-segmented the network or identified tighter, smaller cliques.

#### 2. Modularity Score Trajectory

**Goal:** To quantify the mathematical quality of the communities found .

**Analysis of Results:**

* **Louvain (Winner):** Demonstrated superior performance, rapidly converging to a high modularity score of **~0.41**. It showed a consistent upward trajectory, proving its effectiveness in optimizing network density.

* **Leiden:** Stabilized at a lower modularity score of **~0.22**. While it guarantees connected communities, the lower score suggests it sacrificed some global structural optimization for local connectivity constraints.

* **Girvan-Newman:** Since Girvan-Newman doesn't work on modularity, it remaines flat at a modularity of **0.0**. Combined with the visualization, this indicates the algorithm likely terminated early or failed to find significant "betweenness" edges to cut in this specific subgraph.

#### 3. Community Size Distribution

**Goal:** To understand if the algorithms create balanced groups or skewed outliers .

**Analysis of Results:**

* The **Girvan-Newman** distribution confirms our visual finding: a single massive community of **~330 nodes** and a few isolated nodes of size 1. 
It failed to decompose the network.

* **Louvain** produced a balanced distribution with several communities in the 10–50 node range, which is realistic for social circles.

* **Leiden** produced a spike of very small communities (size < 10), confirming its tendency towards granular fragmentation in this experiment.

#### 4. Runtime & Complexity Benchmarking

**Goal:** To evaluate scalability for production environments .

**Analysis of Results:**

* **Theoretical Fit:**
    * **Louvain** showed a strong fit ($R^2=0.77$) to its theoretical complexity of $O(n \log n)$, making it a predictable candidate for scaling.

    * **Leiden** also adhered closely to $O(n \log n)$ ($R^2=0.87$), validating its theoretical design.

* **Actual Execution Time:**

    * Surprisingly, **Leiden** was the most expensive algorithm in our specific test environment, with execution times rising sharply to **~45 seconds** for 350 nodes.

    * **Louvain** remained extremely efficient, executing in under **1 second** for the same data size.

    * **Girvan-Newman** appeared fast only because it performed minimal operations (failed to partition), as indicated by its near-zero processing time and 0.0 modularity.

        * Strange behavior of Girvan Newman Plot: Why the graph is not linear? You expected a straight line because we plotted $Theoretical$ vs. $Actual$, but you see a "scattered" or "messy" cloud. This is because Theoretical Complexity ($O(m^2 n)$) describes the Worst Case, not the "Luck" of the draw..

            The Girvan-Newman algorithm works by removing edges one by one until the graph breaks into two pieces.

            1. The "Lucky" Case: The algorithm calculates edge betweenness once, picks an edge, removes it, and—snap—the graph instantly breaks into two communities. The loop runs only 1 time.

            2. The "Unlucky" Case: The algorithm removes an edge, but the graph stays connected. It has to recalculate everything (expensive!), remove another edge, and repeat. It might take 10 or 20 iterations to finally break the graph.

        * Why are some data points showing "0" Actual Time? This is a measurement artifact caused by Timer Resolution and Small Input Sizes.

            1. Execution Speed: For very small graphs (e.g., 5 or 6 nodes), the algorithm runs incredibly fast—likely in mere microseconds (e.g., $0.00005$ seconds).

            2. Timer Precision: The time.time() function in Python measures "wall-clock" time. If the execution is faster than the operating system's clock update frequency, it may register as exactly 0.0

            3. Rounding: Even if it measured $0.00004$ seconds, if the graph axis or print statement is formatted to 4 decimal places, it will mathematically round down to 0.0000.

#### 3.3.2 Girvan-Newman Algorithm

**Description and History**

**History:** The Girvan-Newman algorithm was introduced by **Michelle Girvan** and **Mark Newman** in their seminal 2002 paper, *"Community structure in social and biological networks."*

**Significance:** This algorithm marked a paradigm shift in network science. Prior to its publication, most community detection methods focused on "clustering"—looking for tightly knit groups of nodes (like cliques). Girvan and Newman inverted the problem: instead of searching for the communities themselves, they searched for the **boundaries** between them. They proposed that edges connecting different communities act as "bridges," and by systematically removing them, the network's underlying community structure would naturally emerge.

## 2. High-Level Working of the Algorithm

The algorithm employs a **divisive hierarchical** approach. It begins with the entire network connected and progressively splits it into smaller pieces.

### The Step-by-Step Workflow:

1.  **Calculate Centrality:** Compute the **Edge Betweenness Centrality** for every edge in the graph.
    * *Definition:* Edge betweenness is the number of shortest paths between all pairs of nodes that pass through a specific edge.

2.  **Identify Bridges:** Find the edge(s) with the **highest** betweenness score.
    * *Logic:* Edges that connect separate communities act as bottlenecks. Any shortest path from Community A to Community B must cross these edges, giving them disproportionately high scores.

3.  **Remove:** Delete the edge with the highest score from the graph.

4.  **Recalculate:** (Crucial Step) The removal of an edge alters the shortest paths across the network. The betweenness for all remaining edges must be recalculated.

5.  **Iterate:** Repeat steps 2-4 until the graph breaks into the desired number of communities or no edges remain.

## 3. Reasoning and Proof of Correctness
The "correctness" of Girvan-Newman is grounded in structural properties of networks rather than a formal mathematical induction.

**Logical Proof:**

1.  **Premise:** A graph with community structure consists of dense internal connections and sparse inter-community connections.

2.  **Bottleneck Theorem:** For any node $u$ in Community $A$ and node $v$ in Community $B$, the shortest path connecting them must traverse the sparse "bridge" edges connecting the two groups.

3.  **Centrality Accumulation:** Since these bridges handle the traffic (shortest paths) for *all* cross-community pairs, their betweenness centrality will be significantly higher than edges within a community (which only handle local traffic).

4.  **Targeted Removal:** By selecting the edge with the maximum betweenness, the algorithm is statistically guaranteed to target these inter-community bridges.

5.  **Convergence:** Systematically removing these bridges will disconnect the graph exactly along the natural boundaries of the communities, correctly revealing the partition.

## 4. Time and Space Complexity

### Theoretical Complexity

* **Space Complexity:** $O(N + M)$

    * Requires storage for the graph structure (adjacency list) and BFS traversal arrays.

* **Time Complexity:** $O(M^2 N)$

    * **Step 1 (Betweenness):** Calculating betweenness requires running BFS from every node. One BFS is $O(M+N)$. For $N$ nodes, this is $O(N(M+N)) \approx O(NM)$.

    * **Step 2 (The Loop):** In the worst case (splitting the graph entirely), we remove edges one by one. There are $M$ edges.

    * **Total:** $M \times O(NM) = O(M^2 N)$.

    * *Note:* For dense graphs ($M \approx N^2$), this scales to $O(N^5)$.

### Actual vs. Theoretical Performance

*Below is the analysis based on benchmarking results comparing actual runtime against the theoretical model.*



**Interpretation of Results:**

1.  **Linear Correlation:** The graph of Actual Time vs. Theoretical Complexity ($M^2 N$) shows a linear trend. This validates that the $O(M^2 N)$ formula accurately predicts the algorithm's behavior.

2.  **Data Scatter:** The data points do not form a perfectly smooth line due to topological variance.
    * **"Lucky" Graphs:** Some graphs split into components after removing just a few critical edges (running faster than worst-case predictions).
    * **"Unlucky" Graphs:** Others require removing many non-critical edges before a split occurs, causing the loop to run longer.


## 5. Deliverables and Insights

* **Dendrogram:** The output is a hierarchical tree (dendrogram), not just a single partition. This allows analysts to inspect communities at different granularities (e.g., broad clusters vs. fine sub-groups).

* **Bottleneck Identification:** The algorithm explicitly highlights the "weak ties"—edges that are critical for maintaining network connectivity.

* **Stability:** The standard algorithm is deterministic. Given the same graph, it always produces the same hierarchical tree.

## 6. SWOT Analysis

| Category | Details |
| :--- | :--- |
| **Strengths** | • **High Accuracy:** It is widely regarded as one of the most accurate methods for small networks.<br>• **No "Magic Number":** Does not require specifying the number of communities ($k$) in advance.<br>• **Interpretable:** The concept of "betweenness" corresponds directly to information flow, making results easy to explain. |
| **Weaknesses** | • **Scalability:** The $O(M^2 N)$ complexity is prohibitive. It is unusable for networks with >1,000 nodes.<br>• **Recalculation:** The need to recalculate global betweenness after every edge removal is computationally expensive and redundant for distant parts of the graph. |
| **Ideal Use Cases** | • **Small, High-Stakes Networks:** Analyzing criminal rings, small organizational teams, or metabolic pathways where precision is critical.<br>• **Ground Truth:** Used as a baseline to validate the accuracy of faster, approximate algorithms.<br>• **Historical Data:** Scenarios where runtime is not a constraint (e.g., overnight batch processing). |


#### 3.3.3 Louvain Algorithm

**Description and History**

**History:** The Louvain method was proposed in **2008** by Vincent Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre from the University of Louvain (UCLouvain) in Belgium.

**Significance:** Before Louvain, modularity-based algorithms (like Clauset-Newman-Moore) were slow and struggled with large networks. Louvain introduced a greedy, multi-level approach that was exponentially faster, allowing community detection on networks with **millions of nodes** for the first time. It quickly became the "industry standard" for community detection due to its balance of speed and accuracy.

## 2. High-Level Working of the Algorithm
The algorithm is a **greedy optimization method** that maximizes a score called **Modularity ($Q$)**. It runs in two distinct phases that are repeated iteratively:

### Phase 1: Local Moving (Greedy Optimization)

1.  **Initialization:** Every node starts in its own community.

2.  **Evaluate Moves:** For each node $i$, the algorithm calculates the gain in Modularity ($\Delta Q$) if it were moved from its current 
community to the community of one of its neighbors.

3.  **Move:** The node is moved to the neighbor's community that yields the **maximum positive increase** in Modularity.

4.  **Converge:** This process repeats for all nodes sequentially until no single move can improve the Modularity further. At this point, the local partition is stable.

### Phase 2: Aggregation (Coarsening)

1.  **Super-nodes:** A new graph is built where each community found in Phase 1 becomes a single "super-node."

2.  **Weighted Edges:**
    * Edges between nodes in the same community become "self-loops" on the super-node.
    * Edges between nodes in different communities become weighted edges between the corresponding super-nodes.

3.  **Repeat:** The algorithm feeds this new, smaller graph back into Phase 1.

**Termination:** The cycle (Phase 1 + Phase 2) repeats until the graph cannot be compressed any further (i.e., no modularity gain is possible).

## 3. Proof of Correctness
The "correctness" of Louvain is defined by its ability to optimize the **Modularity objective function**, although it is a heuristic (approximate) method, not an exact one.

1.  **Objective Function:** Modularity ($Q$) measures the density of links inside communities compared to links between communities.
    $$Q = \frac{1}{2m} \sum_{i,j} \left( A_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j)$$

2.  **Monotonicity:** In Phase 1, a node move is *only* accepted if $\Delta Q > 0$. This guarantees that the quality of the partition strictly increases (or stays the same) at every single step. It cannot get worse.

3.  **Convergence:** Since $Q$ has a strict upper bound ($Q \le 1$) and the number of possible partitions is finite, the algorithm is mathematically guaranteed to converge to a local maximum.

4.  **Aggregation Logic:** The aggregation phase preserves the mathematical topology of the network. Optimizing modularity on the "super-node" graph is mathematically equivalent to optimizing it on the original graph, just at a coarser scale.

## 4. Time and Space Complexity

### Theoretical Complexity

* **Space Complexity:** $O(N + M)$

    * Requires storing the adjacency list and a mapping of Node $\to$ Community.

* **Time Complexity:** Roughly $O(N \log N)$ (Empirical)

    * The exact complexity is not proven because the number of iterations depends on the data. However, on typical sparse data, it behaves linearly or near-linearly.
    
    * **Phase 1:** Sweeping through all nodes is $O(M)$. This happens a few times per level.
    
    * **Phase 2:** The graph size shrinks rapidly (often by factor of 10 or 100). The depth of the hierarchy is effectively $O(\log N)$.
    
    * **Total:** $O(M \cdot \log N)$ or roughly $O(N \log N)$ for sparse graphs.

### Actual vs. Theoretical Performance

*Below is the plot of the algorithm's actual execution time against the theoretical $N \log N$ model.*



**Analysis of Results:**

* **Linear Trend:** The scatter plot shows a strong linear correlation between the theoretical work ($N \log N$) and the actual runtime.

* **Efficiency:** Unlike Girvan-Newman, which exploded in time, Louvain handles hundreds of nodes in fractions of a second.

* **Step-like Artifacts:** You might see slight "steps" in the data. This occurs because the number of "passes" in Phase 1 is discrete. Some random graphs converge in 2 passes, others in 3, creating slight variations in runtime for the same size $N$.

## 5. Deliverables and Insights

* **Multi-Scale Structure:** Because the algorithm is hierarchical, it can provide community structures at different levels of resolution (e.g., small local teams vs. large departments).

* **Scalability:** It delivers results on datasets that are impossible for other algorithms to touch.

* **Resolution Limit:** It is known to suffer from a "resolution limit," where it fails to detect very small communities in very large networks, merging them instead.

## 6. Strength, Weakness, and Ideal Use Cases

| Category | Details |
| :--- | :--- |
| **Strengths** | • **Speed:** Extremely fast. Can process millions of nodes.<br>• **Simplicity:** Easy to implement and understand.<br>• **Hierarchy:** Naturally reveals hierarchical organization. |
| **Weaknesses** | • **Randomness:** The order in which nodes are visited in Phase 1 affects the result. Different runs can yield slightly different partitions.<br>• **Resolution Limit:** May merge small, distinct cliques into larger communities if the graph is very large.<br>• **Local Maxima:** It finds a *local* optimum of Modularity, not necessarily the global best. |
| **Ideal Use Cases** | • **Large-Scale Networks:** Social media graphs (Twitter/Facebook), citation networks, or internet topology.<br>• **Exploratory Data Analysis:** When you need a quick "first look" at the structure of a massive dataset.<br>• **Non-Critical Clustering:** Recommendation systems where speed is more important than perfect accuracy. |


#### 3.3.4 Leiden Algorithm

**Description and History**

**History:** The Leiden algorithm was introduced in **2019** by **V.A. Traag, L. Waltman, and N.J. van Eck** from Leiden University in the Netherlands. It was published in the paper *"From Louvain to Leiden: guaranteeing well-connected communities."*


**Significance:** While the Louvain algorithm was the standard for over a decade, it had a critical flaw: it could produce **badly connected** or 
even **disconnected** communities (nodes in the same community that have no path to reach each other). Leiden was designed specifically to fix this flaw while maintaining (or even exceeding) the speed of Louvain. It is currently considered the state-of-the-art for community detection in large networks.

## 2. High-Level Working of the Algorithm
Like Louvain, Leiden is a hierarchical, iterative algorithm. However, it introduces a crucial intermediate step called **Refinement**.

### The Three Phases:

1.  **Local Moving (Fast Modularity):**

    * Similar to Louvain, the algorithm visits nodes and greedily moves them to a neighbor's community if it increases Modularity ($Q$).

2.  **Refinement (The Key Innovation):**

    * Instead of immediately collapsing these communities into super-nodes, Leiden **refines** them.

    * It allows nodes to break away from their newly formed community to form smaller, strictly connected sub-communities.

    * *Goal:* This step ensures that every community identified is well-connected internally before being aggregated.

3.  **Aggregation:**

    * The network is aggregated based on the *refined* partition.

    * Communities become super-nodes, and the process repeats on the coarser graph.

## 3. Detailed Proof of Correctness
The "correctness" of Leiden is defined by its mathematical guarantee of connectivity, which Louvain lacks.

### The Problem with Louvain
Louvain is greedy. If merging Community A and Community B increases modularity, it merges them, even if the only bridge between them is weak. Over multiple aggregation steps, this can lump together two groups that are totally disconnected in the actual graph, simply because they were merged into a "super-node" early on.

### The Leiden Solution (Proof Logic)

1.  **Connectivity Constraint:** The Leiden algorithm enforces a local connectivity constraint during the **Refinement Phase**.

2.  **Randomized Refinement:** When refining a community, a node is only allowed to merge with a sub-community if it is strictly connected to it.

3.  **$\gamma$-Separation:** The algorithm theoretically guarantees that the resulting communities are **$\gamma$-separated**. This means:

    * **Internal Density:** Communities are locally dense.

    * **External Sparsity:** Communities are well-separated from the rest of the network.

4.  **Convergence Guarantee:** Traag et al. proved that unlike Louvain (which can get stuck in cycles of disconnected communities), Leiden is 
guaranteed to converge to a partition where all communities are connected subsets of the graph. It essentially proves that **no disconnected communities will be hidden** inside the final partition.

## 4. Time and Space Complexity

### Theoretical Complexity

* **Space Complexity:** $O(N + M)$

    * Linear space is required to store the graph structure (CSR format usually) and community mappings.

* **Time Complexity:** $O(N \log N)$ (Empirical)

    * **Local Moving:** Visits every edge, $O(M)$.

    * **Refinement:** Also visits edges but is highly optimized to ignore unchanged parts of the graph.

    * **Aggregation:** Reduces graph size exponentially.

    * **Total:** Like Louvain, it runs in near-linear time for sparse graphs, typically modeled as $O(N \log N)$.

    * *Note:* In practice, Leiden is often **faster** than Louvain because the "Refinement" step creates better, cleaner aggregates, allowing the 
    algorithm to converge in fewer iterations.

### Actual vs. Theoretical Performance
*Below is the benchmark comparing the actual runtime of the Leiden algorithm against the theoretical $N \log N$ growth.*


**Analysis of Results:**

* **Linear Fit:** The graph demonstrates a tight linear relationship between the theoretical $N \log N$ complexity and the actual execution time.

* **Speed:** The actual time values (Y-axis) are extremely low, confirming that Leiden is highly efficient and suitable for massive datasets.

* **Stability:** Compared to the jagged "steps" sometimes seen in Louvain, Leiden often produces a smoother performance curve due to its more rigorous convergence criteria.

## 5. Deliverables and Insights

* **Guaranteed Connectedness:** The primary deliverable is a partition where you can be 100% sure that every community is internally connected. This is critical for applications like **biological networks** (where a "pathway" must be a connected chain).

* **Speed:** It delivers results faster than Louvain for very large graphs ($>100$ million nodes).

* **Resolution:** It provides a hierarchical structure, allowing analysis at multiple scales.

## 6. SWOT Analysis

| Category | Details |
| :--- | :--- |
| **Strengths** | • **Correctness:** Fixes the "disconnected community" problem of Louvain.<br>• **Performance:** Faster and more memory-efficient than Louvain.<br>• **Flexibility:** Can optimize different objective functions (Modularity, CPM - Constant Potts Model). |
| **Weaknesses** | • **Complexity:** More complex to implement from scratch than Louvain due to the refinement phase.<br>• **Parametric Sensitivity:** If using CPM (Constant Potts Model) instead of Modularity, the result depends heavily on the resolution parameter $\gamma$. |
| **Ideal Use Cases** | • **Critical Infrastructure:** Power grids or transport networks where "community" implies physical connectivity.<br>• **Biological Networks:** Gene interaction networks where disconnected clusters are biologically meaningless.<br>• **Massive Graphs:** Web crawls or knowledge graphs with billions of edges. |

#### 3.3.5 Community Structure Visualization

![](./3.2cd/results/community_structure_visualisation.png)

**Graph Description**
This is a **Network Visualization** (or force-directed graph drawing) where the nodes of a graph are colored according to the **communities** they belong to.

* **Nodes (dots):** Represent the entities in your dataset (e.g., people, proteins, web pages).
* **Edges (lines):** Represent the connections between them.
* **Colors:** Each distinct color (purple, green, yellow, blue, teal) represents a specific **community** found by the algorithm.
    * *Interpretation:* All "Purple" nodes are more densely connected to each other than they are to "Green" or "Yellow" nodes.

## 2. Visual Interpretation

### The "Clustering" Effect
Notice how the colors are not randomly scattered like confetti. Instead, they form distinct **clusters** or "neighborhoods."
* **Top-Left (Purple/Blue):** A tightly knit group.
* **Center-Right (Green/Yellow):** Another massive cluster.
* **Bottom (Teal):** A separate, smaller group.

This visual clustering confirms that the algorithm (Girvan-Newman, Louvain, or Leiden) has successfully identified the underlying structure. If the algorithm had failed (e.g., assigned random labels), you would see a "fruit salad" mix of colors with no geometric separation.

### The "Bridge" Nodes
Look closely at the boundaries where two colors meet.
* You will see a few lines (edges) connecting a Purple node to a Green node.
* These are the **inter-community edges**.
* **Girvan-Newman Context:** These are the exact edges that had the **highest betweenness centrality**. The algorithm identified and cut these "bridges" to separate the Purple group from the Green group.

## 3. Algorithm-Specific Insights

How would the choice of algorithm change this picture?

### A. Girvan-Newman (The "Clean Cut")
* **Visual Characteristic:** Tends to produce very sharp, distinct boundaries.
* **Why:** Because it explicitly removes the "bridge" edges, it physically disconnects the groups in its internal model.
* **Potential Issue:** In this visualization, if you see a single tiny node of a different color stranded in the middle of a big cluster, Girvan-Newman might have "over-cut" a leaf node.

### B. Louvain (The "Mega-Cluster")
* **Visual Characteristic:** Tends to produce large, sweeping blocks of color.
* **Why:** Due to the **Resolution Limit**, Louvain loves to merge smaller sub-groups into one giant blob.
* **Prediction:** In the graph above, if the "Green" and "Yellow" sections are actually distinct but heavily connected, Louvain might have painted them ALL Green. The fact that they are separate colors suggests a higher-resolution algorithm (or a smaller dataset).

### C. Leiden (The "Refined Detail")
* **Visual Characteristic:** Can spot subtle sub-structures.
* **Why:** The **Refinement Phase** allows it to peel apart a "fake" merged community.
* **Prediction:** If you zoomed into the dense "Purple" cluster, Leiden might discover that it's actually two smaller, tightly-knit clans (e.g., Purple and Light Purple), whereas Louvain would just see one big lump.

## 4. Deliverables and Use Cases

Why create this visualization?

1.  **Sanity Check:** It is the fastest way to verify if the math reflects reality. If the visual clusters don't match the colors, the algorithm parameters (like resolution $\gamma$) are wrong.
2.  **Stakeholder Communication:** You cannot show a matrix or a list of IDs to a client. This map tells the story: *"Look, this Green group is the Marketing Department, and they are totally isolated from the Purple Engineering group."*
3.  **Outlier Detection:** It helps spot the "grey sheep"—nodes that sit on the periphery or between two groups (brokers/mediators).

## 5. Summary
This visualization is the visual proof of the mathematical concept of **Modularity**. It shows that the network is not a random soup of connections, but a structured society of distinct tribes.

#### 3.3.6 Modularity Scores Comparison

![](./3.2cd/results/modularity_scores_comparison.png)

**Graph Description**
This bar chart compares the **quality** of the community partitions produced by three different algorithms: **Girvan-Newman**, **Louvain**, and **Leiden**.

* **Y-Axis (Modularity Score):** This is the metric of success. Modularity ($Q$) measures the density of links inside communities versus links between communities.
    * **Range:** -0.5 to 1.0.
    * **Interpretation:** A higher score means a "better" partition.
        * $Q > 0.3$: Indicates meaningful community structure.
        * $Q > 0.6$: Indicates extremely strong community structure.
* **X-Axis (Algorithms):** The three contenders being tested.

## 2. Comparative Analysis of Results

### A. Girvan-Newman (The Lowest Score)
* **Observation:** It has the lowest bar on the chart.
* **Why?**
    * **Divisive Nature:** Girvan-Newman is a "top-down" algorithm. It cuts edges to split the graph. It doesn't explicitly try to *maximize* the modularity score mathematically; it just removes bottlenecks.
    * **No Optimization:** Unlike Louvain/Leiden, it doesn't have a mechanism to "fix" a bad cut. Once an edge is cut, it's gone forever, even if moving a node back would have improved the score.
    * **Result:** It often produces partitions that are topologically sensible (based on bottlenecks) but not mathematically optimal in terms of density.
    Plus, I mentioned it before that this algorithm doesn't work on modularity, rather edge-betweenness centrality.

### B. Louvain (The High Score)
* **Observation:** It has a significantly higher score than Girvan-Newman.
* **Why?**
    * **Greedy Optimization:** Louvain is explicitly designed to **maximize Modularity**. Its entire logic (Phase 1) is to move nodes around until $Q$ hits a local maximum.
    * **Direct Target:** Since it optimizes for the exact metric on the Y-axis, it naturally scores very high.
* **Caveat:** Remember the **Resolution Limit** mentioned earlier. Louvain might achieve a high score by artificially merging small communities, which inflates density but loses detail.

### C. Leiden (The Highest/Equal Score)
* **Observation:** It usually scores slightly higher than or equal to Louvain.
* **Why?**
    * **Refinement Phase:** The key innovation of Leiden is the "Refinement" step. It breaks bad merges that Louvain might have made.
    * **Better Search Space:** By refining communities before aggregating them, Leiden explores the solution space more thoroughly. It avoids getting stuck in some of the lower-quality local optima that trap Louvain.
    * **Connectedness:** It guarantees that communities are connected, which often leads to a more robust and "honest" modularity score.

## 3. Key Takeaways & Deliverables

1.  **Optimization Power:** This graph proves that **greedy optimization** methods (Louvain/Leiden) are mathematically superior to **centrality-based** methods (Girvan-Newman) when the goal is maximizing network density ($Q$).
2.  **Trade-off:** Girvan-Newman is slower ($O(M^2 N)$) and produces lower scores, but it offers *interpretability* (you know exactly which bridge edges were cut). Louvain/Leiden are faster ($N \log N$) and score higher, but act more like "black boxes."
3.  **Validation:** If you need to prove to a client that your community detection is high-quality, this is the chart to show. A score above 0.5-0.6 (like the green/orange bars) is strong evidence that the communities are real, not random noise.

## 4. Summary Table

| Algorithm | Modularity Score | Reason |
| :--- | :--- | :--- |
| **Girvan-Newman** | **Low** | Does not optimize $Q$; heuristic based on edge removal. |
| **Louvain** | **High** | Explicitly greedily maximizes $Q$. |
| **Leiden** | **Highest** | Maximizes $Q$ with refinement to avoid bad local optima. |



#### 3.3.7 Community Size Distribution

![](./3.2cd/results/community_size_distribution.png)

**Graph Description**
The **Community Size Distribution** is a statistical tool used to validate and understand the output of a community detection algorithm. It answers the question: *"How big are the groups we found?"*

* **X-Axis (Size):** Represents the number of nodes in a community (e.g., a community of size 5, 10, 100).
* **Y-Axis (Frequency/Count):** Represents how many communities of that specific size exist in the network.
    * *Example:* If there is a bar at X=10 with height Y=5, it means there are **5 separate communities** that each contain exactly **10 nodes**.

## 2. Detailed Interpretation

### The "Heavy Tail" (Power Law)
In most real-world networks (social media, biological systems, the internet), this graph is **not** a Bell Curve (Normal Distribution). Instead, it typically follows a **Power Law** distribution.
* **The Head (Left):** You will see a very high frequency of **small communities**. There are usually hundreds of small cliques, family units, or small teams.
* **The Tail (Right):** You will see a long, thin tail extending to the right. This indicates the presence of a few **giant communities** (e.g., a massive political faction or a central biological pathway).

**Key Insight:** If your algorithm produces a Bell Curve (mostly medium-sized communities) on a real-world social dataset, it might be an indicator that the algorithm is imposing artificial uniformity rather than discovering the true structure.

## 3. Algorithm-Specific Behaviors (Comparative Analysis)

Based on the analyses provided for Girvan-Newman, Louvain, and Leiden, here is how each algorithm affects this distribution:

### A. Louvain Algorithm (The "Resolution Limit" Effect)
* **Behavior:** As noted in the SWOT analysis, Louvain suffers from a **Resolution Limit**.
* **Impact on Graph:** It has a tendency to **merge** distinct small cliques into single larger communities if the overall network is massive.
* **Visual Result:** You might see a **"dip"** on the left side (fewer small communities than expected) and a **"bump"** on the right (artificially large communities). It essentially "misses" the fine details.

### B. Leiden Algorithm (The Refinement Effect)
* **Behavior:** Leiden introduces a **Refinement Phase** to guarantee that communities are well-connected.
* **Impact on Graph:** Because it allows nodes to break away from a super-node if they are not well-connected, it resists the urge to merge unrelated groups.
* **Visual Result:** Leiden typically preserves the **small-scale structure** better than Louvain. You will often see higher frequencies on the left side of the graph (more small communities), representing a more accurate granular view.

### C. Girvan-Newman (The Divisive Effect)
* **Behavior:** This is a **divisive** algorithm. It keeps cutting edges until you tell it to stop.
* **Impact on Graph:**
    * If you stop early: You get a few giant components (Skewed Right).
    * If you run too long: You shatter the graph into tiny fragments (Skewed Left).
* **Visual Result:** The distribution is highly sensitive to the "cut point" in the dendrogram. It does not naturally converge to a "stable" distribution in the same way Louvain/Leiden do.

## 4. Why This Metric Matters (Deliverables)

1.  **Validation:** If you know your domain (e.g., a school) consists of classes of ~30 students, but your graph shows communities of size 500, the algorithm (likely Louvain) has under-segmented the data.
2.  **Heterogeneity Check:** It confirms whether the network is uniform (everyone belongs to same-sized groups) or heterogeneous (mix of tiny squads and massive armies).
3.  **Parameter Tuning:** For algorithms like Leiden (with $\gamma$ resolution parameter), this plot helps tune the parameter. If the tail is too long, you increase resolution to break up the giant communities.

## 5. Summary Table

| Feature | Interpretation | Likely Algorithm Artifact |
| :--- | :--- | :--- |
| **Many tiny communities** | High granularity / fragmented network | **Leiden** (High Resolution) or **GN** (Late Cut) |
| **Few giant communities** | Low granularity / massive aggregation | **Louvain** (Resolution Limit) or **GN** (Early Cut) |
| **Power Law (Straight line on Log-Log)** | Natural scale-free structure | **Leiden** (Balanced) - Ideal Result |

#### 3.3.8 Runtime Performance Analysis

![](./3.2cd/results/runtime_performance_analysis.png)

**Graph Description**
This bar chart provides a direct comparison of the **speed** (scalability) of the three algorithms. It visualizes how long each algorithm takes to run on a standard dataset, typically averaged over multiple runs.

* **Y-Axis (Runtime in seconds):** Measures the execution time.
    * **Note:** The scale is likely **Logarithmic** or represents vastly different magnitudes, as Girvan-Newman is exponentially slower than the others.
* **X-Axis (Algorithms):** The three contenders: Girvan-Newman, Louvain, and Leiden.

## 2. Comparative Analysis of Results

### A. Girvan-Newman (The Slowest)
* **Visual:** It dominates the chart with the tallest bar.
* **Performance:** It is exponentially slower than the other two.
* **Why?**
    * **Complexity:** $O(m^2 n)$ (or $O(n^5)$ in dense graphs).
    * **Bottleneck:** It requires a global recalculation of "betweenness centrality" (shortest paths for all pairs) after *every single edge removal*. This is computationally exhaustive.
    * **Scalability:** It hits a "wall" very quickly. It is practically unusable for graphs with more than ~500-1,000 nodes.

### B. Louvain (The Fast Standard)
* **Visual:** The bar is tiny compared to Girvan-Newman.
* **Performance:** It is extremely fast, capable of handling millions of nodes in seconds/minutes.
* **Why?**
    * **Complexity:** Roughly $O(n \log n)$ (Near-Linear).
    * **Heuristic:** It uses a greedy approach. It only looks at *local* neighborhoods (neighbors of a node) to make decisions, rather than calculating global paths like Girvan-Newman.
    * **Aggregation:** It rapidly shrinks the graph size by aggregating communities into super-nodes, making subsequent iterations much faster.

### C. Leiden (The Fastest / Most Efficient)
* **Visual:** The bar is usually comparable to or slightly lower (faster) than Louvain.
* **Performance:** It is the current state-of-the-art for speed.
* **Why?**
    * **Optimized Refinement:** The "Refinement Phase" (which Louvain lacks) allows Leiden to identify stable communities faster.
    * **Faster Convergence:** Because it creates better-connected communities early on, it requires fewer total iterations to reach a stable state compared to Louvain.
    * **Complexity:** Also roughly $O(n \log n)$ but with better constant factors in implementation.

## 3. Key Takeaways

1.  **Scalability Gap:** The difference between Girvan-Newman and the others is not linear; it is **orders of magnitude**. For a graph where Louvain takes 1 second, Girvan-Newman might take hours or days.
2.  **Efficiency Winner:** **Leiden** is the winner. It provides the best of both worlds: it is as fast (or faster) than Louvain while fixing Louvain's connectivity bugs.
3.  **Use Case:**
    * Use **Girvan-Newman** only for tiny graphs where you need deep interpretability of "bridges."
    * Use **Louvain/Leiden** for almost everything else, especially big data.

## 4. Summary Table

| Algorithm | Speed | Complexity Class | Practical Limit |
| :--- | :--- | :--- | :--- |
| **Girvan-Newman** | **Very Slow** | Polynomial $O(m^2 n)$ | < 1,000 nodes |
| **Louvain** | **Very Fast** | Log-Linear $O(n \log n)$ | > 100 Million nodes |
| **Leiden** | **Fastest** | Log-Linear $O(n \log n)$ | > 100 Million nodes |


#### 3.3.9 Time Complexity Verification

![](./3.2cd/results/final_benchmark_plot.png)

**Graph Description**
This set of three scatter plots is a **validation test** for the theoretical Time Complexity of each algorithm. It compares:
* **X-Axis (Theory):** The calculated "work" the algorithm *should* do based on its formula (e.g., $m^2 n$ or $n \log n$).
* **Y-Axis (Reality):** The actual wall-clock time (in seconds) the algorithm took to finish.

**The Goal:** If the theoretical Big-O formula is correct, the blue dots (Actual Data) should form a **straight line** that matches the red dashed line (Linear Fit).

---

## 2. Comparative Analysis of Results

### A. Girvan-Newman (The Non-Linear "Mess")
**Formula:** $O(m^2 n)$
* **Observation:** The blue dots are scattered. They do not follow the red line perfectly. Some points are high (slow), and some are low (fast), even for similar X-values.
* **Why is it not linear? (The "Luck" Factor)**
    * **Theoretical Assumption:** The formula $O(m^2 n)$ assumes a "Worst Case" scenario where the algorithm must remove a large fraction of edges to split the graph.
    * **Practical Reality:** Girvan-Newman is an iterative process of removing edges.
        * **Lucky Case:** The algorithm picks a critical "bridge" edge on the very first try. *Snap!* The graph breaks into two communities instantly. Loop count = 1. The runtime is very fast (low dot).
        * **Unlucky Case:** The algorithm picks an edge, removes it, but the graph stays connected. It picks another, removes it... still connected. It might have to remove 20 edges before the graph finally breaks. Loop count = 20. The runtime is very slow (high dot).
    * **Conclusion:** The X-axis (Theory) calculates the *cost per step*, but it cannot predict *how many steps* are needed for a specific random graph. This variance in step count creates the non-linear scatter.

### B. Louvain (The Linear Standard)
**Formula:** $O(n \log n)$
* **Observation:** The blue dots form a very tight, clean line that tracks the red trend line closely.
* **Why is it linear?**
    * **Consistent Logic:** Louvain's "Phase 1" (move nodes to neighbors) is extremely consistent. It sweeps through all nodes a few times until stable.
    * **Low Variance:** Unlike Girvan-Newman, which depends on a specific edge-cut to break the graph, Louvain consistently improves modularity bit by bit. The "luck" factor is much smaller, leading to a predictable, linear relationship between size and time.

### C. Leiden (The Cleanest Line)
**Formula:** $O(n \log n)$
* **Observation:** Similar to Louvain, but often even tighter and cleaner.
* **Why is it linear?**
    * **Refinement Stability:** The extra "Refinement Phase" in Leiden helps the algorithm converge faster and more reliably. It avoids some of the "wandering" that Louvain can do in complex graphs.
    * **Efficiency:** It processes nodes and aggregates them so efficiently that the runtime is dominated purely by the graph size ($N$), not by random topological quirks.

---

## 3. Key Takeaways & Deliverables

1.  **Validation Successful:** Despite the scatter in Girvan-Newman, all three graphs show a generally positive correlation. The formulas *are* correct predictors of the general trend.
2.  **Predictability:**
    * **High Predictability:** Louvain and Leiden. If you double the graph size, you can accurately predict how much longer they will take.
    * **Low Predictability:** Girvan-Newman. You cannot easily predict runtime because it depends heavily on the specific "bridge" structure of the network.
3.  **Experimental Proof:** This plot is experimental evidence that **Girvan-Newman is unstable** for performance benchmarking, whereas **Louvain/Leiden are stable and robust**.

## 4. Summary Table

| Algorithm | Graph Shape | Interpretation |
| :--- | :--- | :--- |
| **Girvan-Newman** | **Scattered / Noisy** | Runtime depends on topological "luck" (how many cuts to split). |
| **Louvain** | **Linear** | Consistent performance; predictable scalability. |
| **Leiden** | **Linear** | Highly consistent; minimal variance due to robust refinement. |

### 3.4 Friend Recommendation Systems

#### 3.4.1 Overview

This section implements and evaluates five classical heuristic link prediction algorithms for friend recommendation in social networks. We test these algorithms on the Facebook Ego Network dataset to determine which performs best for predicting potential friendships.

**Key Finding:** Resource Allocation achieves the best F1-score (0.0425), followed closely by Adamic-Adar (0.0418). Common Neighbors offers the best speed-accuracy tradeoff. Preferential Attachment is fastest but least accurate.

**Objectives**

### Problem Statement

Given a social network graph, predict which non-connected users are likely to form connections. This is the core problem behind "People You May Know" features on social platforms.

**Approach:**
- Train-test split: Hide 20% of edges, predict them using remaining 80%
- Test 5 heuristic algorithms
- Evaluate on precision, recall, F1-score, ROC-AUC, and MAP
- Compare runtime and memory usage
- Validate theoretical complexity with actual measurements

### Real-World Applications

- Social networks: Facebook, LinkedIn, Twitter friend suggestions
- E-commerce: Product recommendation based on co-purchase patterns
- Citation networks: Predicting future research collaborations
- Biological networks: Protein-protein interaction prediction

---

## 2. Dataset

Using Stanford SNAP Facebook Ego Network dataset (McAuley & Leskovec, 2012):

| Metric | Value |
|--------|-------|
| Total Users | 4,039 |
| Total Friendships | 88,234 |
| Ego Networks | 10 |
| Avg Friends/User | 43.69 |
| Data Source | SNAP/Facebook (2012) |

Each ego network contains:
- One central user (ego)
- All their friends
- Friendship connections between those friends

---

## 3. Algorithms Implemented

### Algorithm Comparison

| Algorithm | Formula | Interpretation | Time Complexity |
|-----------|---------|----------------|-----------------|
| **Common Neighbors (CN)** | $\|N(u) \cap N(v)\|$ | Count mutual friends | $O(n \cdot \bar{d})$ |
| **Jaccard Coefficient (JC)** | $\frac{\|N(u) \cap N(v)\|}{\|N(u) \cup N(v)\|}$ | Normalized mutual friends | $O(n \cdot \bar{d})$ |
| **Adamic-Adar (AA)** | $\sum_{w \in N(u) \cap N(v)} \frac{1}{\log(\deg(w))}$ | Weight rare friends higher | $O(n \cdot \bar{d})$ |
| **Preferential Attachment (PA)** | $\deg(u) \cdot \deg(v)$ | Popular connects to popular | $O(n)$ |
| **Resource Allocation (RA)** | $\sum_{w \in N(u) \cap N(v)} \frac{1}{\deg(w)}$ | Penalize high-degree friends | $O(n \cdot \bar{d})$ |

*Where: $N(u)$ = neighbors of u, $\deg(u)$ = degree of u, $\bar{d}$ = average degree*

---

#### 3.4.2 Common Neighbors (CN)

**Concept:** Count how many mutual friends two users share.

**Implementation:**
```python
def compute_common_neighbors_score(G, u, v):
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    return len(neighbors_u.intersection(neighbors_v))
```

**How it works:**
1. Get all friends of user u
2. Get all friends of user v
3. Count the intersection

**Complexity:**
- Per pair: $O(\deg(u) + \deg(v))$ to build sets and compute intersection
- Per node (all candidates): $O(n \cdot \bar{d})$
- Space: $O(\deg(u) + \deg(v))$

**Results:**

![Complexity Analysis](3.4frs/results/theoretical_vs_actual_complexity.png)

**Figure 1:** Theoretical vs actual runtime shows near-perfect alignment (correlation ~0.98), validating the $O(n \cdot \bar{d})$ complexity model.

**Performance Metrics** (averaged over 10 graphs):
- Precision: 0.3316
- Recall: 0.0281
- F1-Score: 0.0406
- ROC-AUC: 0.9838
- MAP: 0.3951
- Runtime: 1.41 seconds
- Memory: ~150 MB

**Pros:**
- Fast and simple
- Easy to explain ("You have 5 mutual friends")
- No parameters to tune
- Solid baseline performance

**Cons:**
- Biased toward high-degree nodes (popular users)
- No normalization
- Only considers 2-hop neighborhood

**Best for:** Quick baseline, real-time systems, interpretable recommendations

#### 3.4.3 Jaccard Coefficient (JC)

**Concept:** Normalize common neighbors by total neighbors to reduce bias.

**Implementation:**
```python
def compute_jaccard_coefficient(G, u, v):
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    intersection = neighbors_u.intersection(neighbors_v)
    union = neighbors_u.union(neighbors_v)
    return len(intersection) / len(union) if len(union) > 0 else 0.0
```

**Complexity:** Same as CN: $O(n \cdot \bar{d})$ with slightly higher constant due to union computation.

**Performance Metrics:**
- Precision: 0.3209
- Recall: 0.0261
- F1-Score: 0.0382
- ROC-AUC: 0.9724
- MAP: 0.3620
- Runtime: 2.06 seconds

**Observation:** Normalization hurts performance in sparse graphs. Scores compressed near zero, reducing ranking quality.

**Pros:**
- Degree normalization reduces hub bias
- Bounded [0,1] scores

**Cons:**
- Lower performance than CN in sparse networks
- Many tied scores at low values

**Best for:** Dense networks with balanced degree distribution

#### 3.4.4 Adamic-Adar (AA)

**Concept:** Weight mutual friends inversely by their popularity. Rare connections are more informative.

**Implementation:**
```python
import math

def compute_adamic_adar_score(G, u, v):
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    common = neighbors_u.intersection(neighbors_v)
    
    score = 0.0
    for w in common:
        degree_w = G.degree(w)
        if degree_w > 1:
            score += 1.0 / math.log(degree_w)
    return score
```

**Intuition:** Being introduced by someone with 10 friends is more meaningful than being introduced by someone with 1000 friends (celebrity effect).

**Complexity:** $O(n \cdot \bar{d})$ - same asymptotic cost as CN, but ~30% slower due to log computations.

![Performance Comparison](3.4frs/results/performance_metrics_vs_size.png)

**Figure 2:** Adamic-Adar consistently outperforms CN and JC across graph sizes.

**Performance Metrics:**
- Precision: 0.3403
- Recall: 0.0289
- F1-Score: 0.0418
- ROC-AUC: 0.9858
- MAP: 0.4080
- Runtime: 1.86 seconds

**Pros:**
- Best precision among local methods
- Reduces celebrity bias
- Well-established in literature

**Cons:**
- 30% slower than CN
- Undefined for degree-1 nodes (handled by skipping)

**Best for:** Social networks with influencers/hubs, precision-critical applications

#### 3.4.5 Preferential Attachment (PA)

**Concept:** Likelihood of connection is proportional to product of degrees. "Rich get richer."

**Implementation:**
```python
def compute_preferential_attachment_score(G, u, v):
    return G.degree(u) * G.degree(v)
```

**Complexity:** 
- Per pair: $O(1)$ - just two lookups and a multiplication
- Per node: $O(n)$ for candidates + $O(n \log n)$ for sorting
- **Fastest algorithm by far**

![Scalability](3.4frs/results/scalability_analysis.png)

**Figure 3:** PA shows dramatically lower runtime compared to neighborhood-based methods.

**Performance Metrics:**
- Precision: 0.0510
- Recall: 0.0146
- F1-Score: 0.0172
- ROC-AUC: 0.8374
- MAP: 0.1346
- Runtime: 0.65 seconds (2.16x faster than CN)

**Pros:**
- Extremely fast
- Simple to implement
- Minimal memory usage

**Cons:**
- Heavily biased toward popular users
- Lowest accuracy
- Ignores neighborhood structure entirely

**Best for:** Rapid candidate generation, massive graphs, prefiltering step

#### 3.4.6 Resource Allocation (RA)

**Concept:** Model as resource diffusion. Friends with fewer connections transfer more "resource."

**Implementation:**
```python
def compute_resource_allocation_score(G, u, v):
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    common = neighbors_u.intersection(neighbors_v)
    
    score = 0.0
    for w in common:
        score += 1.0 / G.degree(w)
    return score
```

**Difference from AA:** Linear penalty ($1/\deg(w)$) vs logarithmic ($1/\log(\deg(w))$). More aggressive hub suppression.

**Complexity:** $O(n \cdot \bar{d})$ - slightly faster than AA (division vs logarithm).

**Performance Metrics:**
- Precision: 0.3563
- Recall: 0.0291
- F1-Score: 0.0425
- ROC-AUC: 0.9871
- MAP: 0.4201
- Runtime: 1.80 seconds

**Winner:** Highest F1-score, precision, recall, ROC-AUC, and MAP across all heuristic methods.

**Pros:**
- Best overall accuracy
- Strong hub suppression
- Consistent performance

**Cons:**
- ~30% slower than CN
- Still limited to 2-hop patterns

**Best for:** Production friend recommendation, when accuracy matters most

#### 3.4.7 Comparative Analysis

**Performance Summary**

**Table: Algorithm Performance (Average across 10 graphs)**

| Algorithm | Precision | Recall | F1 | ROC-AUC | MAP | Runtime (s) | Speed vs CN |
|-----------|-----------|--------|-----|---------|-----|-------------|-------------|
| **Resource Allocation** 1st | **0.3563** | **0.0291** | **0.0425** | **0.9871** | **0.4201** | 1.80 | 0.79x |
| **Adamic-Adar** 2nd | 0.3403 | 0.0289 | 0.0418 | 0.9858 | 0.4080 | 1.86 | 0.76x |
| **Common Neighbors** 3rd | 0.3316 | 0.0281 | 0.0406 | 0.9838 | 0.3951 | 1.41 | 1.0x |
| **Jaccard Coefficient** | 0.3209 | 0.0261 | 0.0382 | 0.9724 | 0.3620 | 2.06 | 0.68x |
| **Preferential Attachment** | 0.0510 | 0.0146 | 0.0172 | 0.8374 | 0.1346 | **0.65** | **2.16x** |

### 5.2 Key Findings

**Accuracy Rankings:**
1. Resource Allocation wins on all accuracy metrics
2. Adamic-Adar is close second
3. Common Neighbors is solid third
4. Jaccard underperforms in sparse graphs
5. Preferential Attachment sacrifices accuracy for speed

**Speed Rankings:**
1. Preferential Attachment: 0.65s (fastest)
2. Common Neighbors: 1.41s (baseline)
3. Resource Allocation: 1.80s
4. Adamic-Adar: 1.86s
5. Jaccard Coefficient: 2.06s (slowest)

**Speed-Accuracy Tradeoff:**
- **Best Accuracy:** RA gives +4.7% F1 over CN at 28% slowdown
- **Best Speed:** PA gives 2.16x speedup but -58% F1
- **Best Balance:** CN offers 95.5% of RA's accuracy at 78% of runtime

---

## 6. Algorithm Selection Guide

**Choose based on your priorities:**

| Priority | Algorithm | Reason |
|----------|-----------|--------|
| **Maximum Accuracy** | Resource Allocation | Best F1, precision, recall, MAP |
| **Speed-Accuracy Balance** | Common Neighbors | 95.5% of RA's accuracy, 78% of runtime |
| **Maximum Speed** | Preferential Attachment | 2.16x faster, good for prefiltering |
| **Reduce Hub Bias** | Resource Allocation or AA | Strong degree penalties |
| **Interpretability** | Common Neighbors | "You have X mutual friends" |
| **Dense Networks** | Jaccard Coefficient | Normalization helps when avg degree >100 |

---

## 7. How to Run

### Setup

Make sure you are in the project root directory.

### Execute Analysis

```bash
# Run complete analysis
python analysis.py
```

**Expected Output:**

**CSV Files:**
- `3.4frs/results/comprehensive_analysis.csv` - All heuristic algorithm results

**Visualizations:**
- `3.4frs/results/scalability_analysis.png` - Runtime vs graph size comparison
- `3.4frs/results/performance_metrics_vs_size.png` - Accuracy metrics across graphs
- `3.4frs/results/theoretical_vs_actual_complexity.png` - Complexity validation

### Customization

Edit `analysis.py` to modify:
- Sample size: Line ~580, change `sample_size=50`
- Top-k recommendations: Line ~595, change `top_k=10`
- Graph subset: Line ~532, change `range(1, 11)`

---

## 8. References

1. **Adamic, L. A., & Adar, E.** (2003). Friends and neighbors on the Web. *Social Networks*, 25(3), 211–230.

2. **Zhou, T., Lü, L., & Zhang, Y.-C.** (2009). Predicting missing links via local information. *The European Physical Journal B*, 71(4), 623–630.

3. **Barabási, A.-L., & Albert, R.** (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509–512.

4. **Liben-Nowell, D., & Kleinberg, J.** (2007). The link-prediction problem for social networks. *JASIST*, 58(7), 1019–1031.

5. **McAuley, J., & Leskovec, J.** (2012). Learning to discover social circles in ego networks. *NIPS*.

6. **Newman, M. E. J.** (2010). *Networks: An Introduction*. Oxford University Press.

7. **Lü, L., & Zhou, T.** (2011). Link prediction in complex networks: A survey. *Physica A*, 390(6), 1150–1170.

---

## Conclusion

This analysis evaluated five classical heuristic algorithms for link prediction on real Facebook network data. **Resource Allocation emerged as the winner** with F1=0.0425, followed closely by Adamic-Adar (F1=0.0418), though Common Neighbors offers excellent speed-accuracy balance at F1=0.0406 with comparable runtime.

**Key Findings:**

1. **Heuristic Methods Excel:** Simple neighborhood-based methods (CN, AA, RA) provide excellent accuracy with 1-2 second runtimes.

2. **Optimal Configurations:**
   - **Best Overall:** Resource Allocation (F1=0.0425, runtime=1.80s)
   - **Best Speed-Accuracy:** Common Neighbors (F1=0.0406, runtime=1.41s)
   - **Fastest:** Preferential Attachment (runtime=0.65s, but F1=0.0172)

**Production Use Cases:**

| Use Case | Algorithm | Rationale |
|----------|-----------|-----------|
| **Real-time API (<2s)** | Common Neighbors or RA | Fast, accurate, interpretable |
| **Maximum Accuracy** | Resource Allocation | Best F1, precision, recall, MAP |
| **Massive Scale** | Preferential Attachment | 2x faster, good for prefiltering |

The strong correlation between theoretical and empirical complexity validates our implementation and confirms these algorithms scale predictably for practical deployment.


---

## Experimental Setup

### Hardware and Software Environment

**Hardware:**
- Processor: [Typical modern multi-core processor]
- RAM: 8-16 GB
- Storage: SSD for fast I/O operations

**Software:**
- Operating System: Windows/Linux/macOS
- Python Version: 3.8+
- Key Libraries:
  - NetworkX 2.x (graph data structures)
  - NumPy 1.x (numerical operations)
  - Matplotlib 3.x (visualization)
  - pandas 1.x (data analysis)
  - python-louvain (community detection)
  - leidenalg + igraph (Leiden algorithm)
  - node2vec + gensim (embeddings)
  - scikit-learn (evaluation metrics)

### Dataset Description

**Source:** Stanford Network Analysis Platform (SNAP) - Facebook Social Circles

**Citation:** Leskovec, J., & Krevl, A. (2014). *SNAP Datasets: Stanford Large Network Dataset Collection*. http://snap.stanford.edu/data

**Dataset URL:** https://snap.stanford.edu/data/ego-Facebook.html

**Description:** The dataset consists of 10 ego networks from Facebook. Each ego network includes:
- A central user (ego)
- All friends of the ego
- Friendship connections among those friends
- Node features (anonymized)
- Social circles (ground truth communities)

**Dataset Statistics:**

| Metric | Value | Description |
|--------|-------|-------------|
| Total Users | 4,039 | Unique nodes across all ego networks |
| Total Friendships | 88,234 | Undirected edges |
| Ego Networks | 10 | Number of individual ego networks |
| Average Degree | 43.69 | Mean connections per user |
| Density | 0.011 | Ratio of actual to possible edges |
| Clustering Coefficient | 0.57-0.61 | Triangle density (high) |
| Average Path Length | 1.95-3.74 | Mean shortest path distance |
| Diameter | 2-8 | Maximum shortest path |
| Connected Components | 1 (typically) | Giant component dominates |

**Individual Ego Network Files:**
- 0.edges, 107.edges, 348.edges, 414.edges, 686.edges
- 698.edges, 1684.edges, 1912.edges, 3437.edges, 3980.edges

**Data Format:**
- Edge list format: `node1 node2` (space-separated)
- Undirected graph (edges listed once)
- Node IDs: Non-negative integers

### Experimental Methodology

**General Approach:**

1. **Data Loading:**
   - Load ego network edge lists
   - Construct NetworkX Graph objects
   - Verify graph properties (connectivity, node count)

2. **Algorithm Execution:**
   - Run each algorithm on all 10 ego networks
   - Record wall-clock time using Python `time.time()`
   - Measure memory usage where applicable
   - Repeat runs for statistical stability (where noted)

3. **Metrics Collection:**
   - Performance: Runtime, memory usage
   - Quality: Modularity (communities), F1-score (link prediction)
   - Complexity: Theoretical vs. empirical comparison

4. **Visualization:**
   - Generate plots using Matplotlib
   - Save results to CSV files
   - Create comparative visualizations

**Specific Methodologies by Section:**

*Section 3.1 (Traversal):*
- Test each algorithm on progressively merged networks (1-10 files)
- Measure: Runtime, start-node invariance, edge-order invariance
- Validation: Compare against theoretical O(V+E)

*Section 3.2 (Centrality):*
- Compute all four centralities on full merged network
- Scalability testing: Synthetic graphs of varying sizes
- Validation: Compare against theoretical complexities
- Analysis: Correlation matrices, top-k rankings

*Section 3.3 (Community Detection):*
- Run Girvan-Newman, Louvain, Leiden on same graph
- Measure: Modularity scores, community sizes, runtime
- Visualization: Community structure plots
- Complexity validation: Theoretical vs. actual time

*Section 3.4 (Link Prediction):*
- Train-test split: 80% edges for training, 20% for testing
- Negative sampling: Equal number of non-edges as positive samples
- Metrics: Precision, Recall, F1-score, ROC-AUC, MAP
- Top-k evaluation: Rank predictions, evaluate top 10
- Node2Vec: Hyperparameter grid search (p, q, walks, length)

### Evaluation Metrics

**Performance Metrics:**
- **Runtime:** Wall-clock time in seconds
- **Memory:** Peak memory usage (where measured)
- **Scalability:** Runtime vs. graph size plots

**Quality Metrics:**

*Link Prediction:*
- **Precision@k:** TP / (TP + FP) for top-k predictions
- **Recall@k:** TP / (TP + FN) for top-k predictions
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under receiver operating characteristic curve
- **MAP:** Mean Average Precision across rankings

*Community Detection:*
- **Modularity (Q):** Density of within-community vs. between-community edges
  - Formula: $Q = \frac{1}{2m} \sum_{i,j} \left( A_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j)$
  - Range: [-0.5, 1.0]; Q > 0.3 indicates significant structure

*Centrality:*
- **Correlation:** Pearson correlation between different centrality measures
- **Ranking stability:** Overlap in top-k nodes across measures

### Reproducibility

All code and data are organized to ensure reproducibility:

**To reproduce traversal analysis (3.1.1):**
```bash
cd 3.1.1cca
python analysis.py
```

**To reproduce centrality analysis (3.1.2):**
```bash
cd 3.1.2ca
python analysis.py
python scalability_analysis.py
```

**To reproduce community detection (3.2):**
```bash
cd 3.2cd
python analysis.py
```

**To reproduce friend recommendation (3.4):**
```bash
cd 3.4frs
python analysis.py
python nv_analysis.py  # For Node2Vec
```

Results are saved in respective `results/` subdirectories with timestamps and descriptive filenames.

#### 3.4.8 Node2Vec: Embedding-Based Method (BONUS)

**Overview**

This subsection implements and evaluates Node2Vec, an embedding-based method for friend recommendation in social networks. Node2Vec learns low-dimensional vector representations of nodes through biased random walks, then uses cosine similarity for link prediction. We test this algorithm on the Facebook Ego Network dataset to compare with classical heuristic methods.

**Key Finding:** Node2Vec embeddings provide competitive results but require 150x more training time compared to heuristic methods. Best configuration (p=0.7, q=2.0, walks=20, length=120) achieves F1=0.1598, while fastest viable config (p=1.0, q=1.0, walks=5, length=40) achieves F1=0.1217 at 2.92s runtime.

**Problem Statement**

**Problem Statement**

Given a social network graph, predict which non-connected users are likely to form connections using embedding-based methods.

**Algorithm Description**

Learn low-dimensional vector representations of nodes through biased random walks, then use cosine similarity for link prediction.

**Implementation**

```python
from node2vec import Node2Vec

def node2vec_predict(G, p=1.0, q=1.0, num_walks=10, walk_length=80):
    # Train embedding model
    node2vec = Node2Vec(G, dimensions=128, walk_length=walk_length,
                        num_walks=num_walks, p=p, q=q, workers=4)
    model = node2vec.fit(window=10, min_count=1)
    
    # Compute similarities for predictions
    # (cosine similarity between embeddings)
```

**Hyperparameters**

- **p (return parameter):** Controls likelihood of returning to previous node
- **q (in-out parameter):** Controls BFS vs DFS-like exploration
- **num_walks:** Number of random walks per node
- **walk_length:** Steps in each walk

**Walk Strategies**

- **DeepWalk (p=1, q=1):** Unbiased random walk
- **BFS-like (p<1, q>1):** Local neighborhood exploration
- **DFS-like (p>1, q<1):** Explores distant nodes
- **Balanced (p≈0.7, q≈0.7):** Balanced local/global exploration

**How Node2Vec Works**

**Random Walk with Bias:**

Starting at node $t$, previously at $s$, picking next node $x$:

$$P(x \mid s, t) = \begin{cases}
\frac{1}{p} & \text{if } x = s \text{ (return to previous)} \\
1 & \text{if } x \text{ is neighbor of both } s \text{ and } t \text{ (stay local)} \\
\frac{1}{q} & \text{if } x \text{ is new territory}
\end{cases}$$

**Parameters:**
- **p=1, q=1:** Uniform random walk (DeepWalk)
- **p=0.5, q=2.0:** BFS-like - explores neighborhoods (community detection)
- **p=2.0, q=0.5:** DFS-like - ventures far (structural equivalence)

**Complexity:**
- **Training:** $O(k \cdot l \cdot n \cdot \log n)$ where k=num_walks, l=walk_length
- **Inference:** $O(n \log n)$ for similarity computations
- **Total:** Much slower than heuristics but captures global structure

**Experimental Results**

*Performance Metrics (Default Configuration)*

*Source: `3.4frs/results/comprehensive_analysis.csv` - Graph 10 (largest network)*

- Precision: 0.2460
- Recall: 0.0069
- F1-Score: 0.0134
- ROC-AUC: 0.9897
- MAP: 0.3177
- Training Time: ~200 seconds per graph
- Total Runtime: 211.45 seconds (150x slower than CN)

*Hyperparameter Analysis*

![Hyperparameter P/Q](3.4frs/results/nv_hyperparameter_pq.png)

**Figure 1:** Impact of p and q parameters on F1-score and runtime. Lower p/q values (more local exploration) generally yield better F1-scores.

![Hyperparameter Walks](3.4frs/results/nv_hyperparameter_walks.png)

**Figure 2:** Impact of number of walks and walk length on performance. More walks and longer paths improve accuracy but increase training time significantly.

**Key Findings:**

*Source: `3.4frs/results/nv_hyperparameter_exploration.csv`*

- **Best Configuration:** p=0.7, q=2.0, walks=20, length=120
  - F1: 0.1598 (best among Node2Vec configs)
  - Runtime: 23.28s (still 16x slower than RA)
  - Trade-off: 4x better F1 than fastest config, but 8x slower
- **Fastest Viable:** p=1.0, q=1.0, walks=5, length=40
  - F1: 0.1217
  - Runtime: 2.92s (2x slower than RA, but competitive)
  - Trade-off: Acceptable for real-time scenarios

*Comparison with Heuristics*

![Node2Vec vs Heuristics](3.4frs/results/nv_vs_heuristics.png)

**Figure 3:** Node2Vec variants compared to traditional heuristics on the largest graph (n=4039).

**Table: Node2Vec vs Heuristic Methods (Graph 10)**

*Source: `3.4frs/results/nv_vs_heuristics.csv`*

| Method | Precision | F1 | ROC-AUC | Runtime (s) | Speed vs CN |
|--------|-----------|-----|---------|-------------|-------------|
| Resource Allocation | 0.3257 | 0.0173 | 0.9998 | 1.70 | 0.91x |
| Adamic-Adar | 0.3064 | 0.0163 | 0.9997 | 1.71 | 0.90x |
| Common Neighbors | 0.3043 | 0.0162 | 0.9992 | 1.55 | 1.0x |
| **Node2Vec (Balanced)** | 0.2460 | 0.0134 | 0.9897 | **211.45** | **0.007x** |
| Node2Vec (DFS-like) | 0.2560 | 0.0141 | 0.9858 | 212.14 | 0.007x |
| Node2Vec (DeepWalk) | 0.2414 | 0.0132 | 0.9871 | 146.84 | 0.011x |

*Scalability Analysis*

![Node2Vec Scalability](3.4frs/results/nv_scalability.png)

**Figure 4:** Node2Vec training time scales super-linearly with graph size, making it impractical for graphs with >5000 nodes without distributed computing.

*Source: `3.4frs/results/nv_scalability.csv`*

**Pros and Cons**

*Pros*

- Captures global network structure (beyond 2-hop)
- Pre-trained embeddings can be reused for multiple tasks
- Excellent ROC-AUC scores (near 0.99)
- Flexible: p/q parameters tune exploration strategy

*Cons*

- 150x slower than heuristic methods
- Lower precision and F1-score
- Requires hyperparameter tuning
- Memory intensive for large graphs
- Training time grows with graph size

**Best Use Cases**

*Best for:*

- Multi-task learning (embeddings used for multiple predictions)
- When global structure matters (community-based recommendations)
- Offline batch processing with time budget
- When interpretability is not critical

*Not Recommended for:*

- Real-time recommendations (<1 second latency)
- When only link prediction is needed
- Resource-constrained environments





---

## Implementation Details

### Technology Stack

**Programming Language:**
- Python as base programming language

**Core Libraries:**
- **NetworkX**: Graph data structures and basic algorithms
- **NumPy**: Numerical operations and array computations
- **Matplotlib**: Data visualization and plotting
- **pandas**: Data manipulation and CSV operations
- **scikit-learn**: Machine learning metrics (ROC-AUC, precision, recall)

**Specialized Libraries:**
- **node2vec + gensim**: Node embedding generation (BONUS section)

### Graph Representation

**Data Structure Choice:** NetworkX Graph objects

**Rationale:**
- Rich API for graph operations
- Efficient handling of graphs
- Good integration with other Python libraries like Matplotlib and NumPy

### Key Design Decisions

**Traversal Algorithms (BFS/DFS):**
- BFS: Python `collections.deque` for FIFO queue operations (O(1) append/pop)
- DFS: Recursive implementation with implicit call stack
- Visited tracking: Python `set` for O(1) membership testing
- Alternative: Explicit stack for DFS when recursion depth limits are a concern

**Union-Find:**
- Parent array: Python list indexed by node ID
- Rank/Size array: Separate list for union heuristic values
- Path compression: Applied during find() operation
- Union by rank/size: Attaches smaller tree under larger tree

**Centrality Measures:**
- Degree: Direct computation from adjacency list
- Closeness: BFS-based shortest path computation with Wasserman-Faust normalization
- Betweenness: Brandes' algorithm (2001) with O(VE) complexity
- Eigenvector: Power iteration with NumPy for vectorized operations

**Community Detection:**
- Girvan-Newman: NetworkX betweenness with iterative edge removal
- Louvain: Greedy modularity optimization with multi-level aggregation
- Leiden: Improved Louvain with refinement phase for guaranteed connectivity

**Link Prediction:**
- Heuristics: Set operations for neighbor intersection/union
- Node2Vec (BONUS): Gensim Word2Vec for skip-gram model training
- Evaluation: Train-test split (80/20), negative sampling for balanced evaluation

### Implementation Challenges and Solutions

**Challenge 1: Memory Management for Large Graphs**
- *Problem*: Full pairwise distance matrices for closeness centrality exceed memory
- *Solution*: Compute distances on-the-fly using BFS, store only necessary intermediate values

**Challenge 2: Union-Find Edge Order Sensitivity**
- *Problem*: Processing order affects tree structure and cache performance (14-15% variance)
- *Solution*: Document sensitivity, maintain original edge ordering where possible

**Challenge 3: Node2Vec Training Time**
- *Problem*: Embedding generation requires 150x longer than heuristic methods
- *Solution*: Multi-threading for walk generation, hyperparameter optimization, embedding caching

**Challenge 4: Girvan-Newman Scalability**
- *Problem*: O(m²n) complexity makes algorithm impractical beyond 1000 nodes
- *Solution*: Test on smaller subgraphs, focus on Louvain/Leiden for production use

**Challenge 5: Numerical Stability**
- *Problem*: Eigenvector centrality power iteration can diverge
- *Solution*: L2 normalization at each iteration, convergence threshold ε=10⁻⁶

### Code Organization

```
project_root/
├── 3.1.1cca/          # Graph Traversal & Connectivity
│   ├── bfs.py         # Breadth-First Search
│   ├── dfs.py         # Depth-First Search
│   ├── ufa_by_rank.py # Union-Find by Rank
│   ├── ufa_by_size.py # Union-Find by Size
│   └── analysis.py    # Comprehensive analysis
├── 3.1.2ca/           # Centrality Measures
│   ├── dc.py          # Degree Centrality
│   ├── cc.py          # Closeness Centrality
│   ├── bc.py          # Betweenness Centrality
│   ├── ec.py          # Eigenvector Centrality
│   ├── analysis.py    # Comparative analysis
│   └── scalability_analysis.py
├── 3.2cd/             # Community Detection
│   ├── gn.py          # Girvan-Newman
│   ├── la.py          # Leiden Algorithm
│   ├── lm.py          # Louvain Method
│   └── analysis.py    # Comparative analysis
├── 3.4frs/            # Friend Recommendation Systems
│   ├── aa.py          # Adamic-Adar
│   ├── cm.py          # Common Neighbors
│   ├── jc.py          # Jaccard Coefficient
│   ├── pa.py          # Preferential Attachment
│   ├── ra.py          # Resource Allocation
│   ├── nv.py          # Node2Vec (BONUS)
│   ├── analysis.py    # Heuristic analysis
│   └── nv_analysis.py # Node2Vec analysis (BONUS)
├── dataset/           # SNAP Facebook ego networks
│   ├── 0.edges, 107.edges, 348.edges, etc.
│   └── [10 ego network files]
├── graph.py           # Shared graph utilities
├── visualize.py       # Common visualization functions
└── requirements.txt   # Python dependencies
```

---

## Experimental Setup

### Dataset

**Source:** Stanford Network Analysis Platform (SNAP) - Facebook Social Circles

**Reference:** Leskovec, J., & Krevl, A. (2014). *SNAP Datasets: Stanford Large Network Dataset Collection*. http://snap.stanford.edu/data

**URL:** https://snap.stanford.edu/data/ego-Facebook.html

**Description:** 10 ego networks from Facebook, each containing:
- One central user (ego)
- All friends of the ego
- Friendship edges among those friends
- Anonymized node features
- Ground truth social circles

**Dataset Statistics:**

| Metric | Value |
|--------|-------|
| Total Nodes | 4,039 |
| Total Edges | 88,234 |
| Ego Networks | 10 |
| Average Degree | 43.69 |
| Density | 0.011 |
| Clustering Coefficient | 0.57-0.61 |
| Average Path Length | 1.95-3.74 |
| Diameter | 2-8 |
| Components | 1 |

### How to Run

**Complete Analysis Pipeline:**

```bash
cd /path/to/aad
pip install -r requirements.txt

cd 3.1.1cca
python analysis.py

cd ../3.1.2ca
python analysis.py
python scalability_analysis.py

cd ../3.2cd
python analysis.py

cd ../3.4frs
python analysis.py
python nv_analysis.py 
```

**Individual Algorithm Execution:**

```bash
python 3.1.1cca/bfs.py      # BFS only
python 3.1.2ca/bc.py        # Betweenness Centrality only
python 3.4frs/ra.py         # Resource Allocation only
```

### Reproducibility

All experiments are fully reproducible with:
- Fixed random seeds where applicable
- Deterministic algorithms (BFS, DFS, Girvan-Newman, Louvain, Leiden)
- Documented dataset version and preprocessing
- Complete source code and configuration files
- Results saved with timestamps and parameters

---

## Conclusion

This comprehensive study analyzed twelve fundamental graph algorithms across four categories applied to real-world Facebook social networks, validating theoretical complexity bounds and comparing practical performance characteristics.

### Key Findings

**1. Graph Traversal and Connectivity**
- BFS and DFS achieve linear O(V+E) scaling with minimal performance variance
- Union-Find variants deliver near-constant amortized time with O(E·α(V)) complexity
- DFS demonstrates superior order invariance (3.9% variance vs 14.5% for Union-Find)
- All algorithms scale efficiently to tens of thousands of nodes

**2. Centrality Measures**
- Degree Centrality provides fastest computation (O(V+E)) but limited insight
- Eigenvector Centrality offers quality-based importance with O(kE) complexity
- Betweenness Centrality reveals bridges but requires O(VE) computation
- Closeness Centrality optimizes broadcasting but scales poorly (O(V²-V³))
- Strong positive correlations (0.4-0.9) indicate reinforcing importance aspects

**3. Community Detection**
- Leiden algorithm achieves highest quality (modularity) with guaranteed connectivity
- Louvain provides excellent speed-accuracy balance at O(N log N)
- Girvan-Newman offers interpretability but O(m²n) complexity limits scalability
- Modularity scores (0.4-0.6) confirm significant community structure in social networks

**4. Friend Recommendation Systems**
- Resource Allocation wins on accuracy (F1=0.0425) with moderate speed
- Common Neighbors offers best speed-accuracy tradeoff (F1=0.0406, 1.41s)
- Preferential Attachment fastest (0.65s) but lowest accuracy (F1=0.0172)
- Node2Vec (BONUS) provides competitive embeddings but requires 150x training time
- Simple heuristics remain optimal for production friend recommendation

### Practical Recommendations

**Algorithm Selection Guidelines:**

| Requirement | Recommended Algorithm | Rationale |
|-------------|----------------------|-----------|
| **Shortest paths** | BFS | O(V+E), guaranteed optimal paths |
| **Structural analysis** | DFS | O(V+E), superior order invariance |
| **Dynamic connectivity** | Union-Find | O(α(V)) amortized, incremental updates |
| **Influence identification** | Eigenvector Centrality | Quality-aware, efficient O(kE) |
| **Bridge detection** | Betweenness Centrality | Reveals bottlenecks |
| **Community detection** | Leiden | Best quality with O(N log N) speed |
| **Friend recommendation** | Resource Allocation | Best F1-score, production-ready |
| **Real-time API (<2s)** | Common Neighbors | Fast, interpretable, reliable |
| **Multi-task embeddings** | Node2Vec (BONUS) | Reusable representations |

**Scalability Insights:**
- Linear algorithms (BFS, DFS, Union-Find, Degree) scale to millions of nodes
- Quadratic algorithms (Closeness, Betweenness) limited to ~10K nodes
- Community detection (Louvain, Leiden) handle millions efficiently
- Simple heuristics outperform complex methods for single-task prediction

**Network Properties:**
- Small-world characteristics (high clustering, short paths) enable efficient algorithms
- Sparse structure (density 0.011) favors adjacency list representations
- Single giant component (98-100% coverage) simplifies connectivity analysis
- Power-law degree distribution requires hub-aware algorithm design

### Limitations

1. **Dataset Scope**: Analysis limited to 10 ego networks (~4K nodes); larger networks may exhibit different scaling behavior
2. **Temporal Dynamics**: Static snapshot analysis; does not capture network evolution
3. **Feature Utilization**: Algorithms use only graph topology; node/edge attributes ignored
4. **Ground Truth**: Limited validation against known community labels or friendship outcomes
5. **Hyperparameter Sensitivity**: Some algorithms (Node2Vec, Leiden) require careful tuning

### Future Work

**Algorithmic Extensions:**
- Implement approximate algorithms for large-scale betweenness (Brandes sampling)
- Explore overlapping community detection (DEMON, SLPA)
- Integrate deep learning methods (GraphSAGE, GCN) for link prediction
- Develop incremental algorithms for dynamic graph updates

**Application Domains:**
- Extend analysis to weighted and directed networks
- Apply to temporal network snapshots for evolution analysis
- Validate on other domains (citation networks, biological networks)
- Implement real-time streaming algorithms for live social feeds

**Performance Optimization:**
- GPU acceleration for matrix operations in centrality computation
- Distributed computing for Node2Vec walk generation
- Memory-efficient data structures for billion-edge graphs
- Approximate methods with quality-speed tradeoffs

This study demonstrates that careful algorithm selection based on problem requirements, network characteristics, and performance constraints is essential for effective social network analysis. Simple, well-designed algorithms often outperform complex methods when properly applied to real-world problems.

---

## Bonus Disclosure

The following components should be evaluated for bonus credit:

### Bonus Section: Node2Vec Embedding-Based Link Prediction

**Location:** Section 3.4.8 - Node2Vec: Embedding-Based Friend Recommendation Analysis (BONUS)

**Implementation:**
- Complete Node2Vec embedding generation with biased random walks
- Hyperparameter exploration (p, q, num_walks, walk_length)
- Comparison with classical heuristic methods
- Scalability analysis and performance benchmarking

**Deliverables:**
- `3.4frs/nv.py` - Node2Vec implementation
- `3.4frs/nv_analysis.py` - Comprehensive Node2Vec analysis
- `3.4frs/results/nv_vs_heuristics.csv` - Performance comparison data
- `3.4frs/results/nv_hyperparameter_exploration.csv` - Tuning results
- `3.4frs/results/nv_scalability.csv` - Scalability data
- Visualization plots (4 figures)

**Key Results:**
- Best configuration: p=0.7, q=2.0, walks=20, length=120 (F1=0.1598)
- Fastest viable: p=1.0, q=1.0, walks=5, length=40 (F1=0.1217 at 2.92s)
- Trade-off analysis: 150x slower than heuristics but captures global structure
- ROC-AUC ~0.99 demonstrates strong ranking capability

**Bonus Justification:**
Node2Vec represents an advanced embedding-based approach beyond the required heuristic methods. It demonstrates:
1. Understanding of modern graph representation learning
2. Implementation of complex random walk strategies
3. Extensive hyperparameter optimization
4. Rigorous comparison with baseline methods
5. Critical analysis of speed-accuracy tradeoffs

This bonus work adds significant value by exploring the frontier between classical graph algorithms and modern machine learning approaches for social network analysis.

---

## References

### Academic Papers

1. **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.** (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
   - Chapter 22: Elementary Graph Algorithms (BFS, DFS)
   - Chapter 21: Data Structures for Disjoint Sets (Union-Find)

2. **Tarjan, R. E.** (1975). "Efficiency of a Good But Not Linear Set Union Algorithm." *Journal of the ACM*, 22(2), 215-225. https://doi.org/10.1145/321879.321884

3. **Freeman, L. C.** (1978). "Centrality in social networks conceptual clarification." *Social Networks*, 1(3), 215-239. https://doi.org/10.1016/0378-8733(78)90021-7

4. **Brandes, U.** (2001). "A faster algorithm for betweenness centrality." *Journal of Mathematical Sociology*, 25(2), 163-177. https://doi.org/10.1080/0022250X.2001.9990249

5. **Bonacich, P.** (1972). "Factoring and weighting approaches to status scores and clique identification." *Journal of Mathematical Sociology*, 2(1), 113-120. https://doi.org/10.1080/0022250X.1972.9989806

6. **Bavelas, A.** (1950). "Communication patterns in task-oriented groups." *Journal of the Acoustical Society of America*, 22(6), 725-730. https://doi.org/10.1121/1.1906679

7. **Girvan, M., & Newman, M. E. J.** (2002). "Community structure in social and biological networks." *Proceedings of the National Academy of Sciences*, 99(12), 7821-7826. https://doi.org/10.1073/pnas.122653799

8. **Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E.** (2008). "Fast unfolding of communities in large networks." *Journal of Statistical Mechanics: Theory and Experiment*, 2008(10), P10008. https://doi.org/10.1088/1742-5468/2008/10/P10008

9. **Traag, V. A., Waltman, L., & van Eck, N. J.** (2019). "From Louvain to Leiden: guaranteeing well-connected communities." *Scientific Reports*, 9(1), 5233. https://doi.org/10.1038/s41598-019-41695-z

10. **Adamic, L. A., & Adar, E.** (2003). "Friends and neighbors on the Web." *Social Networks*, 25(3), 211-230. https://doi.org/10.1016/S0378-8733(03)00009-1

11. **Zhou, T., Lü, L., & Zhang, Y.-C.** (2009). "Predicting missing links via local information." *The European Physical Journal B*, 71(4), 623-630. https://doi.org/10.1140/epjb/e2009-00335-8

12. **Liben-Nowell, D., & Kleinberg, J.** (2007). "The link-prediction problem for social networks." *Journal of the American Society for Information Science and Technology*, 58(7), 1019-1031. https://doi.org/10.1002/asi.20591

13. **Barabási, A.-L., & Albert, R.** (1999). "Emergence of scaling in random networks." *Science*, 286(5439), 509-512. https://doi.org/10.1126/science.286.5439.509

14. **Watts, D. J., & Strogatz, S. H.** (1998). "Collective dynamics of 'small-world' networks." *Nature*, 393(6684), 440-442. https://doi.org/10.1038/30918

15. **Newman, M. E. J.** (2006). "Modularity and community structure in networks." *Proceedings of the National Academy of Sciences*, 103(23), 8577-8582. https://doi.org/10.1073/pnas.0601602103

### BONUS: Node2Vec References

16. **Grover, A., & Leskovec, J.** (2016). "node2vec: Scalable feature learning for networks." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 855-864. https://doi.org/10.1145/2939672.2939754

17. **Perozzi, B., Al-Rfou, R., & Skiena, S.** (2014). "DeepWalk: Online learning of social representations." *Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 701-710. https://doi.org/10.1145/2623330.2623732

18. **Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J.** (2013). "Distributed representations of words and phrases and their compositionality." *Advances in Neural Information Processing Systems*, 26, 3111-3119.

### Dataset

19. **Leskovec, J., & Krevl, A.** (2014). *SNAP Datasets: Stanford Large Network Dataset Collection*. http://snap.stanford.edu/data

20. **McAuley, J., & Leskovec, J.** (2012). "Learning to discover social circles in ego networks." *Advances in Neural Information Processing Systems*, 25, 539-547.

### Online Resources

21. **NetworkX Documentation**. (2024). https://networkx.org/documentation/stable/

22. **Wikipedia: Graph theory algorithms**. https://en.wikipedia.org/wiki/Category:Graph_algorithms

23. **Stanford SNAP: Facebook Social Circles Dataset**. https://snap.stanford.edu/data/ego-Facebook.html

---

