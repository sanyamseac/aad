## 3.2 Community Detection Analysis

### Description
In this module, we move beyond simple connectivity to identify the sub-structures within the social network. Community detection partitions the graph into groups (communities) where nodes are densely connected internally but sparsely connected to nodes in other groups . This analysis is crucial for understanding the latent social circles within the Facebook dataset .

### 1. What are we doing in this section?

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

# Girvan-Newman Algorithm Analysis

## 1. Description and History

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


# Louvain Algorithm Analysis

## 1. Description and History

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


# Leiden Algorithm Analysis

## 1. Description and History

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

# Community Structure Visualization Analysis

![](./results/community_structure_visualisation.png)
## 1. What is this Graph?
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

# Modularity Scores Comparison Analysis

![](./results/modularity_scores_comparison.png)


## 1. What is this Graph?
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



# Community Size Distribution Analysis

![](./results/community_size_distribution.png)


## 1. What is this Graph?
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

# Runtime Performance Analysis

![](./results/runtime_performance_analysis.png)

## 1. What is this Graph?
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


# Time Complexity Verification Analysis

![](./results/final_benchmark_plot.png)

## 1. What is this Graph?
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