Network Connectivity Metrics Analysis

This document provides a detailed explanation of the network connectivity metrics used in the analysis.py script. These metrics are crucial for understanding the structural properties and evolution of the Facebook Social Circles network as more data is processed.

1. Global Growth Metrics

These metrics track the fundamental scale and density of the network.

Nodes ($V$) and Edges ($E$)

Definition:

Nodes ($V$): The total count of unique users in the graph.

Edges ($E$): The total count of friendships (undirected connections) between users.

Importance: These are the raw measures of network size. The relationship between $V$ and $E$ (e.g., is $E$ linear with $V$, or quadratic?) determines the computational complexity of algorithms ($O(V+E)$) and the sparsity of the graph.

Expected Trend: Both should increase linearly as more files (ego networks) are loaded. Edges typically grow faster than nodes in social networks.

Graph Density

Definition: The ratio of existing edges to the maximum possible number of edges in a graph with $V$ nodes.


$$D = \frac{2E}{V(V-1)}$$

Importance: Measures how "saturated" the network is with connections.

$D=1$: A complete graph (everyone is friends with everyone).

$D \approx 0$: A sparse graph.

Expected Trend: Decreasing. Real-world social networks are sparse. As the network grows, the number of possible connections ($V^2$) grows much faster than the actual friendships ($E$), causing density to drop.

Average Degree

Definition: The average number of friends a user has.


$$\bar{k} = \frac{2E}{V}$$

Importance: Indicates the typical social reach or connectivity of an individual. In this dataset, a high average degree suggests dense local communities (ego networks).

Expected Trend: relatively stable or slowly increasing. While the network grows, individual users typically maintain a manageable number of friendships (Dunbar's number concept), though merging ego networks can slightly increase this average.

2. Clustering & Cohesion Metrics

These metrics measure how tightly knit the community is.

Average Clustering Coefficient

Definition: A measure of the degree to which nodes tend to cluster together. It is the probability that two friends of a user are also friends with each other (forming a triangle).

Importance: This is the hallmark of social networks (as opposed to random graphs). High clustering indicates "Triadic Closure"—if Person A knows Person B and Person C, it is likely B and C know each other.

Expected Trend: Consistently High (e.g., 0.5 - 0.7). Random graphs of this size would have a near-zero coefficient.

3. Component Analysis Metrics

These metrics describe the fragmentation of the network.

Number of Connected Components

Definition: The count of disjoint subgraphs where every node is reachable from every other node within the subgraph, but not from nodes outside it.

Importance: Measures network fragmentation.

1 Component: The network is fully connected (ideal for broadcasting).

>1 Component: The network has isolated islands.

Expected Trend: Low integer values (1, 2, or 3). It typically stays at 1. A jump to 2 indicates a new file (ego network) was loaded that is temporarily isolated from the main group. It usually drops back to 1 as subsequent files provide the "bridge" connections.

Giant Component (GC) Size & Coverage

Definition:

GC Size: The number of nodes in the largest connected component.

GC Coverage: The percentage of total nodes that belong to the GC: $(\frac{\text{GC Size}}{V}) \times 100$.

Importance: Measures the robustness of the network. A high coverage means the vast majority of users are part of the same "mainstream" social fabric.

Expected Trend: Near 100%. Social networks usually have one massive component that contains almost everyone, with only a tiny fraction of users being isolated.

4. Small-World Metrics (Path Analysis)

These metrics (calculated on the Giant Component) describe how fast information or influence can spread.

Network Diameter (Estimated)

Definition: The longest shortest path between any two nodes in the network. It represents the "width" of the graph.

Importance: A small diameter relative to the network size indicates efficient information diffusion (e.g., a rumor spreads quickly).

Expected Trend: Small integers (e.g., 8-12). It tends to grow very slowly (logarithmically) even as the network size explodes.

Average Path Length (Estimated)

Definition: The average number of steps (hops) required to get from one random user to another.

Importance: Validates the "Small World" property (or "Six Degrees of Separation").

Expected Trend: Very low (e.g., 3-5 hops). This confirms that despite having thousands of users, everyone is surprisingly close to everyone else via mutual friends.3-