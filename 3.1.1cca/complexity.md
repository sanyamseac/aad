Time and Space Complexity Analysis

This document details the theoretical complexity for the four Connected Components algorithms implemented in this project.

1. Breadth-First Search (BFS)

BFS explores the graph layer by layer using a Queue.

Time Complexity: $O(V + E)$

Explanation: In the worst case, every vertex ($V$) and every edge ($E$) will be explored exactly once. $V$ comes from dequeuing each node, and $E$ comes from iterating over adjacency lists.

Space Complexity: $O(V)$

Explanation: In the worst case (a star graph), the queue might store $O(V)$ vertices. We also maintain a visited set of size $V$.

2. Depth-First Search (DFS)

DFS explores as deep as possible along each branch using a Stack (iterative implementation).

Time Complexity: $O(V + E)$

Explanation: Similar to BFS, DFS visits every vertex and traverses every edge once.

Space Complexity: $O(V)$

Explanation: The stack height can grow to $O(V)$ in the worst case (a long line graph). The visited set also takes $O(V)$.

3. Union-Find Algorithm (Union by Rank)

Uses disjoint sets to track components. This implementation uses Path Compression and Union by Rank (attaching the shorter tree to the taller tree).

Time Complexity: $O(E \cdot \alpha(V))$

Explanation: We perform $E$ union/find operations. With Path Compression and Union by Rank, the amortized cost per operation is $\alpha(V)$.

Ackermann Function ($\alpha$): $\alpha(V)$ is the Inverse Ackermann function. It grows extremely slowly. For all practical values of $V$ (up to $10^{80}$, the number of atoms in the universe), $\alpha(V) \le 4$.

Effective Complexity: Linear, $O(E)$.

Space Complexity: $O(V)$

Explanation: Requires arrays/dictionaries for parent and rank, both size $V$.

4. Union-Find Algorithm (Union by Size)

Similar to Rank, but attaches the tree with fewer nodes to the tree with more nodes.

Time Complexity: $O(E \cdot \alpha(V))$

Explanation: Theoretically identical to Union by Rank. The optimizations guarantee the same nearly-constant amortized time per operation.

Space Complexity: $O(V)$

Explanation: Requires parent and size arrays, both size $V$.

Summary Table

Algorithm

Time Complexity

Space Complexity

Constant Factors

BFS

$O(V + E)$

$O(V)$

Moderate (Queue overhead)