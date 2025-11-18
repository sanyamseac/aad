# Connected Components Analysis (3.1.1cca)

This module implements BFS, DFS, and UFA for Connected Components Analysis (CCA).
It follows the structure defined in the project proposal deliverables.

## Code Structure

### Core Algorithms
* `bfs.py`: Iterative Breadth-First Search.
* `dfs.py`: Iterative Depth-First Search.
* `ufa.py`: Union-Find Algorithm (Disjoint Set Union).

### Experimental Analysis (`analysis.py`)
The `analysis.py` script contains four distinct functions corresponding to the project requirements:

1.  **`run_function_A()`: Start Node Invariance**
    * *Valid for:* BFS, DFS.
    * *Description:* Proves that choosing a random start node within a component does not affect the time complexity of the traversal.
    * *UFA Note:* Not applicable as UFA processes the global edge list, not a rooted traversal.

2.  **`run_function_B()`: Time Complexity Analysis**
    * *Valid for:* BFS, DFS, UFA.
    * *Description:* measures runtime across increasing graph sizes (1 to 10 files) to validate $O(V+E)$ for traversals and near-linear time for UFA.
    * *Output:* `analysis_B_complexity.png`

3.  **`run_function_C()`: Order Invariance**
    * *Valid for:* BFS, DFS, UFA.
    * *Description:* Proves that the order of visiting neighbors (BFS/DFS) or processing edges (UFA) does not affect performance. Comparing shuffled vs. non-shuffled inputs.

4.  **`run_function_D()`: CCA Deliverables (Main)**
    * *Valid for:* BFS, DFS, UFA (Comparison).
    * *Description:* Runs the full CCA on the complete dataset.
    * *Deliverables:*
        * Total number of components.
        * Giant Component analysis (size, edge count).
        * Component size distribution plot (`analysis_D_distribution.png`).

## How to Run
```bash
python analysis.py