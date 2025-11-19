Connected Components Analysis (3.1.1cca)

This module implements Connected Components Analysis using BFS, DFS, and two Union-Find variants (Rank and Size) on the Facebook Social Circles dataset. All metrics and algorithms are implemented from scratch, adhering to the "no external libraries for logic" rule.

1. How to Run

A. Full Analysis Suite (Recommended)

Run the main analysis script to generate all 10+ metric plots and reports in one execution.

python analysis.py


Pipeline:

Analysis A (Invariance): Validates that BFS/DFS performance is independent of the starting node (100 randomized runs on the full graph).

Analysis B (Complexity): Measures scalability by incrementally loading the dataset (1 to 10 files) and proving $O(V+E)$ complexity for all 4 algorithms (BFS, DFS, UFA-Rank, UFA-Size).

Analysis C (Stability): Tests if the order of neighbor/edge processing affects runtime (using pre-shuffled structures).

Analysis D (Connectivity): Computes 10 distinct network metrics (Density, Clustering, Diameter, etc.) using custom-built algorithms to track network evolution.

B. Interactive Mode

Run any algorithm individually:

python bfs.py
python dfs.py
python ufa_by_rank.py
python ufa_by_size.py


2. Outputs (plots/ and analysis/)

Metric Trends: Metric_density.png, Metric_clustering.png, Metric_diameter.png, etc.

Complexity: B_Complexity_Grid.png for all algorithms.

Reports: D_Connectivity_Analysis.md (Full data log), `B_Complexity