## Analysis Progress & Output Map

This document maps each analysis section to its metrics, plots, parameters, and the expected outcomes. All paths are relative to this file (`3.1.1cca/progress.md`).

### A. Start-Node Invariance
- Algorithm: BFS, DFS
- Metric: Execution time (seconds) per trial
- Plot: `./plots/A_Invariance.png` (scatter with mean/variance)
- Parameters: Full dataset (10 files), random start nodes, ~100 trials
- Expectation: Flat trend across trials; O(V+E) independent of start node for a fixed connected component.

### B. Time Complexity
- Algorithms: BFS, DFS, UFA (union-find)
- Metrics: Execution time vs V, E, (V+E), and V×E
- Plots:
	- `./plots/B_BFS_Complexity_Grid.png`
	- `./plots/B_DFS_Complexity_Grid.png`
	- `./plots/B_UFA_Complexity_Grid.png`
- Parameters: Increasing dataset size from 1 → 10 files
- Expectations:
	- Time vs E: approximately linear
	- Time vs (V+E): approximately linear (E ≫ V typically)
	- Time vs V: may appear super-linear because edges dominate work
	- Time vs V×E: clearly sub-linear (not quadratic)

### C. Order Invariance
- Algorithms: BFS, DFS, UFA
- Metric: Execution time for normal vs shuffled adjacency orders
- Plot: `./plots/C_Order_Invariance.png` (paired bars)
- Parameters: Full dataset (10 files), adjacency lists shuffled once
- Expectation: Bars should be similar; algorithmic complexity is O(V+E) regardless of iteration order (implementation details may introduce small differences).

### D. Connected Components Snapshots (per k = 1…10 files)
- Plots: `./plots/D_1_Distribution.png` … `./plots/D_10_Distribution.png`
- Each report (`analysis/D_k_Analysis.md`) summarizes:
	- Total nodes (V), total edges (E)
	- Number of connected components
	- Giant component size and coverage (%)
	- Component size distribution (table and/or histogram)
- Expectation: Mostly one dominant giant component; occasional small components may appear for some k (should be explicitly listed).

---

### Notes and Follow-ups
- Ensure UFA invariance is documented alongside BFS/DFS in A.
- If multiple components exist (e.g., k=7), include a table listing all component sizes, not just the giant component.
- Keep headings/plots consistent with existing files in `./plots/` and `./analysis/`.
s