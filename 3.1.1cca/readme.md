# Connected Components Analysis (3.1.1cca)

This folder contains the implementation and analysis for Connected Components.

## Scripts

* `analysis.py`: The main analysis script.
    * **Analysis A & C**: Runs on the **full 10-file dataset**.
    * **Analysis B**: Iterates 1-10 files to plot time complexity trends vs V, E, V+E, and V*E.
    * **Analysis D**: Iterates 1-10 files to generate component size distributions for each step.

* `bfs.py`, `dfs.py`, `ufa.py`: Interactive standalone scripts. Run them to perform a single traversal on a chosen number of files.

## Output

Running `python analysis.py` will create:

* **`plots/`**:
    * `A_Invariance.png`, `C_Order_Invariance.png`
    * `B_BFS_Complexity_Grid.png`, `B_DFS_Complexity_Grid.png`, `B_UFA_Complexity_Grid.png` (4 subplots each)
    * `D_1_Distribution.png` ... `D_10_Distribution.png`
* **`analysis/`**:
    * `A_Invariance.md`, `C_Order.md`, `B_Complexity.md`
    * `D_1_Analysis.md` ... `D_10_Analysis.md`