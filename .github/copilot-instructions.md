---
description: AI rules derived by SpecStory from the project AI interaction history
globs: *
---

## Headers

## PROJECT DOCUMENTATION & CONTEXT SYSTEM

## TECH STACK

## CODING STANDARDS

## DEBUGGING

## WORKFLOW & RELEASE RULES

### Analysis.md Rules

1.  Eliminate redundancy in `analysis.md`. The same information should not be repeated.
2.  For every graph in `analysis.md`, include:
    *   A detailed graph description, including axes, data range, and visual elements.
    *   Interpretation of the graph's data and key insights.
    *   Explanation of any unique or unexpected properties observed in the graph, including numerical details.
3.  You MUST include all relevant metrics in `analysis.md`. Do not omit any metrics.
4.  Include all information about an algorithm, including complexity analysis (theoretical and empirical), under the algorithm's dedicated header in `analysis.md`. Avoid separating complexity analysis from the algorithm's main section.
5.  Integrate all information from `metrics.md` into `analysis.md` where appropriate. Do not delete `metrics.md`.
6.  Remove the "How it works" section from `analysis.md`.
7.  Remove the "Appendix" section from `analysis.md` and integrate the formulae into the relevant sections above.
8. Ensure that all plots are covered in `analysis.md`.
9. Remove the "Additional Metric Visualizations" section entirely from `analysis.md` and instead add a single line referencing these insights where other insights are mentioned.
10. Modify `analysis.py` to save the raw data tables to CSV files in a `3.1.1cca/data/` directory and then remove those tables from `analysis.md`.
11. Modify the `analysis.py` to save the analysis of A, B, C, D as CSV files instead of .md files, saving them as CSV files with appropriate columns as follows:
    *   Analysis A (Start Node Invariance): Save as CSV with columns: `Algorithm`, `Mean (s)`, `StdDev (s)`, `Coeff Var (%)`
    *   Analysis B (Time Complexity): Save as CSV with columns: `Files`, `V`, `E`, `BFS(s)`, `DFS(s)`, `UFA Rank(s)`, `UFA Size(s)`
    *   Analysis C (Order Invariance): Save as CSV with columns: `Algorithm`, `Normal(s)`, `Shuffled(s)`, `Diff(s)`
    *   Analysis D (Connectivity Analysis): Save as CSV with comprehensive columns: `Step`, `Files`, `Nodes`, `Edges`, `Density`, `Avg Degree`, `Clustering Coeff`, `Num Components`, `GC Size (Nodes)`, `GC Size (Edges)`, `GC Coverage (%)`, `Diameter`, `Avg Path Length`
12. When running `analysis.py`, ensure full reproducibility by setting the seed for both Python's `random` module and NumPy's random generator using the same `RANDOM_SEED` variable. For example:

    ```python
    RANDOM_SEED = 67
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    ```
13. Make sure that all the data in the Key Observations section of `analysis.md` are the new and updated values.
14. Move the Complete performance table about Algorithm Comparison under that section in `analysis.md`.