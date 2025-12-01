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
3.  Include *all* relevant metrics in `analysis.md`. Do not omit any metrics.
4.  Include all information about an algorithm, including complexity analysis (theoretical and empirical), under the algorithm's dedicated header in `analysis.md`. Avoid separating complexity analysis from the algorithm's main section.
5.  Integrate all relevant information from `metrics.md` into `analysis.md` where appropriate. Do not delete `metrics.md`.
6.  Remove the "How it works" section from `analysis.md`.
7.  Remove the "Appendix" section from `analysis.md` and integrate the formulae into the relevant sections above.
8. Ensure that all plots are covered in `analysis.md`.
9. Remove the "Additional Metric Visualizations" section entirely from `analysis.md` and instead add a single line referencing these insights where other insights are mentioned.