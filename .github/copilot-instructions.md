---
description: AI rules derived by SpecStory from the project AI interaction history
globs: *
---

## HEADERS

## TECH STACK

## PROJECT DOCUMENTATION & CONTEXT SYSTEM

## CODING STANDARDS

## DEBUGGING

## WORKFLOW & RELEASE RULES

## ANALYSIS.MD FILE STRUCTURE

The `analysis.md` file must adhere to the following structure:

- Heading
- Description
    1. What are we doing in this section?
    2. How's it related to the practical world?
    3. How are we getting our results?
- Algos implemented
- For each algo:
    1. Description about algo (history if any)
    2. Explanation and high level working of algorithm
    3. Proof of correctness of algorithm
    4. Time and Space complexity
        - Calculate theoretically
        - Import and show the graphs of actual running time vs theoretical time complexity. The graph should approximate a straight line.
        - Explain the results
    5. Mention deliverables and insights if any
    6. Strength, weakness, ideal use cases. Include real-world use cases where applicable (e.g., Facebook connection system).
    7. Plot multiple graphs showing comparisons based on multiple things.
- Compare all algos, plot graphs, tables, explain the difference etc. Explain what each metric provides in terms of information related to the SNAP-EGO Facebook graph. For each metric, include the consequence of the data we got from the plot. For each metric, include the consequence of the data we got from the plot. For example: For density plot, the decreasing trend means that not all users are friends with each other and we can narrow down a list for friend recommendation. Do this for every metric that can have this.
- Steps to reproduce
- Citations

When creating the `analysis.md` file:

- Take help from all `.md` files in the analysis directory, including `complexity.md`, `metrics.md`, and `progress.md`.
- Include all plots from the plots folder, referencing them with paths that allow automatic fetching into the `analysis.md` file (using the `![]()` markdown syntax).
- Rigorously follow the file structure and include all the points for all algorithms implemented.
- Include proper explanations where required.
- Always stay within the limits of the code and do not reference anything not covered in the code.
- Ensure to include "Steps to reproduce" and "Citations" at the end of the file.
- Ensure to include "Steps to reproduce" and reference repository scripts like analysis.py or bfs.py.
- Ensure that when including plots from the plots folder, the paths are correct and the images are rendered properly in the markdown.
- Explain what each metric provides in terms of information related to the SNAP-EGO Facebook graph. For each metric, include the consequence of the data we got from the plot. For each metric, include the consequence of the data we got from the plot. For example: For density plot, the decreasing trend means that not all users are friends with each other and we can narrow down a list for friend recommendation. Do this for every metric that can have this.
- Aim to create a perfect analysis, so that anyone can understand exactly what we have done and why and why is it important and what are its implications on a use case.
- For each metric, include the consequence of the data we got from the plot. For example: For density plot, the decreasing trend means that not all users are friends with each other and we can narrow down a list for friend recommendation. Do this for every metric that can have this.
- Include information from `complexity.md`, `metrics.md`, and `progress.md`'s details into `analysis.md`.

## DOCSTRINGS

- All functions in `ufa_by_rank.py`, `ufa_by_size.py`, and `analysis.py` must include concise docstrings explaining their purpose, parameters, and return values.
- The docstrings in `ufa_by_rank.py`, `ufa_by_size.py`, and `analysis.py` should follow the format used in `bfs.py` or `dfs.py`.
- In `analysis.py` explain each function from `run_analysis_A` to `run_analysis_D`, to include what it does and how, include input parameters and results as well.