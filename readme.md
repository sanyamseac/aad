# Social Network Analysis and Friend Recommendation System

## Project Overview

This project implements and analyzes fundamental graph algorithms on the Facebook Social Circles dataset from Stanford SNAP. The implementation focuses on four core areas of social network analysis:

1. **Connected Components Analysis** (3.1.1cca) - Graph connectivity algorithms
2. **Centrality Analysis** (3.1.2ca) - Node importance metrics
3. **Community Detection** (3.2cd) - Social circle identification
4. **Friend Recommendation Systems** (3.4frs) - Link prediction algorithms

All algorithms are implemented from with comprehensive performance analysis and visualization.

---

## Project Structure

```
aad/
├── readme.md                         # Main project documentation (this file)
├── requirements.txt                  # Python dependencies
├── graph.py                          # Shared graph loading utilities
├── visualize.py                      # Shared visualization utilities
├── aad.js_2.pdf                      # Project Proposal
├── complete_facebook_network.png     # Whole Network Graph
│
├── dataset/                          # SNAP Facebook Ego Networks
│   ├── 0.edges, 0.feat, 0.circles    # Ego network files (10 users)
│   ├── 107.edges, 107.feat, ...
│   └── ...
│
├── 3.1.1cca/                         # Connected Components Analysis
│   ├── analysis.py                   # Main analysis script
│   ├── bfs.py                        # Breadth-First Search
│   ├── dfs.py                        # Depth-First Search
│   ├── ufa_by_rank.py                # Union-Find by Rank
│   ├── ufa_by_size.py                # Union-Find by Size
│   └── results/                      # Generated outputs
│       ├── plots/                    # Algorithm visualizations
│       ├── data/                     # Performance CSV files
│       └── analysis/                 # Metric reports
│
├── 3.1.2ca/                          # Centrality Analysis
│   ├── analysis.py                   # Main analysis script
│   ├── dc.py                         # Degree Centrality
│   ├── bc.py                         # Betweenness Centrality
│   ├── cc.py                         # Closeness Centrality
│   ├── ec.py                         # Eigenvector Centrality
│   ├── scalability_analysis.py       # Complexity verification
│   └── results/                      # Generated outputs
│
├── 3.2cd/                            # Community Detection
│   ├── analysis.py                   # Main analysis script
│   ├── gn.py                         # Girvan-Newman Algorithm
│   ├── lm.py                         # Louvain Modularity
│   ├── la.py                         # Leiden Algorithm
│   └── results/                      # Generated outputs
│
└── 3.4frs/                           # Friend Recommendation Systems
    ├── analysis.py                   # Main analysis script
    ├── cm.py                         # Common Neighbors
    ├── aa.py                         # Adamic-Adar
    ├── jc.py                         # Jaccard Coefficient
    ├── pa.py                         # Preferential Attachment
    ├── ra.py                         # Resource Allocation
    ├── nv.py                         # Node2Vec (Bonus)
    ├── nv_analysis.py                # Node2Vec analysis (Bonus)
    └── results/                      # Generated outputs
```

## Prerequisites

- Latest version of Python

---

## Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/sanyamseac/aad.git
cd aad
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# refer doc
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `networkx` - Graph data structures
- `matplotlib`, `seaborn` - Visualization
- `pandas` - Data analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning metrics
- `gensim` - Node2Vec embeddings
- `tqdm` - Progress bars
- `psutil` - System monitoring

---

## Quick Start

### Run All Analyses

Each module has its own comprehensive analysis script:

```bash
# Connected Components Analysis
cd 3.1.1cca
python analysis.py

# Centrality Analysis
cd ../3.1.2ca
python analysis.py

# Community Detection
cd ../3.2cd
python analysis.py

# Friend Recommendation Systems
cd ../3.4frs
python analysis.py
```

### Run Individual Algorithms

Each algorithm can be run independently:

```bash
# Example: Run Breadth-First Search
cd 3.1.1cca
python bfs.py

# Example: Run Betweenness Centrality
cd 3.1.2ca
python bc.py

# Example: Run Louvain Algorithm
cd 3.2cd
python lm.py

# Example: Run Common Neighbors
cd 3.4frs
python cm.py
```

---

## Dataset

**Source**: Stanford SNAP Facebook Social Circles  

The dataset contains 10 ego networks from Facebook users.

**Files per ego network:**
- `.edges` - Friend connections
- `.feat` - Node features
- `.featnames` - Feature descriptions
- `.circles` - Ground truth communities
- `.egofeat` - Ego node features

---
