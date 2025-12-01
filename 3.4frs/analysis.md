# Friend Recommendation Systems Analysis

## Overview

This section implements and evaluates five classical heuristic link prediction algorithms and one embedding-based method (Node2Vec) for friend recommendation in social networks. We test these algorithms on the Facebook Ego Network dataset to determine which performs best for predicting potential friendships.

**Bottom Line:** Resource Allocation achieves the best F1-score (0.0425), followed closely by Adamic-Adar (0.0418). Common Neighbors offers the best speed-accuracy tradeoff. Preferential Attachment is fastest but least accurate. Node2Vec embeddings provide competitive results but require 100x more training time.

---

## 1. What We're Doing

### Problem Statement

Given a social network graph, predict which non-connected users are likely to form connections. This is the core problem behind "People You May Know" features on social platforms.

**Approach:**
- Train-test split: Hide 20% of edges, predict them using remaining 80%
- Test 5 heuristic algorithms + Node2Vec embeddings (6 total)
- Evaluate on precision, recall, F1-score, ROC-AUC, and MAP
- Compare runtime and memory usage
- Validate theoretical complexity with actual measurements
- Perform hyperparameter optimization for Node2Vec

### Real-World Applications

- Social networks: Facebook, LinkedIn, Twitter friend suggestions
- E-commerce: Product recommendation based on co-purchase patterns
- Citation networks: Predicting future research collaborations
- Biological networks: Protein-protein interaction prediction

---

## 2. Dataset

Using Stanford SNAP Facebook Ego Network dataset (McAuley & Leskovec, 2012):

| Metric | Value |
|--------|-------|
| Total Users | 4,039 |
| Total Friendships | 88,234 |
| Ego Networks | 10 |
| Avg Friends/User | 43.69 |
| Data Source | SNAP/Facebook (2012) |

Each ego network contains:
- One central user (ego)
- All their friends
- Friendship connections between those friends

---

## 3. Algorithms Implemented

### Algorithm Comparison

| Algorithm | Formula | Interpretation | Time Complexity |
|-----------|---------|----------------|-----------------|
| **Common Neighbors (CN)** | $\|N(u) \cap N(v)\|$ | Count mutual friends | $O(n \cdot \bar{d})$ |
| **Jaccard Coefficient (JC)** | $\frac{\|N(u) \cap N(v)\|}{\|N(u) \cup N(v)\|}$ | Normalized mutual friends | $O(n \cdot \bar{d})$ |
| **Adamic-Adar (AA)** | $\sum_{w \in N(u) \cap N(v)} \frac{1}{\log(\deg(w))}$ | Weight rare friends higher | $O(n \cdot \bar{d})$ |
| **Preferential Attachment (PA)** | $\deg(u) \cdot \deg(v)$ | Popular connects to popular | $O(n)$ |
| **Resource Allocation (RA)** | $\sum_{w \in N(u) \cap N(v)} \frac{1}{\deg(w)}$ | Penalize high-degree friends | $O(n \cdot \bar{d})$ |

*Where: $N(u)$ = neighbors of u, $\deg(u)$ = degree of u, $\bar{d}$ = average degree*

---

## 4. Algorithm Details

### 4.1 Common Neighbors (CN)

**Concept:** Count how many mutual friends two users share.

**Implementation:**
```python
def compute_common_neighbors_score(G, u, v):
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    return len(neighbors_u.intersection(neighbors_v))
```

**How it works:**
1. Get all friends of user u
2. Get all friends of user v
3. Count the intersection

**Complexity:**
- Per pair: $O(\deg(u) + \deg(v))$ to build sets and compute intersection
- Per node (all candidates): $O(n \cdot \bar{d})$
- Space: $O(\deg(u) + \deg(v))$

**Results:**

![Complexity Analysis](results/theoretical_vs_actual_complexity.png)

**Figure 1:** Theoretical vs actual runtime shows near-perfect alignment (correlation ~0.98), validating the $O(n \cdot \bar{d})$ complexity model.

**Performance Metrics** (averaged over 10 graphs):
- Precision: 0.3316
- Recall: 0.0281
- F1-Score: 0.0406
- ROC-AUC: 0.9838
- MAP: 0.3951
- Runtime: 1.41 seconds
- Memory: ~150 MB

**Pros:**
- Fast and simple
- Easy to explain ("You have 5 mutual friends")
- No parameters to tune
- Solid baseline performance

**Cons:**
- Biased toward high-degree nodes (popular users)
- No normalization
- Only considers 2-hop neighborhood

**Best for:** Quick baseline, real-time systems, interpretable recommendations

---

### 4.2 Jaccard Coefficient (JC)

**Concept:** Normalize common neighbors by total neighbors to reduce bias.

**Implementation:**
```python
def compute_jaccard_coefficient(G, u, v):
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    intersection = neighbors_u.intersection(neighbors_v)
    union = neighbors_u.union(neighbors_v)
    return len(intersection) / len(union) if len(union) > 0 else 0.0
```

**Complexity:** Same as CN: $O(n \cdot \bar{d})$ with slightly higher constant due to union computation.

**Performance Metrics:**
- Precision: 0.3209
- Recall: 0.0261
- F1-Score: 0.0382
- ROC-AUC: 0.9724
- MAP: 0.3620
- Runtime: 2.06 seconds

**Observation:** Normalization hurts performance in sparse graphs. Scores compressed near zero, reducing ranking quality.

**Pros:**
- Degree normalization reduces hub bias
- Bounded [0,1] scores

**Cons:**
- Lower performance than CN in sparse networks
- Many tied scores at low values

**Best for:** Dense networks with balanced degree distribution

---

### 4.3 Adamic-Adar (AA)

**Concept:** Weight mutual friends inversely by their popularity. Rare connections are more informative.

**Implementation:**
```python
import math

def compute_adamic_adar_score(G, u, v):
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    common = neighbors_u.intersection(neighbors_v)
    
    score = 0.0
    for w in common:
        degree_w = G.degree(w)
        if degree_w > 1:
            score += 1.0 / math.log(degree_w)
    return score
```

**Intuition:** Being introduced by someone with 10 friends is more meaningful than being introduced by someone with 1000 friends (celebrity effect).

**Complexity:** $O(n \cdot \bar{d})$ - same asymptotic cost as CN, but ~30% slower due to log computations.

![Performance Comparison](results/performance_metrics_vs_size.png)

**Figure 2:** Adamic-Adar consistently outperforms CN and JC across graph sizes.

**Performance Metrics:**
- Precision: 0.3403
- Recall: 0.0289
- F1-Score: 0.0418
- ROC-AUC: 0.9858
- MAP: 0.4080
- Runtime: 1.86 seconds

**Pros:**
- Best precision among local methods
- Reduces celebrity bias
- Well-established in literature

**Cons:**
- 30% slower than CN
- Undefined for degree-1 nodes (handled by skipping)

**Best for:** Social networks with influencers/hubs, precision-critical applications

---

### 4.4 Preferential Attachment (PA)

**Concept:** Likelihood of connection is proportional to product of degrees. "Rich get richer."

**Implementation:**
```python
def compute_preferential_attachment_score(G, u, v):
    return G.degree(u) * G.degree(v)
```

**Complexity:** 
- Per pair: $O(1)$ - just two lookups and a multiplication
- Per node: $O(n)$ for candidates + $O(n \log n)$ for sorting
- **Fastest algorithm by far**

![Scalability](results/scalability_analysis.png)

**Figure 3:** PA shows dramatically lower runtime compared to neighborhood-based methods.

**Performance Metrics:**
- Precision: 0.0510
- Recall: 0.0146
- F1-Score: 0.0172
- ROC-AUC: 0.8374
- MAP: 0.1346
- Runtime: 0.65 seconds (2.16x faster than CN)

**Pros:**
- Extremely fast
- Simple to implement
- Minimal memory usage

**Cons:**
- Heavily biased toward popular users
- Lowest accuracy
- Ignores neighborhood structure entirely

**Best for:** Rapid candidate generation, massive graphs, prefiltering step

---

### 4.5 Resource Allocation (RA)

**Concept:** Model as resource diffusion. Friends with fewer connections transfer more "resource."

**Implementation:**
```python
def compute_resource_allocation_score(G, u, v):
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    common = neighbors_u.intersection(neighbors_v)
    
    score = 0.0
    for w in common:
        score += 1.0 / G.degree(w)
    return score
```

**Difference from AA:** Linear penalty ($1/\deg(w)$) vs logarithmic ($1/\log(\deg(w))$). More aggressive hub suppression.

**Complexity:** $O(n \cdot \bar{d})$ - slightly faster than AA (division vs logarithm).

**Performance Metrics:**
- Precision: 0.3563 🏆
- Recall: 0.0291 🏆
- F1-Score: 0.0425 🏆
- ROC-AUC: 0.9871 🏆
- MAP: 0.4201 🏆
- Runtime: 1.80 seconds

**Winner:** Highest F1-score, precision, recall, ROC-AUC, and MAP across all heuristic methods.

**Pros:**
- Best overall accuracy
- Strong hub suppression
- Consistent performance

**Cons:**
- ~30% slower than CN
- Still limited to 2-hop patterns

**Best for:** Production friend recommendation, when accuracy matters most

---

### 4.6 Node2Vec (Embedding-Based Method)

**Concept:** Learn low-dimensional vector representations of nodes through biased random walks, then use cosine similarity for link prediction.

**Implementation:**
```python
from node2vec import Node2Vec

def node2vec_predict(G, p=1.0, q=1.0, num_walks=10, walk_length=80):
    # Train embedding model
    node2vec = Node2Vec(G, dimensions=128, walk_length=walk_length,
                        num_walks=num_walks, p=p, q=q, workers=4)
    model = node2vec.fit(window=10, min_count=1)
    
    # Compute similarities for predictions
    # (cosine similarity between embeddings)
```

**Hyperparameters:**
- **p (return parameter):** Controls likelihood of returning to previous node
- **q (in-out parameter):** Controls BFS vs DFS-like exploration
- **num_walks:** Number of random walks per node
- **walk_length:** Steps in each walk

**Walk Strategies:**
- **DeepWalk (p=1, q=1):** Unbiased random walk
- **BFS-like (p<1, q>1):** Local neighborhood exploration
- **DFS-like (p>1, q<1):** Explores distant nodes
- **Balanced (p≈0.7, q≈0.7):** Balanced local/global exploration

![Node2Vec vs Heuristics](results/nv_vs_heuristics.png)

**Figure 4:** Node2Vec variants compared to traditional heuristics on the largest graph (n=4039).

**Complexity:**
- **Training:** $O(k \cdot l \cdot n \cdot \log n)$ where k=num_walks, l=walk_length
- **Inference:** $O(n \log n)$ for similarity computations
- **Total:** Much slower than heuristics but captures global structure

**Performance Metrics (p=0.7, q=0.7, 10 walks, 80 steps):**
- Precision: 0.2460
- Recall: 0.0069
- F1-Score: 0.0134
- ROC-AUC: 0.9897
- MAP: 0.3177
- Training Time: ~200 seconds per graph
- Total Runtime: 211.45 seconds (150x slower than CN)

**Hyperparameter Analysis:**

![Hyperparameter P/Q](results/nv_hyperparameter_pq.png)

**Figure 5:** Impact of p and q parameters on F1-score and runtime. Lower p/q values (more local exploration) generally yield better F1-scores.

![Hyperparameter Walks](results/nv_hyperparameter_walks.png)

**Figure 6:** Impact of number of walks and walk length on performance. More walks and longer paths improve accuracy but increase training time significantly.

**Key Findings:**
- **Best Configuration:** p=0.7, q=2.0, walks=20, length=120
  - F1: 0.1598 (best among Node2Vec configs)
  - Runtime: 23.28s (still 16x slower than RA)
  - Trade-off: 4x better F1 than fastest config, but 8x slower
- **Fastest Viable:** p=1.0, q=1.0, walks=5, length=40
  - F1: 0.1217
  - Runtime: 2.92s (2x slower than RA, but competitive)
  - Trade-off: Acceptable for real-time scenarios

**Comparison with Heuristics:**

| Method | Precision | F1 | ROC-AUC | Runtime (s) | Speed vs CN |
|--------|-----------|-----|---------|-------------|-------------|
| Resource Allocation | 0.3257 | 0.0173 | 0.9998 | 1.70 | 0.91x |
| Adamic-Adar | 0.3064 | 0.0163 | 0.9997 | 1.71 | 0.90x |
| Common Neighbors | 0.3043 | 0.0162 | 0.9992 | 1.55 | 1.0x |
| **Node2Vec (Balanced)** | 0.2460 | 0.0134 | 0.9897 | **211.45** | **0.007x** |
| Node2Vec (DFS-like) | 0.2560 | 0.0141 | 0.9858 | 212.14 | 0.007x |
| Node2Vec (DeepWalk) | 0.2414 | 0.0132 | 0.9871 | 146.84 | 0.011x |

**Pros:**
- Captures global network structure (beyond 2-hop)
- Pre-trained embeddings can be reused for multiple tasks
- Excellent ROC-AUC scores (near 0.99)
- Flexible: p/q parameters tune exploration strategy

**Cons:**
- 150x slower than heuristic methods
- Lower precision and F1-score
- Requires hyperparameter tuning
- Memory intensive for large graphs
- Training time grows with graph size

![Node2Vec Scalability](results/nv_scalability.png)

**Figure 7:** Node2Vec training time scales super-linearly with graph size, making it impractical for graphs with >5000 nodes without distributed computing.

**Best for:** 
- Multi-task learning (embeddings used for multiple predictions)
- When global structure matters (community-based recommendations)
- Offline batch processing with time budget
- When interpretability is not critical

**Not Recommended for:**
- Real-time recommendations (<1 second latency)
- When only link prediction is needed
- Resource-constrained environments

---

## 5. Experimental Results

### 5.1 Performance Summary

**Table: Algorithm Performance (Average across 10 graphs)**

| Algorithm | Precision | Recall | F1 | ROC-AUC | MAP | Runtime (s) | Speed vs CN |
|-----------|-----------|--------|-----|---------|-----|-------------|-------------|
| **Resource Allocation** 🥇 | **0.3563** | **0.0291** | **0.0425** | **0.9871** | **0.4201** | 1.80 | 0.79x |
| **Adamic-Adar** 🥈 | 0.3403 | 0.0289 | 0.0418 | 0.9858 | 0.4080 | 1.86 | 0.76x |
| **Common Neighbors** 🥉 | 0.3316 | 0.0281 | 0.0406 | 0.9838 | 0.3951 | 1.41 | 1.0x |
| **Jaccard Coefficient** | 0.3209 | 0.0261 | 0.0382 | 0.9724 | 0.3620 | 2.06 | 0.68x |
| **Preferential Attachment** | 0.0510 | 0.0146 | 0.0172 | 0.8374 | 0.1346 | **0.65** | **2.16x** |

### 5.2 Key Findings

**Accuracy Rankings:**
1. Resource Allocation wins on all accuracy metrics
2. Adamic-Adar is close second
3. Common Neighbors is solid third
4. Jaccard underperforms in sparse graphs
5. Preferential Attachment sacrifices accuracy for speed

**Speed Rankings:**
1. Preferential Attachment: 0.65s (fastest)
2. Common Neighbors: 1.41s (baseline)
3. Resource Allocation: 1.80s
4. Adamic-Adar: 1.86s
5. Jaccard Coefficient: 2.06s (slowest)

**Speed-Accuracy Tradeoff:**
- **Best Accuracy:** RA gives +4.7% F1 over CN at 28% slowdown
- **Best Speed:** PA gives 2.16x speedup but -58% F1
- **Best Balance:** CN offers 95.5% of RA's accuracy at 78% of runtime

---

## 6. Algorithm Selection Guide

**Choose based on your priorities:**

| Priority | Algorithm | Reason |
|----------|-----------|--------|
| **Maximum Accuracy** | Resource Allocation | Best F1, precision, recall, MAP among all methods |
| **Speed-Accuracy Balance** | Common Neighbors | 95.5% of RA's accuracy, 78% of runtime |
| **Maximum Speed** | Preferential Attachment | 2.16x faster, good for prefiltering |
| **Reduce Hub Bias** | Resource Allocation or AA | Strong degree penalties |
| **Interpretability** | Common Neighbors | "You have X mutual friends" |
| **Dense Networks** | Jaccard Coefficient | Normalization helps when avg degree >100 |
| **Global Structure** | Node2Vec | Captures communities and long-range patterns |
| **Multi-task Learning** | Node2Vec | Embeddings reusable for classification, clustering |
| **Batch Processing** | Node2Vec (optimized) | Acceptable when offline training is viable |

---

## 7. How to Run

### Setup

```bash
# Install dependencies
pip install networkx numpy pandas matplotlib scikit-learn psutil

# Navigate to directory
cd d:/aad/3.4frs
```

### Execute Analysis

```bash
# Run complete analysis
python analysis.py
```

**Expected Output:**

**CSV Files:**
- `results/comprehensive_analysis.csv` - All heuristic algorithm results
- `results/nv_vs_heuristics.csv` - Node2Vec comparison with heuristics
- `results/nv_hyperparameter_exploration.csv` - Node2Vec parameter tuning results
- `results/nv_scalability.csv` - Node2Vec scalability analysis

**Visualizations:**
- `results/scalability_analysis.png` - Runtime vs graph size comparison
- `results/performance_metrics_vs_size.png` - Accuracy metrics across graphs
- `results/theoretical_vs_actual_complexity.png` - Complexity validation
- `results/nv_vs_heuristics.png` - Node2Vec vs traditional methods
- `results/nv_hyperparameter_pq.png` - Impact of p/q parameters
- `results/nv_hyperparameter_walks.png` - Impact of walk parameters
- `results/nv_scalability.png` - Node2Vec training time scaling

**Runtime:** ~5-8 minutes for heuristics, +30-60 minutes if running Node2Vec analysis

### Customization

Edit `analysis.py` to modify:
- Sample size: Line ~580, change `sample_size=50`
- Top-k recommendations: Line ~595, change `top_k=10`
- Graph subset: Line ~532, change `range(1, 11)`

---

## 8. References

1. **Adamic, L. A., & Adar, E.** (2003). Friends and neighbors on the Web. *Social Networks*, 25(3), 211–230.

2. **Zhou, T., Lü, L., & Zhang, Y.-C.** (2009). Predicting missing links via local information. *The European Physical Journal B*, 71(4), 623–630.

3. **Barabási, A.-L., & Albert, R.** (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509–512.

4. **Liben-Nowell, D., & Kleinberg, J.** (2007). The link-prediction problem for social networks. *JASIST*, 58(7), 1019–1031.

5. **McAuley, J., & Leskovec, J.** (2012). Learning to discover social circles in ego networks. *NIPS*.

6. **Newman, M. E. J.** (2010). *Networks: An Introduction*. Oxford University Press.

7. **Lü, L., & Zhou, T.** (2011). Link prediction in complex networks: A survey. *Physica A*, 390(6), 1150–1170.

8. **Grover, A., & Leskovec, J.** (2016). node2vec: Scalable feature learning for networks. *KDD*, 855–864.

9. **Perozzi, B., Al-Rfou, R., & Skiena, S.** (2014). DeepWalk: Online learning of social representations. *KDD*, 701–710.

---

## Conclusion

This analysis evaluated five classical heuristic algorithms and one embedding-based method (Node2Vec) for link prediction on real Facebook network data. **Resource Allocation emerged as the winner among heuristics** with F1=0.0425, followed closely by Adamic-Adar (F1=0.0418), though Common Neighbors offers excellent speed-accuracy balance at F1=0.0406 with comparable runtime.

**Key Findings:**

1. **Heuristic Methods Dominate for Speed:** Simple neighborhood-based methods (CN, AA, RA) provide excellent accuracy with 1-2 second runtimes.

2. **Node2Vec Trade-off:** Embedding methods achieve competitive ROC-AUC (~0.99) but sacrifice 150x runtime and lower precision/F1 scores. Only viable for batch processing or multi-task scenarios.

3. **Optimal Configurations:**
   - **Best Overall:** Resource Allocation (F1=0.0425, runtime=1.80s)
   - **Best Speed-Accuracy:** Common Neighbors (F1=0.0406, runtime=1.41s)
   - **Fastest:** Preferential Attachment (runtime=0.65s, but F1=0.0172)
   - **Best Node2Vec:** p=0.7, q=2.0, walks=20, length=120 (F1=0.1598, runtime=23.28s)

**Production Recommendations:**

| Use Case | Algorithm | Rationale |
|----------|-----------|-----------|
| **Real-time API (<2s)** | Common Neighbors or RA | Fast, accurate, interpretable |
| **Maximum Accuracy** | Resource Allocation | Best F1, precision, recall, MAP |
| **Massive Scale** | Preferential Attachment | 2x faster, good for prefiltering |
| **Multi-task ML** | Node2Vec (p=1, q=1, walks=5) | Reusable embeddings, ~3s training |
| **Batch Processing** | Node2Vec (optimized params) | Higher recall potential, offline acceptable |

The strong correlation between theoretical and empirical complexity validates our implementation and confirms these algorithms scale predictably for practical deployment.
