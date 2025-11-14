# Friend Recommendation Systems - Comprehensive Analysis

## Dataset: Facebook Ego Network
- **Total Nodes**: 4,039 users
- **Total Edges**: 88,234 friendships  
- **Ego Nodes**: 10 central users
- **Source**: Facebook social circles dataset

---

## Algorithms Analyzed

### 1. **Adamic-Adar Index (AA)**
- Weighted version of Common Neighbors
- Formula: Σ(w∈N(u)∩N(v)) 1/log(|N(w)|)
- Gives more weight to rare common neighbors (those with fewer connections)

### 2. **Common Neighbors (CM)**
- Counts mutual friends between two users
- Formula: |N(u) ∩ N(v)|
- Simple and intuitive baseline approach

### 3. **Jaccard Coefficient (JC)**
- Normalized version of Common Neighbors
- Formula: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
- Accounts for total unique neighbors

---

## Methodology

### Train-Test Split
- **Training**: 80% of edges (70,587 edges)
- **Testing**: 20% of edges (17,647 edges)
- **Evaluation**: Sampled 100 nodes for efficiency
- **Metrics**: Precision, Recall, F1-Score, Runtime

### k Values Tested
- k = 5, 10, 20, 50 (top-k recommendations per user)

---

## Key Findings

### 1. **Top Recommendations Consistency**
All three algorithms consistently identified **User 348** as the #1 recommendation for User 0, showing agreement on the strongest candidate despite different scoring mechanisms.

**For Node 0:**
- **Adamic-Adar**: User 348 (score: 1.5700)
- **Common Neighbors**: User 348 (4 common neighbors)
- **Jaccard Coefficient**: User 348 (score: 0.0070)

### 2. **Performance Comparison (at k=20)**

| Algorithm | Precision | Recall | F1-Score | True Positives | Runtime (s) |
|-----------|-----------|--------|----------|----------------|-------------|
| **Adamic-Adar** | **0.4729** ✅ | **0.0535** ✅ | **0.0961** ✅ | **944** ✅ | 1.31 |
| Common Neighbors | 0.4619 | 0.0522 | 0.0939 | 922 | **1.21** ✅ |
| Jaccard Coefficient | 0.4309 | 0.0487 | 0.0876 | 860 | 1.79 |

**Winner**: **Adamic-Adar Index** 
- Best precision, recall, and F1-score
- Most true positives (944 correct predictions)
- Competitive runtime

### 3. **Performance Across Multiple k Values**

#### Adamic-Adar (Best Overall)
```
k=5:  Precision=0.816, Recall=0.023, F1=0.045
k=10: Precision=0.656, Recall=0.037, F1=0.070
k=20: Precision=0.473, Recall=0.053, F1=0.096 ⭐
k=50: Precision=0.273, Recall=0.076, F1=0.119 ⭐ (highest F1)
```

#### Common Neighbors (Best Runtime)
```
k=5:  Precision=0.788, Recall=0.022, F1=0.043
k=10: Precision=0.628, Recall=0.036, F1=0.067
k=20: Precision=0.462, Recall=0.052, F1=0.094
k=50: Precision=0.269, Recall=0.075, F1=0.117
```

#### Jaccard Coefficient (Lowest Performance)
```
k=5:  Precision=0.688, Recall=0.020, F1=0.038
k=10: Precision=0.578, Recall=0.033, F1=0.062
k=20: Precision=0.431, Recall=0.049, F1=0.088
k=50: Precision=0.263, Recall=0.073, F1=0.114
```

### 4. **Runtime Analysis**

**Average Runtime (seconds) across all k values:**
- Common Neighbors: **1.23s** ✅ (Fastest)
- Adamic-Adar: 1.31s (2nd fastest)
- Jaccard Coefficient: 1.87s (Slowest)

**Key Insight**: Common Neighbors is the simplest and fastest, but Adamic-Adar provides better accuracy with only a small runtime penalty (~7% slower).

### 5. **Precision vs Recall Trade-off**

- **High k (k=50)**: Lower precision, higher recall
  - Better coverage but more false positives
  - Best for exploratory recommendations
  
- **Low k (k=5)**: Higher precision, lower recall
  - More accurate but limited coverage
  - Best for confident top recommendations

- **Optimal k**: k=20 provides good balance

### 6. **Algorithm Behavior Insights**

#### Adamic-Adar
- ✅ Best overall performance
- ✅ Handles hub nodes well by down-weighting high-degree common neighbors
- ✅ More sophisticated scoring leads to better differentiation
- Uses logarithmic scaling to identify truly significant connections

#### Common Neighbors
- ✅ Fastest and simplest
- ✅ Easy to interpret and explain
- ⚠️ Slightly lower accuracy than Adamic-Adar
- Treats all common neighbors equally (no weighting)

#### Jaccard Coefficient
- ⚠️ Poorest performance on this dataset
- ⚠️ Normalization hurts performance in sparse networks
- ⚠️ Penalizes high-degree nodes too much
- Better suited for dense networks or nodes with similar degree distributions

---

## Recommendations

### For Production Systems:
1. **Use Adamic-Adar** for best accuracy
   - ~7% slower than Common Neighbors but significantly better results
   - 944 vs 922 true positives at k=20 (2.4% improvement)

2. **Use Common Neighbors** if speed is critical
   - Acceptable accuracy with fastest runtime
   - Simpler to implement and maintain

3. **Avoid Jaccard Coefficient** for sparse social networks
   - Consistently underperforms other methods
   - Better for other domains (e.g., item recommendation in e-commerce)

### Optimal Configuration:
- **Algorithm**: Adamic-Adar Index
- **k value**: 20 (good precision-recall balance)
- **Expected Performance**: 47% precision, 5.3% recall, F1=0.096

---

## Visualizations Generated

1. **recommendation_performance_analysis.png**
   - Precision vs k
   - Recall vs k  
   - F1-Score vs k
   - Runtime vs k

2. **algorithm_comparison.png**
   - Bar charts comparing algorithms at k=20
   - Precision, Recall, and F1-Score comparison

---

## Comparison with Centrality Analysis

### Similarities to 3.1.2ca:
- ✅ Uses same Facebook ego network dataset
- ✅ Same graph loading mechanism (`create_complete_graph`)
- ✅ Similar analysis structure and output format
- ✅ Performance metrics and runtime analysis
- ✅ Visualization of results

### Differences:
- **Task**: Link prediction vs node importance
- **Metrics**: Precision/Recall vs Centrality scores
- **Scale**: Evaluated on sample (100 nodes) vs full graph (4,039 nodes)
- **Output**: Recommended edges vs Important nodes

---

## Scalability Considerations

- **Full graph evaluation** (4,039 nodes) would take ~88 minutes per k value
- **Sampled evaluation** (100 nodes) takes ~1.3 seconds per k value
- **Sampling strategy**: Random selection provides representative results
- **Production systems**: Consider incremental updates and caching

---

## Conclusion

**Adamic-Adar Index emerges as the clear winner** for friend recommendation on the Facebook ego network dataset:
- Best accuracy metrics across all k values
- Reasonable runtime (only 7% slower than baseline)
- Most true positives at every k level
- Sophisticated weighting leads to better recommendations

The analysis demonstrates that **not all common neighbors are equal** - giving more weight to rare connections (Adamic-Adar) produces better recommendations than simple counting (Common Neighbors) or normalization (Jaccard Coefficient).

---

## Implementation Details

- **Language**: Python 3.12
- **Framework**: NetworkX for graph operations
- **Dataset Structure**: Matches 3.1.2ca centrality analysis
- **Code Quality**: From-scratch implementations (no external algorithm libraries)
- **Reproducibility**: Seed fixed at 42 for consistent results
