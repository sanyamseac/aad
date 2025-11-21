# Friend Recommendation Systems – Comprehensive Analysis

This document provides a structured, end‑to‑end analysis of seven link prediction / friend recommendation algorithms applied to the Facebook Ego Network. It covers methodology, per‑algorithm deep dives (definition, correctness, complexity, empirical validation, strengths/weaknesses), comparative results, reproducibility steps, and scholarly citations.

---

## 1. What Are We Doing?
We implement and evaluate multiple graph‑based friend recommendation algorithms on a combined Facebook ego‑network. For each algorithm we:

- Implement the scoring function (heuristics from scratch; embeddings via random walks + Skip‑Gram)
- Generate top‑k recommendations for sampled nodes
- Measure Precision, Recall, F1, ROC‑AUC, MAP, runtime, memory
- Compare theoretical vs observed runtime trends

## 2. Practical Relevance
“People You May Know” systems rely on predicting plausible future edges. Heuristics (CN, JC, AA, PA, RA) give interpretable, low‑latency baselines; DeepWalk / Node2Vec embeddings capture higher‑order and structural patterns used in production social, content, and professional graphs.

## 3. Methodology Overview
1. Build graph via `graph.py` (SNAP Facebook ego circles merged)  
2. 80/20 edge train–test split; sample equal count random non‑edges for negatives  
3. Sample nodes (e.g. 50 for scalability; larger set for aggregate metrics)  
4. Run `recommend_friends` (heuristics) or embedding similarity ranking  
5. Compute Precision, Recall, F1, ROC‑AUC (sampled), MAP; gather runtime & memory  
6. Plot scalability, performance, and theoretical vs actual complexity in `results/`

## Dataset: Facebook Ego Network
| Property | Value |
|----------|-------|
| Total Nodes | 4,039 |
| Total Edges | 88,234 |
| Ego Nodes | 10 |
| Source | SNAP Facebook social circles (McAuley & Leskovec 2012) |

---

## 4. Algorithms Implemented
We group algorithms into heuristics and embedding‑based methods.

### 4.1 Heuristic‑Based Methods

- **Common Neighbors (CN)**  
   $\text{CN}(u,v) = |N(u) \cap N(v)|$

- **Jaccard Coefficient (JC)**  
   $\text{JC}(u,v) = \dfrac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}$

- **Adamic–Adar (AA)**  
   $\text{AA}(u,v) = \sum_{w \in N(u) \cap N(v)} \dfrac{1}{\log(\deg(w))}$

- **Preferential Attachment (PA)**  
   $\text{PA}(u,v) = \deg(u)\,\deg(v)$

- **Resource Allocation (RA)**  
   $\text{RA}(u,v) = \sum_{w \in N(u) \cap N(v)} \dfrac{1}{\deg(w)}$

### 4.2 Embedding‑Based Methods
6. DeepWalk (DW): Uniform random walks → Skip‑Gram → Embeddings → Cosine similarity.
7. Node2Vec (NV): Biased (p,q) random walks balancing BFS (community) vs DFS (structural roles) → Skip‑Gram → Embeddings → Cosine similarity.

---

## 5. Per‑Algorithm Detailed Analysis

Each subsection follows the required template:
1. Description & (brief) history
2. High‑level working
3. Proof of correctness (implementation vs formula)
4. Time & Space complexity (theoretical) + empirical runtime vs theory
5. Deliverables & insights
6. Strengths, weaknesses, ideal use cases
7. Plots / references to comparative visualizations

### 5.1 Common Neighbors (CN)
| Aspect | Details |
|--------|---------|
| Description | Classical triadic closure heuristic; counts mutual friends |
| Working | Score = size of intersection of neighbor sets |
| Correctness | Set intersection cardinality implements $|N(u) \cap N(v)|$ exactly |
| Complexity | Pair: $O(\deg(u)+\deg(v))$; node recommendation: $O(n\,\overline{d})$ |
| Empirical | Near‑linear runtime vs candidate count; theory vs actual curves align |
| Strengths | Fast, interpretable, minimal overhead |
| Weaknesses | Hub bias; no normalization |
| Ideal Use | Ultra‑low latency baseline generation |
| Plots | `performance_metrics_vs_size.png`, `scalability_analysis.png` |
| Insight | Strong baseline; often similar top sets to AA/RA |

### 5.2 Jaccard Coefficient (JC)
| Aspect | Details |
|--------|---------|
| Description | Set overlap ratio; penalizes large unions |
| Working | $\text{JC}(u,v)=\frac{|N(u)\cap N(v)|}{|N(u)\cup N(v)|}$ |
| Correctness | Directly computes intersection & union sets; guarded for empty union |
| Complexity | Same asymptotics as CN; slightly higher constant (union) |
| Empirical | Slightly slower than CN; lower precision/F1 in sparse graph |
| Strengths | Degree normalization reduces hub inflation |
| Weaknesses | Tiny scores; reduced discrimination; still local |
| Ideal Use | Denser graphs with balanced degree distribution |
| Plots | `algorithm_comparison.png` (lower F1), complexity plot stable |
| Insight | Normalization helpful but outweighed by sparsity effects here |

### 5.3 Adamic–Adar (AA)
| Aspect | Details |
|--------|---------|
| Description | Weights rare mutual neighbors more (Adamic & Adar 2003) |
| Working | $\sum_{w \in N(u)\cap N(v)} 1/\log(\deg(w))$ (skip $\deg=1$) |
| Correctness | Loop computes weighted sum exactly; reflects definition |
| Complexity | Same order as CN; log factor constant overhead |
| Empirical | Near top F1; modest runtime increase over CN |
| Strengths | Better ranking quality; mitigates hub dominance |
| Weaknesses | Local only; log undefined at degree 1 (skip rule) |
| Ideal Use | Skewed degree distributions (celebrity effects) |
| Plots | Strong precision/recall curves; complexity alignment |
| Insight | Effective trade‑off between accuracy and simplicity |

### 5.4 Preferential Attachment (PA)
| Aspect | Details |
|--------|---------|
| Description | Barabási–Albert inspired degree‑product heuristic |
| Working | $\deg(u)\times \deg(v)$ |
| Correctness | Direct degree product; no intersection step |
| Complexity | Pair $O(1)$; node recommendation $O(n)$ |
| Empirical | Fastest heuristic; lower precision due to hub bias |
| Strengths | Extremely low latency; trivial to compute |
| Weaknesses | Over‑recommends high‑degree nodes; ignores mutuality |
| Ideal Use | Pre‑filter candidate expansion stage |
| Plots | Minimal runtime; moderate F1 in comparisons |
| Insight | Best when speed outweighs accuracy needs |

### 5.5 Resource Allocation (RA)
| Aspect | Details |
|--------|---------|
| Description | Diffusion‑inspired heuristic (Zhou et al. 2009) |
| Working | $\sum_{w \in N(u)\cap N(v)} 1/\deg(w)$ |
| Correctness | Reciprocal degree accumulation; matches definition |
| Complexity | Same as AA/CN; division constant factor |
| Empirical | Often top F1 at $k=10$; runtime comparable to AA |
| Strengths | Penalizes hubs strongly; high ranking quality |
| Weaknesses | Degree‑1 neighbors contribute 1 (potential noise) |
| Ideal Use | High precision required; hub suppression scenarios |
| Plots | F1 leader in `algorithm_comparison.png` |
| Insight | Strongest pure heuristic under tested settings |

### 5.6 DeepWalk (DW)
| Aspect | Details |
|--------|---------|
| Description | Uniform random walk embeddings (Perozzi et al. 2014) |
| Working | Walk corpus → Skip‑Gram → node vectors → cosine similarity |
| Correctness | Follows published pipeline; similarity reflects proximity |
| Complexity | Train $O(|V|\cdot \text{num\_walks}\cdot \text{walk\_length})$; inference $O(n)$ per node |
| Empirical | Higher runtime due to embedding; improved recall potential |
| Strengths | Captures higher‑order structure & communities |
| Weaknesses | Training overhead; parameter tuning |
| Ideal Use | Batch/offline generation with richer features |
| Plots | Appears in embedding vs heuristic comparisons |
| Insight | Adds structural nuance beyond local heuristics |

### 5.7 Node2Vec (NV)
| Aspect | Details |
|--------|---------|
| Description | Biased walk embeddings with $(p,q)$ controls (Grover & Leskovec 2016) |
| Working | Second‑order biased walks → Skip‑Gram → cosine similarity |
| Correctness | Transition bias & embedding training match spec |
| Complexity | Similar to DeepWalk with bias overhead; proportional to walk tokens |
| Empirical | Walk parameter tuning changes recall/runtime trade‑offs |
| Strengths | Interpolates BFS (homophily) / DFS (structural roles) |
| Weaknesses | Parameter sensitivity; higher training cost |
| Ideal Use | Role/community aware recommendations in large graphs |
| Plots | `nv_hyperparameter_pq.png`, `nv_hyperparameter_walks.png`, `nv_vs_heuristics.png` |
| Insight | Most flexible; rewards careful hyperparameter tuning |

---

## 6. Theoretical vs Actual Complexity (Unified View)
Heuristic per‑node recommendation (scanning all candidates) theoretical costs:

| Algorithm | Theoretical Cost | Notes |
|-----------|------------------|-------|
| CN / JC / AA / RA | $O(n\,\overline{d})$ | Set builds & intersections dominate |
| PA | $O(n)$ | Degree lookups only |

Normalized theoretical vs empirical runtime lines (`theoretical_vs_actual_complexity.png`) form near‑parallel increasing trends (expected). Minor deviations stem from Python interpreter overhead & memory locality. Ratio (Actual/Theoretical) remains approximately constant, supporting adequacy of coarse models.

Observed speed ordering: PA < CN ≈ JC < AA ≈ RA < (DW/NV training phases).

---

## 7. Comparative Summary (All Algorithms)
Key observations (see `algorithm_comparison.png`, `performance_metrics_vs_size.png`, `nv_vs_heuristics.png`):

| Metric | Strongest Heuristic | Embedding Advantage |
|--------|---------------------|---------------------|
| Precision | AA / RA | Slight gains possible with tuned NV; sometimes lower |
| Recall | RA / AA | NV often improves recall with exploratory walks |
| F1 | RA (k=10) | NV competitive; benefits from walk length tuning |
| Runtime | PA (fastest) | Embeddings cost dominated by training |
| MAP | AA / RA | NV can match/exceed with balanced $(p,q)$ |

Trade‑offs: RA/AA yield strong precision–recall locally; NV enhances recall & structural nuance at higher cost; PA ideal as rapid prefilter.

---

## 8. Steps to Reproduce
1. Install dependencies: `networkx`, `numpy`, `pandas`, `matplotlib`, `psutil`, `scikit-learn`, `questionary`, `gensim`.
2. Run analyses:
   ```bash
   cd 3.4frs
   python analysis.py      # Heuristics + scalability + complexity
   python nv_analysis.py   # Node2Vec hyperparameters & comparisons
   ```
3. Review outputs in `3.4frs/results/` (CSV + PNG).
4. Adjust `TOP_K`, sample sizes, or hyperparameters for experiments.

---

## 9. Citations
Primary algorithm and dataset references:
1. Adamic, L. A., & Adar, E. (2003). Friends and neighbors on the Web. Social Networks, 25(3), 211–230.
2. Jaccard, P. (1901). Étude comparative de la distribution florale... Bulletin de la Société Vaudoise des Sciences Naturelles.
3. Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. Science, 286(5439), 509–512.
4. Zhou, T., Lü, L., & Zhang, Y.-C. (2009). Predicting missing links via local information. Eur. Phys. J. B 71, 623–630. (Resource Allocation index)
5. Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). DeepWalk: Online learning of social representations. KDD.
6. Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. KDD.
7. McAuley, J., & Leskovec, J. (2012). Learning to discover social circles in ego networks. NIPS (SNAP Facebook dataset).
8. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.

---

## 10. Future Improvements (Optional)
- Cache 2‑hop candidate sets to reduce full graph scans.
- Factor shared recommendation logic into a utility module.
- Add batched vectorized scoring for CN/AA/RA.
- Extend evaluation to AUC‑PR and calibration metrics.
- Integrate incremental / streaming update scenario tests.

---

## 11. Deliverables Recap
| Artifact | Location | Purpose |
|----------|----------|---------|
| Heuristic implementations | `3.4frs/*.py` | Score & recommend functions |
| Analysis script | `3.4frs/analysis.py` | Metrics, scalability, complexity plots |
| Node2Vec analysis | `3.4frs/nv_analysis.py` | Hyperparameters & comparisons |
| Result plots | `3.4frs/results/*.png` | Visual performance & complexity |
| CSV summaries | `3.4frs/results/*.csv` | Tabular metrics for downstream use |

---

End of structured analysis.
