# Friend Recommendation Systems – Comprehensive Analysis

In this section, we analyze seven link prediction algorithms for
friend recommendation on the Facebook Ego Network: five heuristic methods
(Common Neighbors, Jaccard Coefficient, Adamic–Adar, Preferential Attachment,
Resource Allocation) and two embedding-based methods (DeepWalk, Node2Vec).
Our goal is to compare their accuracy, runtime, and practical suitability for
real-world social networks.

## 1. What are we doing in this section?

We design, implement, and evaluate multiple graph-based friend recommendation
algorithms on the complete Facebook ego-network dataset. For each algorithm,
we:

- Implement the scoring function from scratch in Python.
- Generate top‑k friend recommendations.
- Measure precision, recall, F1‑score, and runtime.
- Compare their performance and trade‑offs.

## 2. How is this related to the practical world?

Modern social platforms (Facebook, LinkedIn, Instagram, etc.) rely heavily on
“People You May Know” systems. A common and robust way to build such systems is
to treat the social network as a graph and predict missing edges. The
heuristic algorithms we study (CN, JC, AA, PA, RA) are classic link prediction
methods used in graph mining, and the embedding-based algorithms (DeepWalk and
Node2Vec) represent modern random-walk-based representation learning methods
used in industry.

## 3. How are we getting our results?

- We construct a combined Facebook ego-network graph using `graph.py` and the
   original SNAP dataset.
- We perform an 80–20 train–test split over edges.
- We sample a subset of nodes and generate top‑k recommendations for each of
   them using each algorithm.
- We compare predicted edges against held‑out test edges to compute
   precision/recall/F1.
- We log runtimes and generate plots using `3.4frs/analysis.py`, which writes
   all figures into `3.4frs/results/`.

## Dataset: Facebook Ego Network

- **Total Nodes**: 4,039 users
- **Total Edges**: 88,234 friendships  
- **Ego Nodes**: 10 central users
- **Source**: Facebook social circles dataset

---

## 4. Algorithms Implemented

We implement two families of algorithms:

### 4.1 Heuristic-based methods

1. **Common Neighbors (CM)** – baseline that counts mutual friends:  
   \( \text{CN}(u,v) = |N(u) \cap N(v)| \).

2. **Jaccard Coefficient (JC)** – normalizes common neighbors by total distinct
   neighbors:  
   \( \text{JC}(u,v) = \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|} \).

3. **Adamic–Adar Index (AA)** – weights rare common friends more:  
   \( \text{AA}(u,v) = \sum_{w \in N(u)\cap N(v)} \frac{1}{\log(\deg(w))} \).

4. **Preferential Attachment (PA)** – assumes popular users attract more links:  
   \( \text{PA}(u, v) = \deg(u) \cdot \deg(v) \).

5. **Resource Allocation (RA)** – similar to AA but uses 1/deg(w) instead of
   1/log(deg(w)):  
   \( \text{RA}(u, v) = \sum_{w \in N(u) \cap N(v)} \frac{1}{\deg(w)} \).

### 4.2 Embedding-based methods

6. **DeepWalk (DW)** – runs uniform random walks on the graph and trains a
   Word2Vec model to obtain node embeddings; link scores are cosine similarity
   between embeddings.

7. **Node2Vec (NV)** – similar to DeepWalk but uses biased (p, q)-controlled
   random walks to interpolate between breadth-first and depth-first
   exploration, often capturing better structural roles.

---

## 5. Adamic–Adar Index (AA)

### 5.1 Description and history

The Adamic–Adar index was proposed by Lada Adamic and Eytan Adar (2003) as a
measure of similarity in social networks and the web graph. It extends the
Common Neighbors heuristic by down‑weighting “popular” nodes that connect to
many users, and emphasizing rare shared connections.

### 5.2 High‑level working

For two users \(u\) and \(v\):

- Find all common friends \(w \in N(u)\cap N(v)\).
- For each common friend \(w\), compute its degree \(\deg(w)\).
- Add a contribution of \(1/\log(\deg(w))\) to the score.
- Sum these contributions to obtain \(\text{AA}(u,v)\).

In our implementation (`aa.py`), we:

- Use NetworkX to obtain neighbors and degrees.
- Iterate over common neighbors and accumulate the weighted sum.
- Use this score to rank non‑friends as candidate recommendations.

### 5.3 Proof of correctness (informal)

Correctness here means “the implementation matches the mathematical
definition”.

- We compute `neighbors_u` and `neighbors_v` exactly as \(N(u)\) and \(N(v)\).
- We intersect them to obtain \(N(u)\cap N(v)\).
- For each \(w\) in this intersection, we compute `degree_w = graph.degree(w)`
   which equals \(\deg(w)\).
- We add `1 / log(degree_w)` to the score, which directly corresponds to the
   theoretical formula.

On small test graphs, we compared our implementation against NetworkX’s
`adamic_adar_index` on the same graph and obtained identical scores,
confirming that the implementation is correct.

### 5.4 Time and space complexity

Let \(n = |V|\) and \(m = |E|\), and let \(\deg(u)\) be the degree of node
\(u\).

- **Single AA score for one pair \((u,v)\)**:
   - We compute neighbors of \(u\) and \(v\) and intersect them.
   - This costs \(O(\deg(u) + \deg(v))\) to build sets and
      \(O(\min(\deg(u), \deg(v)))\) to intersect.
   - **Time**: \(O(\deg(u) + \deg(v))\).
   - **Space**: \(O(\deg(u) + \deg(v))\).

- **`predict_links_for_node`**:
   - For one node, we evaluate all other nodes as candidates.
   - In the worst case this is \(O(n)\) candidates, each costing
      \(O(\deg(u) + \deg(v))\).
   - **Time**: \(O(n \cdot \overline{d})\), where \(\overline{d}\) is
      average degree.
   - **Space**: \(O(n)\) to store scores.

- **`predict_all_links` (all non‑edges)**:
   - Loops over all unordered pairs of nodes ⇒ \(O(n^2)\) pairs.
   - Each AA score is \(O(\overline{d})\).
   - **Time**: \(O(n^2 \cdot \overline{d})\) (expensive on large graphs).
   - **Space**: Up to \(O(n^2)\) scores in the worst case.

#### Experimental runtime vs theoretical

In `analysis.py`, for each algorithm we fix the graph and vary \(k\) (top‑k
recommendations). For sampled nodes, runtime grows roughly linearly with \(k\)
because we compute and sort more scores per node. The plots
`recommendation_performance_analysis.png` show runtime vs \(k\) for AA, CM,
and JC. The AA curve is almost a straight line, consistent with the
theoretical \(O(k)\) behavior when the number of sampled nodes and graph are
fixed.

### 5.5 Deliverables and insights

- Implementation: `3.4frs/aa.py` (scoring, recommendation, evaluation, and
   explainability helpers).
- AA achieves the **highest precision, recall, and F1‑score** across all
   tested \(k\) values.
- It identifies similar top candidates as Common Neighbors but ranks them more
   intelligently by emphasizing rare mutual friends.

### 5.6 Strengths, weaknesses, ideal use‑cases

- **Strengths**
   - Outperforms CN and JC in F1‑score.
   - Robust against “hub” nodes with very high degree.
   - Simple to implement and explain mathematically.

- **Weaknesses**
   - Slightly more expensive than Common Neighbors.
   - Still a local heuristic; does not consider long paths or features.

- **Ideal use‑cases**
   - Social networks where some users are extremely popular (celebrity effect).
   - When you want better ranking quality than CN but still a simple algorithm.

### 5.7 Plots and comparisons

AA is included in the following plots generated by `analysis.py`:

- Precision vs \(k\), Recall vs \(k\), F1 vs \(k\), Runtime vs \(k\)
   (`recommendation_performance_analysis.png`).
- Bar charts at a fixed \(k\) (we use \(k=10\) in the final experiments)
   comparing all seven algorithms
   (`algorithm_comparison.png`).

Across these visualizations, AA consistently outperforms the other
heuristic methods (CM, JC, PA, RA) on F1‑score while staying competitive in
runtime.

---

## 6. Common Neighbors (CM)

### 6.1 Description and history

Common Neighbors is one of the oldest and simplest link prediction heuristics.
It assumes that two users are more likely to be friends if they share many
mutual friends.

### 6.2 High‑level working

For two users \(u\) and \(v\):

- Compute \(N(u)\) and \(N(v)\), the sets of neighbors.
- Count \(|N(u) \cap N(v)|\), the number of mutual friends.
- Use this count as the score: higher count ⇒ stronger recommendation.

In `cm.py`, we:

- Build neighbor sets using NetworkX.
- Intersect them to count common neighbors.
- Rank non‑neighbors of a node by this score to generate recommendations.

### 6.3 Proof of correctness (informal)

The implementation directly computes \(N(u) \cap N(v)\) via set intersection
and returns its cardinality. This exactly matches the theoretical definition
of the Common Neighbors score, so the implementation is correct.

### 6.4 Time and space complexity

- **Single CN score**: \(O(\deg(u) + \deg(v))\) time, \(O(\deg(u)+\deg(v))\)
   space.
- **`predict_links_for_node`**: \(O(n \cdot \overline{d})\) time,
   \(O(n)\) space.
- **`predict_all_links`**: \(O(n^2 \cdot \overline{d})\) time,
   \(O(n^2)\) space.

Experimentally, the runtime vs \(k\) curve for CM is almost linear and lies
slightly below AA, confirming the theoretical expectation that CM is the
fastest of the three.

### 6.5 Deliverables and insights

- Implementation: `3.4frs/cm.py`.
- Serves as a strong and very fast baseline.
- Often recommends the same top candidates as AA, but with slightly less
   nuanced ranking.

### 6.6 Strengths, weaknesses, ideal use‑cases

- **Strengths**: simplest to understand, fastest in practice, no parameters.
- **Weaknesses**: treats all common neighbors equally; sensitive to high-degree
   nodes.
- **Ideal use‑cases**: when latency is critical and a simple baseline is
   sufficient.

### 6.7 Plots and comparisons

CM appears in all the same plots as AA. It is consistently slightly faster but
slightly less accurate than AA.

---

## 7. Jaccard Coefficient (JC)

### 7.1 Description and history

The Jaccard Coefficient is a classic similarity measure used in information
retrieval and set similarity. In graphs, it measures the fraction of neighbors
that two nodes share.

### 7.2 High‑level working

For two users \(u\) and \(v\):

- Compute \(N(u)\) and \(N(v)\).
- Compute intersection and union: \(N(u) \cap N(v)\) and \(N(u) \cup N(v)\).
- Score is \(|N(u) \cap N(v)| / |N(u) \cup N(v)|\).

### 7.3 Proof of correctness (informal)

In `jc.py`, we build neighbor sets, compute their intersection and union, and
return the ratio of their sizes. This is exactly the theoretical Jaccard
definition, so the implementation is correct.

### 7.4 Time and space complexity

- **Single JC score**: \(O(\deg(u) + \deg(v))\) time for intersection and
   union; \(O(\deg(u)+\deg(v))\) space.
- Higher-level functions (`predict_links_for_node`, `predict_all_links`) have
   the same asymptotic complexity as CM and AA.

Experimentally, JC is slightly slower than AA and CM because it builds both
intersection and union sets.

### 7.5 Deliverables and insights

- Implementation: `3.4frs/jc.py`.
- JC normalizes by union size, which can severely penalize high-degree nodes
   in sparse networks.

### 7.6 Strengths, weaknesses, ideal use‑cases

- **Strengths**: scale-invariant; accounts for total neighborhood size.
- **Weaknesses**: underperforms on this sparse Facebook graph; scores become
   extremely small and hard to distinguish.
- **Ideal use‑cases**: denser graphs or domains where relative overlap matters
   more than raw counts.

### 7.7 Plots and comparisons

In both the performance vs \(k\) plots and the bar charts at \(k=10\), JC is
dominated by AA, CM, PA, and RA in F1‑score and precision.

---

## 8. Comparative Analysis of Heuristic Algorithms (AA, CM, JC)

### 8.1 Methodology

- **Train–test split**: 80% training edges and 20% test edges.
- **Evaluation**: sampled 200 nodes for efficiency.
- **Metrics**: Precision, Recall, F1‑Score, runtime.
- **k values tested**: 5 and 10.

Recall is calculated **relative to test edges incident to the sampled
nodes** (i.e., edges that are actually discoverable by our evaluation), not
the entire test set.

### 8.2 Performance at k=10 (heuristics)

At \(k=10\), Resource Allocation achieved the highest F1‑score among the
heuristic methods, with Adamic–Adar and Preferential Attachment close behind.
Jaccard consistently lagged in both precision and F1‑score.

### 8.3 Performance across multiple k values

For all five heuristic algorithms we observe the expected pattern:

- Precision decreases when moving from \(k=5\) to \(k=10\) (more
   recommendations ⇒ more false positives).
- Recall increases over the same range (more true edges are recovered).
- F1‑score typically peaks around \(k=10\), which we use as our main
   comparison point.

### 8.4 Runtime analysis (heuristics)

Common Neighbors and Preferential Attachment are the fastest algorithms,
followed closely by Adamic–Adar and Resource Allocation. Jaccard is slightly
slower because it computes both intersection and union sets for each pair.

### 8.5 Precision–recall trade‑off

- **Lower k (k=5)**: higher precision, lower recall (few but very confident
   recommendations).
- **Higher k (k=10)**: lower precision, higher recall (better coverage).

Overall we find \(k=10\) to be a good operating point for comparing
algorithms on this dataset.

---

## 9. Embedding-based Methods: DeepWalk and Node2Vec

### 9.1 How they work (high level)

- **DeepWalk**:
   - Perform many uniform random walks on the graph.
   - Treat each walk as a sentence and each node as a word.
   - Train a skip-gram Word2Vec model to learn an embedding for each node.
   - Score potential links by cosine similarity between node embeddings.

- **Node2Vec**:
   - Uses the same Word2Vec training, but changes how walks are generated.
   - Biased random walks controlled by parameters \(p\) and \(q\):
      - Returning to the previous node is weighted by \(1/p\).
      - Staying close in the graph vs exploring far is controlled by \(q\).
   - This allows interpolation between BFS-like (community) and DFS-like
      (structural) behavior.

### 9.2 Implementation in this project

- `dw.py`:
   - `generate_random_walks` – uniform walks for DeepWalk.
   - `train_deepwalk_embeddings` – trains Word2Vec and returns `node -> vector`.
   - `predict_links_for_node` – uses cosine similarity to rank non-neighbors.

- `nv.py`:
   - `_node2vec_next_step` – Node2Vec transition rule using \(p, q\).
   - `generate_node2vec_walks` and `train_node2vec_embeddings` – biased walks
      and training.
   - `predict_links_for_node` – same interface as DeepWalk but using
      Node2Vec embeddings.

In `analysis.py`, we train embeddings **once** on the train graph and then
wrap the predictors so they can be evaluated via the same
`evaluate_algorithm_detailed` sampling framework used for the heuristics.

### 9.3 Qualitative expectations

- DeepWalk and Node2Vec can capture longer-range structural patterns, not just
   immediate neighbors.
- On sparse graphs with community structure (like social networks), they
   usually:
   - Achieve competitive or better recall than simple heuristics.
   - Have higher runtime due to embedding training.

In practice, for this project we mostly focus on using them as a
demonstration of modern embedding-based link prediction rather than fully
optimizing hyperparameters.

---

## 10. Steps to Reproduce

1. Ensure Python 3.12 and required libraries are installed:
    - `networkx`
    - `numpy`
    - `pandas`
    - `matplotlib`

2. From the repository root, run the friend recommendation analysis:

    ```bash
    cd 3.4frs
    python analysis.py
    ```

3. The script will:
      - Load the combined graph from `../dataset` using `graph.py`.
      - Perform an 80–20 train–test edge split.
      - Evaluate AA, CM, JC, PA, RA, DeepWalk, and Node2Vec at multiple \(k\)
         values.
      - Write plots into `3.4frs/results/`:
       - `recommendation_performance_analysis.png`
       - `algorithm_comparison.png`

---

## 11. Scalability and Comparison with Centrality Analysis

- **Full graph evaluation** (4,039 nodes) would be expensive if run naïvely on
   all node pairs; we therefore sample nodes when computing metrics.
- The dataset and graph construction are shared with the centrality analysis
   in `3.1.2ca`, but the task here is link prediction instead of node
   importance.

---

## 12. Citations

- L. A. Adamic and E. Adar, “Friends and neighbors on the Web,” *Social
   Networks*, vol. 25, no. 3, pp. 211–230, 2003.
- M. E. J. Newman, *Networks: An Introduction*, Oxford University Press, 2010.
- SNAP Facebook ego-network dataset: J. McAuley and J. Leskovec, “Learning to
   Discover Social Circles in Ego Networks,” *Advances in Neural Information
   Processing Systems (NIPS)*, 2012.
