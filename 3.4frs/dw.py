"""
DeepWalk-style random-walk embeddings for friend recommendation.

This module implements a very small DeepWalk-like pipeline using
NetworkX for random walks and gensim's Word2Vec to learn embeddings.
For consistency with the heuristic methods (CN/JC/AA/PA/RA), we
provide functions to:

- generate random walks
- train node embeddings
- compute link scores via cosine similarity (Vectorized for performance)
- predict links for a node
- evaluate on a train/test split

Note: This is a lightweight educational implementation.
"""

import os
import sys
import time
from typing import List, Tuple, Dict, Set

import networkx as nx
import numpy as np
from gensim.models import Word2Vec

# Import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graph import create_complete_graph


def generate_random_walks(
    graph: nx.Graph,
    num_walks: int = 10,
    walk_length: int = 40,
    seed: int = 42,
) -> List[List[str]]:
    """Generate random walks over the graph.

    Nodes are converted to strings because gensim's Word2Vec expects
    sequences of string tokens.
    """
    rng = np.random.RandomState(seed)
    walks: List[List[str]] = []
    nodes = list(graph.nodes())

    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            current = start
            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(current))
                if not neighbors:
                    break
                current = rng.choice(neighbors)
                walk.append(current)
            walks.append([str(n) for n in walk])

    return walks


def train_deepwalk_embeddings(
    graph: nx.Graph,
    embedding_dim: int = 64,
    num_walks: int = 10,
    walk_length: int = 40,
    window_size: int = 5,
    workers: int = 1,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Train DeepWalk-style embeddings and return a mapping node -> vector."""
    
    # Generate walks (The algorithm's core "structure learning" phase)
    walks = generate_random_walks(graph, num_walks=num_walks, walk_length=walk_length, seed=seed)

    # Train Word2Vec (The "representation learning" phase)
    model = Word2Vec(
        sentences=walks,
        vector_size=embedding_dim,
        window=window_size,
        min_count=0,
        sg=1,  # skip-gram
        workers=workers,
        seed=seed,
        epochs=5,
    )

    embeddings: Dict[int, np.ndarray] = {}
    for node in graph.nodes():
        # Ensure vector is normalized for faster cosine similarity later
        vec = model.wv[str(node)]
        norm = np.linalg.norm(vec)
        embeddings[node] = vec / norm if norm > 0 else vec
    
    return embeddings


def predict_links_for_node(
    graph: nx.Graph,
    embeddings: Dict[int, np.ndarray],
    node: int,
    k: int = 10,
) -> List[Tuple[int, float]]:
    """Predict top-k friend recommendations using vectorized cosine similarity.
    
    Optimized: Uses numpy matrix operations instead of a Python loop.
    """
    if node not in graph or node not in embeddings:
        return []

    # 1. Identify candidates (nodes not already connected)
    neighbors = set(graph.neighbors(node))
    candidates = [n for n in graph.nodes() if n != node and n not in neighbors]
    
    if not candidates:
        return []

    # 2. Prepare vectors
    # Target node vector (1, D)
    target_vec = embeddings[node].reshape(1, -1)
    
    # Candidate vectors (N_candidates, D)
    # Note: embeddings are already pre-normalized in train_deepwalk_embeddings
    candidate_vecs = np.array([embeddings[n] for n in candidates])
    
    # 3. Compute Cosine Similarity (Vectorized)
    # Since vectors are normalized, Cosine Similarity = Dot Product
    scores_array = np.dot(candidate_vecs, target_vec.T).flatten()
    
    # 4. Pair candidates with scores
    scores = list(zip(candidates, scores_array))

    # 5. Sort and return top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def evaluate_algorithm(
    train_graph: nx.Graph,
    test_edges: List[Tuple[int, int]],
    embedding_dim: int = 64,
    num_walks: int = 10,
    walk_length: int = 40,
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate DeepWalk-based link prediction on a train/test split.
    Uses sampling for large graphs to ensure responsiveness.
    """
    start_time = time.time()

    # Train embeddings
    embeddings = train_deepwalk_embeddings(
        train_graph,
        embedding_dim=embedding_dim,
        num_walks=num_walks,
        walk_length=walk_length,
    )

    # Sample nodes for evaluation (Efficiency fix)
    # Evaluating N^2 pairs is too slow; we sample 100 nodes.
    all_nodes = list(train_graph.nodes())
    np.random.seed(42)
    sample_size = min(100, len(all_nodes))
    sample_nodes = np.random.choice(all_nodes, sample_size, replace=False)
    
    hits = 0
    total_predictions_made = 0
    
    # Build test set for fast lookup
    test_edges_set = set()
    relevant_test_edges_count = 0
    for u, v in test_edges:
        edge = tuple(sorted((u, v)))
        test_edges_set.add(edge)
        # Only count edges incident to our sample for accurate Recall calculation
        if u in sample_nodes or v in sample_nodes:
            relevant_test_edges_count += 1

    # Predict
    for node in sample_nodes:
        preds = predict_links_for_node(train_graph, embeddings, node, k=k)
        for candidate, _ in preds:
            edge = tuple(sorted((node, candidate)))
            if edge in test_edges_set:
                hits += 1
        total_predictions_made += len(preds)

    # Metrics
    precision = hits / total_predictions_made if total_predictions_made > 0 else 0.0
    # Recall relative to discoverable edges (Methodological fix)
    recall = hits / relevant_test_edges_count if relevant_test_edges_count > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    runtime = time.time() - start_time

    return {
        "algorithm": "DeepWalk",
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "runtime": runtime,
        "total_predictions": total_predictions_made,
        "true_positives": hits,
        "k": k,
    }


def demo() -> None:
    """Small demo showing DeepWalk recommendations for one node."""
    print("=" * 70)
    print("DeepWalk-based Friend Recommendation - Demo")
    print("=" * 70)

    print("\nLoading graph from dataset...")
    dataset_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dataset")
    )
    G, _, _, _ = create_complete_graph(dataset_path=dataset_path)

    print(f"\nDataset: Facebook Ego Network")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    print("\nTraining embeddings (this may take a moment)...")
    embeddings = train_deepwalk_embeddings(
        G, embedding_dim=32, num_walks=5, walk_length=20
    )

    node = 0
    recommendations = predict_links_for_node(G, embeddings, node, k=5)

    print(f"\n{'='*70}")
    print(f"Top 5 friend recommendations for user {node} (DeepWalk):")
    print(f"{'='*70}")
    for i, (candidate, score) in enumerate(recommendations, 1):
        print(f"{i}. User {candidate}: cosine similarity={score:.4f}")

if __name__ == "__main__":
    demo()