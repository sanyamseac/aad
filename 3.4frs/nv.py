"""Node2Vec-style embeddings for friend recommendation.

This module reuses the DeepWalk pipeline but with biased random walks
controlled by parameters p and q.
"""

import os
import sys
import time
from typing import List, Tuple, Dict, Set

import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graph import create_complete_graph


def _node2vec_next_step(
    rng: np.random.RandomState,
    graph: nx.Graph,
    prev: int,
    current: int,
    p: float,
    q: float,
) -> int:
    """Choose next node for Node2Vec random walk.

    Weights:
    - return to prev: 1/p
    - neighbor of prev: 1
    - otherwise: 1/q
    """

    neighbors = list(graph.neighbors(current))
    if not neighbors:
        return current

    if prev is None:
        return rng.choice(neighbors)

    weights = []
    for dst in neighbors:
        if dst == prev:
            weight = 1.0 / p
        elif graph.has_edge(dst, prev):
            weight = 1.0
        else:
            weight = 1.0 / q
        weights.append(weight)

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return rng.choice(neighbors, p=weights)


def generate_node2vec_walks(
    graph: nx.Graph,
    num_walks: int = 10,
    walk_length: int = 40,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 42,
) -> List[List[str]]:
    """Generate Node2Vec biased random walks over the graph."""

    rng = np.random.RandomState(seed)
    walks: List[List[str]] = []
    nodes = list(graph.nodes())

    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            prev = None
            current = start
            for _ in range(walk_length - 1):
                nxt = _node2vec_next_step(rng, graph, prev, current, p, q)
                if nxt == current:
                    break
                prev, current = current, nxt
                walk.append(current)
            walks.append([str(n) for n in walk])

    return walks


def train_node2vec_embeddings(
    graph: nx.Graph,
    embedding_dim: int = 64,
    num_walks: int = 10,
    walk_length: int = 40,
    p: float = 1.0,
    q: float = 1.0,
    window_size: int = 5,
    workers: int = 1,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Train Node2Vec embeddings and return node -> vector mapping."""

    walks = generate_node2vec_walks(
        graph, num_walks=num_walks, walk_length=walk_length, p=p, q=q, seed=seed
    )

    model = Word2Vec(
        sentences=walks,
        vector_size=embedding_dim,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        seed=seed,
        epochs=5,
    )

    embeddings: Dict[int, np.ndarray] = {}
    for node in graph.nodes():
        embeddings[node] = model.wv[str(node)]
    return embeddings


def compute_embedding_score(embeddings: Dict[int, np.ndarray], u: int, v: int) -> float:
    if u not in embeddings or v not in embeddings:
        return 0.0
    u_vec = embeddings[u].reshape(1, -1)
    v_vec = embeddings[v].reshape(1, -1)
    return float(cosine_similarity(u_vec, v_vec)[0, 0])


def predict_links_for_node(
    graph: nx.Graph,
    embeddings: Dict[int, np.ndarray],
    node: int,
    k: int = 10,
) -> List[Tuple[int, float]]:
    if node not in graph or node not in embeddings:
        return []

    scores: List[Tuple[int, float]] = []
    neighbors = set(graph.neighbors(node))
    for candidate in graph.nodes():
        if candidate != node and candidate not in neighbors:
            score = compute_embedding_score(embeddings, node, candidate)
            scores.append((candidate, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def predict_all_links(
    graph: nx.Graph,
    embeddings: Dict[int, np.ndarray],
    existing_edges: Set[Tuple[int, int]],
) -> List[Tuple[int, int, float]]:
    predictions: List[Tuple[int, int, float]] = []
    nodes = list(graph.nodes())

    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            if (u, v) not in existing_edges and (v, u) not in existing_edges:
                score = compute_embedding_score(embeddings, u, v)
                predictions.append((u, v, score))

    return predictions


def evaluate_algorithm(
    train_graph: nx.Graph,
    test_edges: List[Tuple[int, int]],
    embedding_dim: int = 64,
    num_walks: int = 10,
    walk_length: int = 40,
    p: float = 1.0,
    q: float = 1.0,
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate Node2Vec-based link prediction on a train/test split."""

    start_time = time.time()

    embeddings = train_node2vec_embeddings(
        train_graph,
        embedding_dim=embedding_dim,
        num_walks=num_walks,
        walk_length=walk_length,
        p=p,
        q=q,
    )

    existing_edges = set(train_graph.edges())
    all_predictions = predict_all_links(train_graph, embeddings, existing_edges)
    all_predictions.sort(key=lambda x: x[2], reverse=True)

    test_edges_set: Set[Tuple[int, int]] = set()
    for u, v in test_edges:
        test_edges_set.add((min(u, v), max(u, v)))

    top_k_predictions: Set[Tuple[int, int]] = set()
    for u, v, _ in all_predictions[:k]:
        top_k_predictions.add((min(u, v), max(u, v)))

    true_positives = len(top_k_predictions.intersection(test_edges_set))
    precision = true_positives / k if k > 0 else 0.0
    recall = true_positives / len(test_edges_set) if test_edges_set else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    runtime = time.time() - start_time

    return {
        "algorithm": "Node2Vec",
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "runtime": runtime,
        "total_predictions": len(all_predictions),
        "true_positives": true_positives,
        "k": k,
    }


def demo() -> None:
    print("=" * 70)
    print("Node2Vec-based Friend Recommendation - Demo")
    print("=" * 70)

    print("\nLoading graph from dataset...")
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(dataset_path=dataset_path)

    print(f"\nDataset: Facebook Ego Network")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    embeddings = train_node2vec_embeddings(G, embedding_dim=32, num_walks=5, walk_length=20, p=1.0, q=1.0)

    node = 0
    recommendations = predict_links_for_node(G, embeddings, node, k=5)

    print(f"\n{'='*70}")
    print(f"Top 5 friend recommendations for user {node} (Node2Vec):")
    print(f"{'='*70}")
    for i, (candidate, score) in enumerate(recommendations, 1):
        print(f"{i}. User {candidate}: cosine similarity={score:.4f}")


if __name__ == "__main__":
    demo()
