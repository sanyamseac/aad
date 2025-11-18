"""Node2Vec-style embeddings for friend recommendation."""

import os
import sys
import time
from typing import List, Tuple, Dict

import networkx as nx
import numpy as np
from gensim.models import Word2Vec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graph import create_complete_graph


def _node2vec_next_step(rng, graph, prev, current, p, q):
    """Choose next node for Node2Vec random walk (biased)."""
    neighbors = list(graph.neighbors(current))
    if not neighbors:
        return current

    if prev is None:
        return rng.choice(neighbors)

    weights = []
    for dst in neighbors:
        if dst == prev:
            weights.append(1.0 / p)
        elif graph.has_edge(dst, prev):
            weights.append(1.0)
        else:
            weights.append(1.0 / q)

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return rng.choice(neighbors, p=weights)


def generate_node2vec_walks(graph, num_walks=10, walk_length=40, p=1.0, q=1.0, seed=42):
    """Generate Node2Vec biased random walks."""
    rng = np.random.RandomState(seed)
    walks = []
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


def train_node2vec_embeddings(graph, embedding_dim=64, num_walks=10, walk_length=40, p=1.0, q=1.0, window_size=5, workers=1, seed=42):
    """Train Node2Vec embeddings and return node -> vector mapping."""
    walks = generate_node2vec_walks(graph, num_walks, walk_length, p, q, seed)
    
    model = Word2Vec(sentences=walks, vector_size=embedding_dim, window=window_size, min_count=0, sg=1, workers=workers, seed=seed, epochs=5)

    embeddings = {}
    for node in graph.nodes():
        # Pre-normalize vectors
        vec = model.wv[str(node)]
        norm = np.linalg.norm(vec)
        embeddings[node] = vec / norm if norm > 0 else vec
    return embeddings


def predict_links_for_node(graph, embeddings, node, k=10):
    """Predict top-k friend recommendations using vectorized cosine similarity."""
    if node not in graph or node not in embeddings:
        return []

    neighbors = set(graph.neighbors(node))
    candidates = [n for n in graph.nodes() if n != node and n not in neighbors]
    
    if not candidates:
        return []

    target_vec = embeddings[node].reshape(1, -1)
    candidate_vecs = np.array([embeddings[n] for n in candidates])
    
    # Vectorized Dot Product (since vectors are normalized)
    scores_array = np.dot(candidate_vecs, target_vec.T).flatten()
    
    scores = list(zip(candidates, scores_array))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def evaluate_algorithm(train_graph, test_edges, embedding_dim=64, num_walks=10, walk_length=40, p=1.0, q=1.0, k=10):
    """Evaluate Node2Vec using sampling."""
    start_time = time.time()
    
    embeddings = train_node2vec_embeddings(train_graph, embedding_dim, num_walks, walk_length, p, q)
    
    # Efficient Sampling Evaluation
    all_nodes = list(train_graph.nodes())
    np.random.seed(42)
    sample_nodes = np.random.choice(all_nodes, min(100, len(all_nodes)), replace=False)
    
    test_edges_set = set(tuple(sorted((u, v))) for u, v in test_edges)
    relevant_test_edges = 0
    for u, v in test_edges:
        if u in sample_nodes or v in sample_nodes:
            relevant_test_edges += 1
            
    hits = 0
    total_preds = 0
    
    for node in sample_nodes:
        preds = predict_links_for_node(train_graph, embeddings, node, k=k)
        for candidate, _ in preds:
            if tuple(sorted((node, candidate))) in test_edges_set:
                hits += 1
        total_preds += len(preds)

    precision = hits / total_preds if total_preds > 0 else 0.0
    recall = hits / relevant_test_edges if relevant_test_edges > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "algorithm": "Node2Vec",
        "precision": precision, 
        "recall": recall, 
        "f1_score": f1, 
        "runtime": time.time() - start_time,
        "true_positives": hits
    }

def demo():
    print("Node2Vec Demo")
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    G, _, _, _ = create_complete_graph(dataset_path=dataset_path)
    
    embeddings = train_node2vec_embeddings(G, embedding_dim=32, num_walks=5, walk_length=20)
    recs = predict_links_for_node(G, embeddings, 0, k=5)
    
    print("\nTop 5 Recommendations for Node 0:")
    for i, (cand, score) in enumerate(recs, 1):
        print(f"{i}. User {cand}: score={score:.4f}")

if __name__ == "__main__":
    demo()