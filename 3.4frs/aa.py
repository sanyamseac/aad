"""
Adamic-Adar Index Algorithm for Friend Recommendation
Weighted version of Common Neighbors that gives more weight to rare common neighbors.
Formula: Σ(w∈N(u)∩N(v)) 1/log(|N(w)|)
Common neighbors with fewer connections are weighted more highly.

Author: Team aad.js
Course: Algorithm Analysis & Design
Implementation: From scratch (no external algorithm libraries)
"""

import os
import sys
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Set
import time
import math

# Import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph


def compute_adamic_adar_score(graph: nx.Graph, u: int, v: int) -> float:
    """
    Compute Adamic-Adar Index between two nodes.
    
    The Adamic-Adar index weights common neighbors by the inverse logarithm
    of their degree, giving more importance to rare shared connections.
    
    Time Complexity: O(min(deg(u), deg(v)))
    Space Complexity: O(deg(u) + deg(v))
    
    Args:
        graph (nx.Graph): The social network graph
        u (int): First node
        v (int): Second node
        
    Returns:
        float: Adamic-Adar index score
    """
    neighbors_u = set(graph.neighbors(u))
    neighbors_v = set(graph.neighbors(v))
    common_neighbors = neighbors_u.intersection(neighbors_v)
    
    score = 0.0
    for w in common_neighbors:
        degree_w = graph.degree(w)
        if degree_w > 1:  # Avoid log(1) = 0
            score += 1.0 / math.log(degree_w)
    
    return score


def predict_links_for_node(graph: nx.Graph, node: int, k: int = 10) -> List[Tuple[int, float]]:
    """
    Predict top-k friend recommendations for a given node using Adamic-Adar Index.
    
    Time Complexity: O(|V| * avg_degree * log(avg_degree))
    Space Complexity: O(|V|)
    
    Args:
        graph (nx.Graph): The social network graph
        node (int): Node to generate recommendations for
        k (int): Number of recommendations to return (default: 10)
        
    Returns:
        List[Tuple[int, float]]: List of (candidate_node, score) tuples sorted by score
    """
    if node not in graph:
        return []
    
    scores = []
    neighbors = set(graph.neighbors(node))
    
    # Iterate through all nodes not directly connected
    for candidate in graph.nodes():
        if candidate != node and candidate not in neighbors:
            score = compute_adamic_adar_score(graph, node, candidate)
            if score > 0:  # Only consider if there are common neighbors
                scores.append((candidate, score))
    
    # Sort by score (descending) and return top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def predict_all_links(graph: nx.Graph, existing_edges: Set[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
    """
    Predict scores for all possible non-existing edges in the graph.
    
    Time Complexity: O(|V|^2 * avg_degree * log(avg_degree))
    Space Complexity: O(|V|^2)
    
    Args:
        graph (nx.Graph): The social network graph
        existing_edges (Set[Tuple[int, int]]): Set of existing edges
        
    Returns:
        List[Tuple[int, int, float]]: List of (u, v, score) tuples for non-existing edges
    """
    predictions = []
    nodes = list(graph.nodes())
    
    for i, u in enumerate(nodes):
        for v in nodes[i+1:]:
            # Skip if edge already exists
            if (u, v) not in existing_edges and (v, u) not in existing_edges:
                score = compute_adamic_adar_score(graph, u, v)
                if score > 0:
                    predictions.append((u, v, score))
    
    return predictions


def evaluate_algorithm(train_graph: nx.Graph, test_edges: List[Tuple[int, int]], k: int = 10) -> Dict[str, float]:
    """
    Evaluate the Adamic-Adar algorithm on test edges.
    
    Metrics computed:
    - Precision: Proportion of correct predictions in top-k
    - Recall: Proportion of actual edges found in top-k
    - F1 Score: Harmonic mean of precision and recall
    - Runtime: Execution time in seconds
    
    Args:
        train_graph (nx.Graph): Training graph (with some edges removed)
        test_edges (List[Tuple[int, int]]): List of true edges to predict
        k (int): Number of top recommendations to consider (default: 10)
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
    """
    start_time = time.time()
    
    # Get all possible edges to rank
    existing_edges = set(train_graph.edges())
    all_predictions = predict_all_links(train_graph, existing_edges)
    
    # Sort by score (descending)
    all_predictions.sort(key=lambda x: x[2], reverse=True)
    
    # Normalize test edges (ensure min, max ordering)
    test_edges_set = set()
    for u, v in test_edges:
        test_edges_set.add((min(u, v), max(u, v)))
    
    # Get top-k predictions
    top_k_predictions = set()
    for u, v, _ in all_predictions[:k]:
        top_k_predictions.add((min(u, v), max(u, v)))
    
    # Calculate metrics
    true_positives = len(top_k_predictions.intersection(test_edges_set))
    precision = true_positives / k if k > 0 else 0.0
    recall = true_positives / len(test_edges_set) if len(test_edges_set) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate average score
    avg_score = np.mean([s for _, _, s in all_predictions]) if all_predictions else 0.0
    
    runtime = time.time() - start_time
    
    return {
        'algorithm': 'Adamic-Adar',
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'runtime': runtime,
        'total_predictions': len(all_predictions),
        'true_positives': true_positives,
        'avg_score': avg_score,
        'k': k
    }


def get_adamic_adar_explanation(graph: nx.Graph, u: int, v: int) -> Dict[str, any]:
    """
    Provide detailed explanation for Adamic-Adar recommendation score.
    Shows which common neighbors contribute most to the score.
    
    Args:
        graph (nx.Graph): The social network graph
        u (int): First node
        v (int): Second node
        
    Returns:
        Dict[str, any]: Dictionary with explanation details including per-neighbor weights
    """
    neighbors_u = set(graph.neighbors(u))
    neighbors_v = set(graph.neighbors(v))
    common = neighbors_u.intersection(neighbors_v)
    
    common_details = []
    total_score = 0.0
    
    for w in common:
        degree = graph.degree(w)
        weight = 1.0 / math.log(degree) if degree > 1 else 0.0
        total_score += weight
        common_details.append({
            'neighbor_id': w,
            'degree': degree,
            'weight': weight
        })
    
    # Sort by weight (rarer neighbors first - higher contribution)
    common_details.sort(key=lambda x: x['weight'], reverse=True)
    
    return {
        'user_u_neighbors': len(neighbors_u),
        'user_v_neighbors': len(neighbors_v),
        'common_neighbors': len(common),
        'adamic_adar_score': total_score,
        'common_neighbor_details': common_details
    }


def compare_with_common_neighbors(graph: nx.Graph, u: int, v: int) -> Dict[str, float]:
    """
    Compare Adamic-Adar score with simple Common Neighbors count.
    Useful to understand the impact of weighting by rarity.
    
    Args:
        graph (nx.Graph): The social network graph
        u (int): First node
        v (int): Second node
        
    Returns:
        Dict[str, float]: Dictionary comparing both scoring methods
    """
    neighbors_u = set(graph.neighbors(u))
    neighbors_v = set(graph.neighbors(v))
    common = neighbors_u.intersection(neighbors_v)
    
    cn_score = len(common)
    aa_score = compute_adamic_adar_score(graph, u, v)
    
    return {
        'common_neighbors_score': cn_score,
        'adamic_adar_score': aa_score,
        'difference': aa_score - cn_score,
        'relative_boost': (aa_score / cn_score - 1) * 100 if cn_score > 0 else 0.0
    }


def calculate_metrics_at_multiple_k(train_graph: nx.Graph, test_edges: List[Tuple[int, int]], 
                                   k_values: List[int] = [5, 10, 20, 50, 100]) -> List[Dict[str, float]]:
    """
    Evaluate the algorithm at multiple k values to understand performance trends.
    
    Args:
        train_graph (nx.Graph): Training graph
        test_edges (List[Tuple[int, int]]): Test edges
        k_values (List[int]): List of k values to evaluate
        
    Returns:
        List[Dict[str, float]]: List of metric dictionaries for each k
    """
    results = []
    for k in k_values:
        metrics = evaluate_algorithm(train_graph, test_edges, k)
        results.append(metrics)
    return results


def demo():
    """
    Demonstration of Adamic-Adar algorithm using Facebook ego network dataset.
    """
    print("="*70)
    print("Adamic-Adar Index Algorithm - Demo")
    print("="*70)
    
    # Load graph
    print("\nLoading graph from dataset...")
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    G, all_ego_nodes, _, _ = create_complete_graph(dataset_path=dataset_path)
    
    # 1. Single User Recommendation (Fast)
    node = 0
    recommendations = predict_links_for_node(G, node, k=5)
    
    print(f"\n{'='*70}")
    print(f"Top 5 friend recommendations for user {node}:")
    print(f"{'='*70}")
    for i, (candidate, score) in enumerate(recommendations, 1):
        print(f"{i}. User {candidate}: Adamic-Adar = {score:.4f}")
    
    # 2. Evaluation (Optimized with Sampling)
    print(f"\n{'='*70}")
    print("Evaluation: Train-Test Split (80-20)")
    print(f"{'='*70}")
    
    edges = list(G.edges())
    np.random.seed(42)
    np.random.shuffle(edges)
    split = int(0.8 * len(edges))
    train_edges = edges[:split]
    test_edges = edges[split:]
    
    train_graph = nx.Graph()
    train_graph.add_nodes_from(G.nodes())
    train_graph.add_edges_from(train_edges)
    
    print("Evaluating on a sample of 100 nodes (for speed)...")
    
    # MANUAL SAMPLING EVALUATION (Fast)
    sample_size = 100
    all_nodes = list(train_graph.nodes())
    sample_nodes = np.random.choice(all_nodes, min(sample_size, len(all_nodes)), replace=False)
    
    test_edges_set = set(tuple(sorted((u,v))) for u,v in test_edges)
    hits = 0
    total_preds = 0
    k = 20
    
    start_time = time.time()
    for node in sample_nodes:
        preds = predict_links_for_node(train_graph, node, k=k)
        for cand, _ in preds:
            if tuple(sorted((node, cand))) in test_edges_set:
                hits += 1
        total_preds += len(preds)
    runtime = time.time() - start_time
    
    precision = hits / total_preds if total_preds > 0 else 0.0
    
    print(f"\nPerformance Metrics (k={k}):")
    print(f"Precision:         {precision:.4f}")
    print(f"Runtime:           {runtime:.4f} seconds")
    print(f"Total Predictions: {total_preds}")
    print(f"True Positives:    {hits}")


if __name__ == "__main__":
    demo()