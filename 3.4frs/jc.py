"""
Jaccard Coefficient Algorithm for Friend Recommendation
Normalized version of Common Neighbors that accounts for total number of neighbors.
Formula: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

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

# Import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph


def compute_jaccard_coefficient(graph: nx.Graph, u: int, v: int) -> float:
    """
    Compute Jaccard Coefficient between two nodes.
    
    Time Complexity: O(deg(u) + deg(v))
    Space Complexity: O(deg(u) + deg(v))
    
    Args:
        graph (nx.Graph): The social network graph
        u (int): First node
        v (int): Second node
        
    Returns:
        float: Jaccard coefficient (value between 0 and 1)
    """
    neighbors_u = set(graph.neighbors(u))
    neighbors_v = set(graph.neighbors(v))
    
    intersection = neighbors_u.intersection(neighbors_v)
    union = neighbors_u.union(neighbors_v)
    
    # Avoid division by zero
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)


def predict_links_for_node(graph: nx.Graph, node: int, k: int = 10) -> List[Tuple[int, float]]:
    """
    Predict top-k friend recommendations for a given node using Jaccard Coefficient.
    
    Time Complexity: O(|V| * avg_degree)
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
            score = compute_jaccard_coefficient(graph, node, candidate)
            if score > 0:  # Only consider if there is overlap
                scores.append((candidate, score))
    
    # Sort by score (descending) and return top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def predict_all_links(graph: nx.Graph, existing_edges: Set[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
    """
    Predict scores for all possible non-existing edges in the graph.
    
    Time Complexity: O(|V|^2 * avg_degree)
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
                score = compute_jaccard_coefficient(graph, u, v)
                if score > 0:
                    predictions.append((u, v, score))
    
    return predictions


def evaluate_algorithm(train_graph: nx.Graph, test_edges: List[Tuple[int, int]], k: int = 10) -> Dict[str, float]:
    """
    Evaluate the Jaccard Coefficient algorithm on test edges.
    
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
        'algorithm': 'Jaccard Coefficient',
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'runtime': runtime,
        'total_predictions': len(all_predictions),
        'true_positives': true_positives,
        'avg_score': avg_score,
        'k': k
    }


def get_jaccard_explanation(graph: nx.Graph, u: int, v: int) -> Dict[str, any]:
    """
    Provide detailed explanation for Jaccard coefficient score.
    Useful for understanding why certain recommendations were made.
    
    Args:
        graph (nx.Graph): The social network graph
        u (int): First node
        v (int): Second node
        
    Returns:
        Dict[str, any]: Dictionary with explanation details
    """
    neighbors_u = set(graph.neighbors(u))
    neighbors_v = set(graph.neighbors(v))
    common = neighbors_u.intersection(neighbors_v)
    union = neighbors_u.union(neighbors_v)
    
    return {
        'user_u_neighbors': len(neighbors_u),
        'user_v_neighbors': len(neighbors_v),
        'common_neighbors': len(common),
        'total_unique_neighbors': len(union),
        'jaccard_score': len(common) / len(union) if len(union) > 0 else 0.0,
        'common_neighbor_ids': sorted(list(common))
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
    Demonstration of Jaccard Coefficient algorithm using Facebook ego network dataset.
    Shows basic usage, evaluation metrics, and explainability features.
    """
    print("="*70)
    print("Jaccard Coefficient Algorithm - Demo")
    print("="*70)
    
    # Load the complete graph using the function from 'graph.py'
    print("\nLoading graph from dataset...")
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(dataset_path=dataset_path)
    
    print(f"\nDataset: Facebook Ego Network")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Ego nodes: {len(all_ego_nodes)}")
    
    # Get recommendations for a specific user
    node = 0
    recommendations = predict_links_for_node(G, node, k=5)
    
    print(f"\n{'='*70}")
    print(f"Top 5 friend recommendations for user {node}:")
    print(f"{'='*70}")
    for i, (candidate, score) in enumerate(recommendations, 1):
        explanation = get_jaccard_explanation(G, node, candidate)
        print(f"{i}. User {candidate}: Jaccard = {score:.4f}")
        print(f"   Common: {explanation['common_neighbors']}, "
              f"Total unique: {explanation['total_unique_neighbors']}")
        print(f"   Common neighbors: {explanation['common_neighbor_ids']}")
    
    # Evaluate on train-test split
    print(f"\n{'='*70}")
    print("Evaluation: Train-Test Split (80-20)")
    print(f"{'='*70}")
    
    edges = list(G.edges())
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(edges)
    split = int(0.8 * len(edges))
    train_edges = edges[:split]
    test_edges = edges[split:]
    
    print(f"Training edges: {len(train_edges)}")
    print(f"Test edges: {len(test_edges)}")
    
    # Create train graph
    train_graph = nx.Graph()
    train_graph.add_nodes_from(G.nodes())
    train_graph.add_edges_from(train_edges)
    
    # Evaluate at k=20
    metrics = evaluate_algorithm(train_graph, test_edges, k=20)
    
    print(f"\n{'='*70}")
    print("Performance Metrics (k=20):")
    print(f"{'='*70}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1 Score:          {metrics['f1_score']:.4f}")
    print(f"Average Score:     {metrics['avg_score']:.4f}")
    print(f"Runtime:           {metrics['runtime']:.4f} seconds")
    print(f"Total Predictions: {metrics['total_predictions']}")
    print(f"True Positives:    {metrics['true_positives']}")
    
    # Evaluate at multiple k values
    print(f"\n{'='*70}")
    print("Performance at Multiple k Values:")
    print(f"{'='*70}")
    k_values = [5, 10, 20, 50]
    results = calculate_metrics_at_multiple_k(train_graph, test_edges, k_values)
    
    print(f"{'k':>5} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print(f"{'-'*5} {'-'*12} {'-'*12} {'-'*12}")
    for result in results:
        print(f"{result['k']:>5} {result['precision']:>12.4f} {result['recall']:>12.4f} {result['f1_score']:>12.4f}")


if __name__ == "__main__":
    demo()