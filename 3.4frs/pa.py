"""
Preferential Attachment Algorithm for Friend Recommendation
Score is the product of node degrees: PA(u, v) = deg(u) * deg(v).

Author: Team aad.js
Course: Algorithm Analysis & Design
Implementation: From scratch (no external algorithm libraries)
"""

import os
import sys
import time
from typing import List, Tuple, Dict, Set

import networkx as nx
import numpy as np

# Import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graph import create_complete_graph


def compute_preferential_attachment_score(graph: nx.Graph, u: int, v: int) -> int:
	"""Compute Preferential Attachment score between two nodes.

	PA(u, v) = deg(u) * deg(v)

	Time Complexity: O(1)
	Space Complexity: O(1)
	"""

	return graph.degree(u) * graph.degree(v)


def predict_links_for_node(graph: nx.Graph, node: int, k: int = 10) -> List[Tuple[int, int]]:
	"""Predict top-k friend recommendations for a given node using
	Preferential Attachment.

	Follows the same interface as cm/jc/aa: returns a list of
	(candidate_node, score) pairs sorted by score.
	"""

	if node not in graph:
		return []

	scores: List[Tuple[int, int]] = []
	neighbors = set(graph.neighbors(node))

	for candidate in graph.nodes():
		if candidate != node and candidate not in neighbors:
			score = compute_preferential_attachment_score(graph, node, candidate)
			if score > 0:
				scores.append((candidate, score))

	scores.sort(key=lambda x: x[1], reverse=True)
	return scores[:k]


def predict_all_links(graph: nx.Graph, existing_edges: Set[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
	"""Predict scores for all possible non-existing edges in the graph.

	Time Complexity: O(|V|^2)
	Space Complexity: O(|V|^2)
	"""

	predictions: List[Tuple[int, int, int]] = []
	nodes = list(graph.nodes())

	for i, u in enumerate(nodes):
		for v in nodes[i + 1 :]:
			if (u, v) not in existing_edges and (v, u) not in existing_edges:
				score = compute_preferential_attachment_score(graph, u, v)
				if score > 0:
					predictions.append((u, v, score))

	return predictions


def evaluate_algorithm(train_graph: nx.Graph, test_edges: List[Tuple[int, int]], k: int = 10) -> Dict[str, float]:
	"""Evaluate the Preferential Attachment algorithm on test edges.

	Metrics: precision, recall, F1, runtime, number of predictions, true
	positives. Mirrors the structure of cm/jc/aa evaluate_algorithm.
	"""

	start_time = time.time()

	existing_edges = set(train_graph.edges())
	all_predictions = predict_all_links(train_graph, existing_edges)
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
	f1_score = (
		2 * precision * recall / (precision + recall)
		if (precision + recall) > 0
		else 0.0
	)

	runtime = time.time() - start_time

	return {
		"algorithm": "Preferential Attachment",
		"precision": precision,
		"recall": recall,
		"f1_score": f1_score,
		"runtime": runtime,
		"total_predictions": len(all_predictions),
		"true_positives": true_positives,
		"k": k,
	}


def calculate_metrics_at_multiple_k(
	train_graph: nx.Graph,
	test_edges: List[Tuple[int, int]],
	k_values: List[int] = [5, 10, 20, 50, 100],
) -> List[Dict[str, float]]:
	"""Evaluate the algorithm at multiple k values to understand
	performance trends."""

	results: List[Dict[str, float]] = []
	for k in k_values:
		results.append(evaluate_algorithm(train_graph, test_edges, k))
	return results


def demo() -> None:
	"""Demo for Preferential Attachment using the Facebook ego dataset."""

	print("=" * 70)
	print("Preferential Attachment Algorithm - Demo")
	print("=" * 70)

	print("\nLoading graph from dataset...")
	dataset_path = os.path.abspath(
		os.path.join(os.path.dirname(__file__), "..", "dataset")
	)
	G, all_ego_nodes, all_circles, all_features = create_complete_graph(
		dataset_path=dataset_path
	)

	print(f"\nDataset: Facebook Ego Network")
	print(f"Nodes: {G.number_of_nodes()}")
	print(f"Edges: {G.number_of_edges()}")
	print(f"Ego nodes: {len(all_ego_nodes)}")

	node = 0
	recommendations = predict_links_for_node(G, node, k=5)

	print(f"\n{'='*70}")
	print(f"Top 5 friend recommendations for user {node} (PA):")
	print(f"{'='*70}")
	for i, (candidate, score) in enumerate(recommendations, 1):
		print(f"{i}. User {candidate}: score={score} (deg(u)*deg(v))")

	print(f"\n{'='*70}")
	print("Evaluation: Train-Test Split (80-20)")
	print(f"{'='*70}")

	edges = list(G.edges())
	np.random.seed(42)
	np.random.shuffle(edges)
	split = int(0.8 * len(edges))
	train_edges = edges[:split]
	test_edges = edges[split:]

	print(f"Training edges: {len(train_edges)}")
	print(f"Test edges: {len(test_edges)}")

	train_graph = nx.Graph()
	train_graph.add_nodes_from(G.nodes())
	train_graph.add_edges_from(train_edges)

	metrics = evaluate_algorithm(train_graph, test_edges, k=20)

	print(f"\n{'='*70}")
	print("Performance Metrics (k=20):")
	print(f"{'='*70}")
	print(f"Precision:         {metrics['precision']:.4f}")
	print(f"Recall:            {metrics['recall']:.4f}")
	print(f"F1 Score:          {metrics['f1_score']:.4f}")
	print(f"Runtime:           {metrics['runtime']:.4f} seconds")
	print(f"Total Predictions: {metrics['total_predictions']}")
	print(f"True Positives:    {metrics['true_positives']}")

	print(f"\n{'='*70}")
	print("Performance at Multiple k Values:")
	print(f"{'='*70}")
	k_values = [5, 10, 20, 50]
	results = calculate_metrics_at_multiple_k(train_graph, test_edges, k_values)
	print(f"{'k':>5} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
	print(f"{'-'*5} {'-'*12} {'-'*12} {'-'*12}")
	for result in results:
		print(
			f"{result['k']:>5} {result['precision']:>12.4f} {result['recall']:>12.4f} {result['f1_score']:>12.4f}"
		)


if __name__ == "__main__":
	demo()

