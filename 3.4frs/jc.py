"""
Jaccard Coefficient Algorithm for Friend Recommendation
Normalized version of Common Neighbors that accounts for total number of neighbors.
Formula: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Dict
import time


class JaccardCoefficient:
    def __init__(self, graph: nx.Graph):
        """
        Initialize Jaccard Coefficient recommender.
        
        Args:
            graph: NetworkX graph representing the social network
        """
        self.graph = graph
        self.predictions = []
    
    def compute_score(self, u: int, v: int) -> float:
        """
        Compute Jaccard Coefficient between two nodes.
        
        Args:
            u: First node
            v: Second node
            
        Returns:
            Jaccard coefficient (0 to 1)
        """
        neighbors_u = set(self.graph.neighbors(u))
        neighbors_v = set(self.graph.neighbors(v))
        
        intersection = neighbors_u.intersection(neighbors_v)
        union = neighbors_u.union(neighbors_v)
        
        # Avoid division by zero
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def predict_links(self, node: int, k: int = 10) -> List[Tuple[int, float]]:
        """
        Predict top-k friend recommendations for a given node.
        
        Args:
            node: Node to generate recommendations for
            k: Number of recommendations to return
            
        Returns:
            List of (node_id, score) tuples sorted by score
        """
        if node not in self.graph:
            return []
        
        scores = []
        neighbors = set(self.graph.neighbors(node))
        
        # Iterate through all nodes not directly connected
        for candidate in self.graph.nodes():
            if candidate != node and candidate not in neighbors:
                score = self.compute_score(node, candidate)
                if score > 0:  # Only consider if there is overlap
                    scores.append((candidate, score))
        
        # Sort by score (descending) and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def predict_all_links(self, existing_edges: set) -> List[Tuple[int, int, float]]:
        """
        Predict scores for all possible non-existing edges.
        
        Args:
            existing_edges: Set of existing edges (as tuples)
            
        Returns:
            List of (u, v, score) tuples for all non-existing edges
        """
        predictions = []
        nodes = list(self.graph.nodes())
        
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                # Skip if edge already exists
                if (u, v) not in existing_edges and (v, u) not in existing_edges:
                    score = self.compute_score(u, v)
                    if score > 0:
                        predictions.append((u, v, score))
        
        return predictions
    
    def evaluate(self, train_graph: nx.Graph, test_edges: List[Tuple[int, int]], 
                 k: int = 10) -> Dict[str, float]:
        """
        Evaluate the algorithm on test edges.
        
        Args:
            train_graph: Training graph (with some edges removed)
            test_edges: List of true edges to predict
            k: Number of top recommendations to consider
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.graph = train_graph
        start_time = time.time()
        
        # Get all possible edges to rank
        existing_edges = set(train_graph.edges())
        all_predictions = self.predict_all_links(existing_edges)
        
        # Sort by score
        all_predictions.sort(key=lambda x: x[2], reverse=True)
        
        # Convert test edges to set for fast lookup
        test_edges_set = set()
        for u, v in test_edges:
            test_edges_set.add((min(u, v), max(u, v)))
        
        # Calculate metrics at different k values
        top_k_predictions = set()
        for u, v, _ in all_predictions[:k]:
            top_k_predictions.add((min(u, v), max(u, v)))
        
        true_positives = len(top_k_predictions.intersection(test_edges_set))
        precision = true_positives / k if k > 0 else 0
        recall = true_positives / len(test_edges_set) if len(test_edges_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate ROC-AUC using ranking
        predicted_scores = {(min(u, v), max(u, v)): score for u, v, score in all_predictions}
        
        runtime = time.time() - start_time
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'runtime': runtime,
            'total_predictions': len(all_predictions),
            'true_positives': true_positives,
            'avg_score': np.mean([s for _, _, s in all_predictions]) if all_predictions else 0
        }
    
    def get_explanation(self, u: int, v: int) -> Dict[str, any]:
        """
        Provide explanation for recommendation score.
        
        Args:
            u: First node
            v: Second node
            
        Returns:
            Dictionary with explanation details
        """
        neighbors_u = set(self.graph.neighbors(u))
        neighbors_v = set(self.graph.neighbors(v))
        common = neighbors_u.intersection(neighbors_v)
        union = neighbors_u.union(neighbors_v)
        
        return {
            'user_u_neighbors': len(neighbors_u),
            'user_v_neighbors': len(neighbors_v),
            'common_neighbors': len(common),
            'total_neighbors': len(union),
            'jaccard_score': len(common) / len(union) if len(union) > 0 else 0,
            'common_neighbor_ids': list(common)
        }


def demo():
    """Demonstration of Jaccard Coefficient algorithm."""
    # Create sample graph
    G = nx.karate_club_graph()
    
    # Initialize recommender
    jc = JaccardCoefficient(G)
    
    # Get recommendations for a specific user
    node = 0
    recommendations = jc.predict_links(node, k=5)
    
    print(f"Top 5 friend recommendations for user {node}:")
    for candidate, score in recommendations:
        explanation = jc.get_explanation(node, candidate)
        print(f"  User {candidate}: Jaccard = {score:.4f}")
        print(f"    (Common: {explanation['common_neighbors']}, " 
              f"Total: {explanation['total_neighbors']})")
    
    # Evaluate on train-test split
    edges = list(G.edges())
    np.random.shuffle(edges)
    split = int(0.8 * len(edges))
    train_edges = edges[:split]
    test_edges = edges[split:]
    
    # Create train graph
    train_graph = nx.Graph()
    train_graph.add_nodes_from(G.nodes())
    train_graph.add_edges_from(train_edges)
    
    # Evaluate
    metrics = jc.evaluate(train_graph, test_edges, k=20)
    print(f"\nEvaluation Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Average Score: {metrics['avg_score']:.4f}")
    print(f"  Runtime: {metrics['runtime']:.4f} seconds")


if __name__ == "__main__":
    demo()