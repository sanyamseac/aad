"""Resource Allocation Friend Recommendation Algorithm.

This module implements the Resource Allocation index for link prediction.
It models resource transfer through common neighbors, weighting by 1/degree.
Empirical studies show this often outperforms other heuristics.

Time Complexity: O(n * d²) where n=nodes, d=average degree
Space Complexity: O(n)
"""

import os
import sys
import networkx as nx
import random
import questionary

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

def compute_resource_allocation_score(G, u, v):
    """Calculate Resource Allocation score between two nodes.
    
    Sum of 1/degree(w) over all common neighbors w.
    
    Args:
        G (networkx.Graph): The graph structure.
        u (int): First node ID.
        v (int): Second node ID.
        
    Returns:
        float: Resource allocation score.
    """
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    common_neighbors = neighbors_u.intersection(neighbors_v)
    
    score = 0.0
    for w in common_neighbors:
        degree_w = G.degree(w) # Degree centrality implemented in 3.1.2ca/dc
        if degree_w > 0:
            score += 1.0 / degree_w
    
    return score

def recommend_friends(G, node, top_k=10):
    """Recommend friends for a node using Resource Allocation.
    
    Args:
        G (networkx.Graph): The graph structure.
        node (int): Target node for recommendations.
        top_k (int, optional): Number of recommendations to return. Defaults to 10.
        
    Returns:
        list: List of (node_id, score) tuples sorted by score descending.
    """
    if node not in G:
        return []
    
    scores = []
    neighbors = set(G.neighbors(node))
    
    for candidate in G.nodes():
        if candidate != node and candidate not in neighbors:
            score = compute_resource_allocation_score(G, node, candidate)
            if score > 0:
                scores.append((candidate, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

if __name__ == "__main__":
    nx_G, _, _, _ = create_complete_graph(1)
    
    choice = questionary.select(
        "How would you like to select nodes for recommendations?",
        choices=[
            "Use randomly selected sample nodes",
            "Enter a specific node ID"
        ]
    ).ask()
    
    if choice == "Use randomly selected sample nodes":
        sample_nodes = random.sample(list(nx_G.nodes()), min(3, nx_G.number_of_nodes()))
        print("\nSample Friend Recommendations:")
        
        for node in sample_nodes:
            recommendations = recommend_friends(nx_G, node, top_k=5)
            print(f"\nTop 5 friend recommendations for node {node}:")
            if recommendations:
                for rec_node, score in recommendations:
                    print(f"  → Node {rec_node}: RA score = {score:.4f}")
            else:
                print(f"  No recommendations available for node {node}")
            
    elif choice == "Enter a specific node ID":
        node_input = questionary.text(f"Enter node ID:").ask()
        
        try:
            node = int(node_input)
            if node in nx_G.nodes():
                print("\nFriend Recommendations:")
                recommendations = recommend_friends(nx_G, node, top_k=5)
                print(f"\nTop 5 friend recommendations for node {node}:")
                if recommendations:
                    for rec_node, score in recommendations:
                        print(f"  → Node {rec_node}: RA score = {score:.4f}")
                else:
                    print(f"  No recommendations available for node {node}")
            else:
                print(f"Error: Node {node} does not exist in the graph.")
        except ValueError:
            print("Error: Please enter a valid integer node ID.")
    
    else:
        print("No valid choice made. Exiting.")
        exit(1)