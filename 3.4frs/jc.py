"""Jaccard Coefficient Friend Recommendation Algorithm.

This module implements the Jaccard Coefficient for link prediction.
It normalizes the common neighbors by the union of neighborhoods,
giving a similarity score between 0 and 1.

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

def compute_jaccard_coefficient(G, u, v):
    """Calculate Jaccard coefficient between two nodes.
    
    Jaccard coefficient = |intersection| / |union| of neighbor sets.
    
    Args:
        G (networkx.Graph): The graph structure.
        u (int): First node ID.
        v (int): Second node ID.
        
    Returns:
        float: Jaccard coefficient in range [0, 1].
    """
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    
    intersection = neighbors_u.intersection(neighbors_v)
    union = neighbors_u.union(neighbors_v)
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)

def recommend_friends(G, node, top_k=10):
    """Recommend friends for a node using Jaccard Coefficient.
    
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
            score = compute_jaccard_coefficient(G, node, candidate)
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
                    print(f"  → Node {rec_node}: Jaccard score = {score:.4f}")
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
                        print(f"  → Node {rec_node}: Jaccard score = {score:.4f}")
                else:
                    print(f"  No recommendations available for node {node}")
            else:
                print(f"Error: Node {node} does not exist in the graph.")
        except ValueError:
            print("Error: Please enter a valid integer node ID.")
    
    else:
        print("No valid choice made. Exiting.")
        exit(1)
