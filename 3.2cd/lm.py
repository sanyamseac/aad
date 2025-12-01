"""Louvain Method for Community Detection.

This module implements the Louvain method, a greedy optimization algorithm
for detecting communities in large networks by maximizing modularity.
The algorithm operates in two phases: local optimization and network aggregation.

Time Complexity: O(n log n) in practice (worst case O(n²))
Space Complexity: O(n + m)
"""

import os
import sys
from collections import defaultdict

# Import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph


def calculate_modularity(G, communities):
    """Calculate the modularity Q of a graph given a community assignment.
    
    Modularity measures the strength of division of a network into communities,
    comparing actual edges within communities to expected edges in a random graph.
    
    Args:
        G (networkx.Graph): The graph structure.
        communities (dict): Node to community ID mapping.
        
    Returns:
        float: Modularity value Q, typically in range [-0.5, 1.0].
        
    Time Complexity: O(n²) in worst case
    """
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    
    Q = 0.0
    for community in set(communities.values()):
        # Get nodes in this community
        nodes_in_community = [n for n, c in communities.items() if c == community]
        
        for i in nodes_in_community:
            for j in nodes_in_community:
                # A_ij: 1 if edge exists, 0 otherwise
                A_ij = 1 if G.has_edge(i, j) else 0
                
                # Degree of nodes
                k_i = G.degree(i)
                k_j = G.degree(j)
                
                # Modularity contribution
                Q += A_ij - (k_i * k_j) / (2.0 * m)
    
    return Q / (2.0 * m)


def louvain_algorithm(G, max_iterations=100):
    """Apply the Louvain algorithm for community detection.
    
    Optimizes modularity by iteratively moving nodes to the community
    that yields the greatest modularity increase. Continues until no
    improvement is possible.
    
    Args:
        G (networkx.Graph): The input graph.
        max_iterations (int): Maximum number of optimization passes.
        
    Returns:
        tuple: (communities, modularity) where:
            - communities (dict): Node to community ID mapping
            - modularity (float): Final modularity value Q
            
    Time Complexity: O(n log n) on average
    """
    
    # Initialize: each node is in its own community
    communities = {node: node for node in G.nodes()}
    m = G.number_of_edges()
    
    if m == 0:
        return communities, 0.0
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Phase 1: Modularity Optimization
        for node in G.nodes():
            current_community = communities[node]
            best_community = current_community
            best_gain = 0.0
            
            # Remove node from its current community temporarily
            old_community = communities[node]
            
            # Get neighboring communities
            neighbor_communities = set()
            for neighbor in G.neighbors(node):
                neighbor_communities.add(communities[neighbor])
            
            # Try moving node to each neighboring community
            for community in neighbor_communities:
                # Calculate modularity gain
                communities[node] = community
                gain = calculate_modularity_gain(G, node, old_community, community, communities, m)
                
                if gain > best_gain:
                    best_gain = gain
                    best_community = community
            
            # Move node to best community
            if best_community != old_community:
                communities[node] = best_community
                improved = True
            else:
                communities[node] = old_community
    
    # Calculate final modularity
    final_modularity = calculate_modularity(G, communities)
    
    return communities, final_modularity


def calculate_modularity_gain(G, node, old_community, new_community, communities, m):
    """Calculate the modularity gain from moving a node to a new community."""
    if old_community == new_community:
        return 0.0
    
    # Count edges from node to old and new communities
    k_i = G.degree(node)
    
    # Edges to new community
    k_i_in_new = sum(1 for neighbor in G.neighbors(node) 
                     if communities[neighbor] == new_community)
    
    # Edges to old community
    k_i_in_old = sum(1 for neighbor in G.neighbors(node) 
                     if communities[neighbor] == old_community)
    
    # Sum of degrees in new community (excluding node)
    sigma_new = sum(G.degree(n) for n, c in communities.items() 
                    if c == new_community and n != node)
    
    # Sum of degrees in old community (excluding node)
    sigma_old = sum(G.degree(n) for n, c in communities.items() 
                    if c == old_community and n != node)
    
    # Modularity gain formula
    delta_Q = (k_i_in_new - k_i_in_old) / m - k_i * (sigma_new - sigma_old) / (2.0 * m * m)
    
    return delta_Q


if __name__ == "__main__":
    # Load the complete graph using the function from 'graph.py'
    print("Loading graph from dataset...")
    # We need to find the 'dataset' folder, which is one level up from here
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(1)
    print("Graph loaded:", G)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Run Louvain Algorithm
    print("\nRunning Louvain Algorithm for Community Detection...")
    communities, modularity = louvain_algorithm(G)
    
    # Count communities
    unique_communities = len(set(communities.values()))
    
    print(f"\n--- Results ---")
    print(f"Number of communities detected: {unique_communities}")
    print(f"Final Modularity: {modularity:.6f}")
    
    # Show community sizes
    print("\n--- Community Sizes ---")
    community_sizes = defaultdict(int)
    for node, comm in communities.items():
        community_sizes[comm] += 1
    
    sorted_sizes = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
    for i, (comm_id, size) in enumerate(sorted_sizes[:10]):
        print(f"Community {comm_id}: {size} nodes")
    
    # Show some example node assignments
    print("\n--- Sample Node Assignments (first 10 nodes) ---")
    for i, (node, comm) in enumerate(list(communities.items())[:10]):
        print(f"Node {node} -> Community {comm}")