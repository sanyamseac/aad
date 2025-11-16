import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

def degree_centrality(G, normalized=True):
    """Computes the degree centrality for all nodes, normalized by (n-1)."""
    
    centrality = {} # centrality dictionary to store results
    n = G.number_of_nodes()

    if n <= 1:
        return {node: 0.0 for node in G.nodes()}

    # Calculate the normalization factor
    if normalized:
        # The score is normalized by n-1, the maximum possible degree
        scale = 1.0 / (n - 1)
    else:
        # No normalization
        scale = 1.0

    for node in G.nodes():
        degree = len(G[node]) # degree => length of the neighbor list
        centrality[node] = degree * scale
            
    return centrality

if __name__ == "__main__":
    G, all_ego_nodes, all_circles, all_features = create_complete_graph()

    print("\nCalculating Degree Centrality...")
    dc = degree_centrality(G, normalized=True)
    
    print("\n--- Top 10 Nodes by Degree Centrality ---")
    sorted_dc = sorted(dc.items(), key=lambda item: item[1], reverse=True)
    for i, (node, score) in enumerate(sorted_dc[:10]):
        print(f"{i+1}. Node {node}: {score:.8f}")