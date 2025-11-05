import os
import sys

# This path insertion allows us to import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from graph import create_complete_graph

def degree_centrality(G, normalized=True):
    """
    Computes the degree centrality of all nodes in a graph G.
    
    Args:
        G (networkx.Graph): The graph.
        normalized (bool): If True, normalize by (n-1).
        
    Returns:
        dict: A dictionary of nodes with their degree centrality scores.
    """
    
    # 1. Initialize the centrality dictionary
    centrality = {}
    
    # 2. Get the total number of nodes
    n = G.number_of_nodes()
    
    # 3. Handle trivial case (empty or single-node graph)
    if n <= 1:
        return {node: 0.0 for node in G.nodes()}

    # 4. Calculate the normalization factor
    if normalized:
        # The score is normalized by n-1, the maximum possible degree
        scale = 1.0 / (n - 1)
    else:
        # If not normalizing, the "scale" is just 1
        scale = 1.0

    # 5. Iterate over all nodes to calculate their centrality
    # G.degree provides a "DegreeView" which is like a dict of (node, degree) pairs
    # This is the most efficient way to get all degrees
    for node, degree in G.degree():
        # Calculate the centrality score and store it
        centrality[node] = degree * scale
            
    return centrality

if __name__ == "__main__":
    # Load the complete graph using the function from 'graph.py'
    print("Loading graph from dataset...")
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(
        dataset_path=os.path.join("dataset") # Assuming 'dataset' is in the 'aad' folder
    )
    print("Graph loaded:", G)

    # Calculate Degree Centrality using our custom function
    print("\nCalculating Degree Centrality (custom implementation)...")
    dc = degree_centrality(G, normalized=True)
    
    print("\n--- Top 10 Nodes by Degree Centrality ---")
    
    # Sort the results
    sorted_dc = sorted(dc.items(), key=lambda item: item[1], reverse=True)
    
    # Print the top 10
    for i, (node, score) in enumerate(sorted_dc[:10]):
        print(f"{i+1}. Node {node}: {score:.8f}")