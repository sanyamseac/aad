import os
import sys

# import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

def degree_centrality(G, normalized=True):
    """Computtles the degree centrality for all nodes, normalized by (n-1)."""
    
    # 1) Initialize the centrality dictionary
    centrality = {}
    
    # 2) Get the total number of nodes
    n = G.number_of_nodes()
    
    # 3) for empty or single-node graph
    if n <= 1:
        return {node: 0.0 for node in G.nodes()}

    # 4) Calculate the normalization factor
    if normalized:
        # The score is normalized by n-1, the maximum possible degree
        scale = 1.0 / (n - 1)
    else:
        # If not normalizing ==> scale = 1
        scale = 1.0

    # 5) Iterate over all nodes to calculate their centrality
    for node in G.nodes():
        # Manually find the degree by counting the node's neighbors
        degree = len(G[node]) # degree => length of the neighbor list
        
        # Calculate the centrality score and store it
        centrality[node] = degree * scale
            
    return centrality

if __name__ == "__main__":
    # Load the complete graph using the function from 'graph.py'
    print("Loading graph from dataset...")
    # We need to find the 'dataset' folder, which is one level up from here
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(
        dataset_path=dataset_path
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