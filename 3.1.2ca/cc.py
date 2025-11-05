import os
import sys
from collections import deque # We need a double-ended queue for an efficient BFS

# This path insertion allows us to import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from graph import create_complete_graph

def closeness_centrality(G, normalized=True):
    """
    Computes the closeness centrality of all nodes in a graph G.
    This is a custom implementation that handles disconnected graphs.
    
    Args:
        G (networkx.Graph): The graph.
        normalized (bool): If True, normalize by the fraction of reachable nodes.
        
    Returns:
        dict: A dictionary of nodes with their closeness centrality scores.
    """
    
    # 1. Initialize the centrality dictionary with 0.0 for all nodes
    centrality = {node: 0.0 for node in G.nodes()}
    
    # Get the total number of nodes in the graph
    n = G.number_of_nodes()
    
    # Handle trivial case (single node or empty graph)
    if n <= 1:
        return centrality
        
    # 2. Iterate over every node 'u' in the graph. 'u' will be our source.
    for u in G.nodes():
        
        # --- 3. Single-Source Shortest Path (BFS) ---
        # We run a separate BFS from each node 'u'
        
        # Dictionary to store the distance from 'u' to 'w'
        distance = {w: -1 for w in G.nodes()}
        distance[u] = 0 # The distance from 'u' to 'u' is 0
        
        # A queue for our BFS
        queue = deque([u])
        
        # This will be the sum of distances to all reachable nodes (Farness)
        total_distance = 0.0 
        
        # This will be the count of nodes reachable from 'u'
        reachable_nodes = 0  
        
        # Start the BFS from 'u'
        while queue:
            v = queue.popleft() # Get the next node 'v' from the queue
            
            # Iterate over all neighbors 'w' of 'v'
            for w in G.neighbors(v):
                
                # Case 1: 'w' is found for the first time
                if distance[w] == -1:
                    distance[w] = distance[v] + 1 # Set distance
                    queue.append(w)               # Add 'w' to the queue
                    
                    # We have found a new reachable node
                    total_distance += distance[w] # Add its distance to the sum
                    reachable_nodes += 1          # Increment the count
                    
        # --- 4. Calculate Closeness for Node 'u' ---
        
        # If the node is isolated (or the graph is just 1 node)
        if total_distance == 0:
            centrality[u] = 0.0
        else:
            # Calculate the "un-normalized" closeness
            # (number of reachable nodes) / (sum of distances)
            cc = reachable_nodes / total_distance
            
            if normalized:
                # Apply Wasserman & Faust normalization
                # This scales the closeness by the fraction of the graph
                # that 'u' can actually reach.
                scale_factor = reachable_nodes / (n - 1)
                centrality[u] = cc * scale_factor
            else:
                # Store the un-normalized closeness
                centrality[u] = cc
                
    return centrality

if __name__ == "__main__":
    # Load the complete graph using the function from 'graph.py'
    print("Loading graph from dataset...")
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(
        dataset_path=os.path.join("dataset") # Assuming 'dataset' is in the 'aad' folder
    )
    print("Graph loaded:", G)

    # Calculate Closeness Centrality using our custom function
    print("\nCalculating Closeness Centrality (custom implementation)...")
    cc = closeness_centrality(G, normalized=True)
    
    print("\n--- Top 10 Nodes by Closeness Centrality ---")
    
    # Sort the results
    sorted_cc = sorted(cc.items(), key=lambda item: item[1], reverse=True)
    
    # Print the top 10
    for i, (node, score) in enumerate(sorted_cc[:10]):
        print(f"{i+1}. Node {node}: {score:.8f}")