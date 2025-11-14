import os
import sys
from collections import deque # We need a double-ended queue for an efficient BFS

# Import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

def closeness_centrality(G, normalized=True):
    """Computes the closeness centrality for all nodes, handling disconnected graphs."""
    
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
        
        # 'distance' tracks the shortest path from 'u' to all other nodes
        distance = {w: -1 for w in G.nodes()}
        distance[u] = 0
        
        # 'total_distance' = sum of distances to all *reachable* nodes
        # 'reachable_nodes' = count of all *reachable* nodes
        total_distance = 0.0
        reachable_nodes = 0
        
        queue = deque([u])
        
        # Start the BFS
        while queue:
            v = queue.popleft()
            
            # Look at all neighbors 'w' of 'v'
            for w in G.neighbors(v):
                
                # If 'w' hasn't been seen yet
                if distance[w] < 0:
                    distance[w] = distance[v] + 1
                    queue.append(w)
                    
                    # Update our running totals for reachable nodes
                    total_distance += distance[w]
                    reachable_nodes += 1
                    
        # --- 4. Calculate Final Score ---
        
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
    # We need to find the 'dataset' folder, which is one level up from here
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(
        dataset_path=dataset_path
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