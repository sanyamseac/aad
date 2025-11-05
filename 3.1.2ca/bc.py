import os
import sys
from collections import deque # We need a double-ended queue for an efficient BFS

# This path insertion allows us to import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from graph import create_complete_graph

def betweenness_centrality(G, normalized=True):
    """
    Computes the betweenness centrality of all nodes in a graph G.
    This is a custom implementation of Brandes' Algorithm.
    
    Args:
        G (networkx.Graph): The graph.
        normalized (bool): If True, normalize the centrality by the number of node pairs.
        
    Returns:
        dict: A dictionary of nodes with their betweenness centrality scores.
    """
    
    # 1. Initialize the centrality dictionary with 0.0 for all nodes
    centrality = {node: 0.0 for node in G.nodes()}
    
    # 2. Iterate over every node 's' in the graph. 's' will be our source.
    for s in G.nodes():
        
        # --- 3. Forward Pass (BFS) ---
        # These data structures are re-initialized for EACH source 's'
        
        # Stack to store nodes in the order they are visited (for backward pass)
        stack = []
        
        # Dictionary to store all predecessors of a node 'w' on shortest paths from 's'
        predecessors = {w: [] for w in G.nodes()}
        
        # Dictionary to store the number of shortest paths (sigma) from 's' to 'w'
        sigma = {w: 0.0 for w in G.nodes()}
        sigma[s] = 1.0 # The number of shortest paths from 's' to 's' is 1
        
        # Dictionary to store the distance from 's' to 'w'
        distance = {w: -1 for w in G.nodes()}
        distance[s] = 0 # The distance from 's' to 's' is 0
        
        # A queue for our BFS
        queue = deque([s])
        
        # Start the BFS
        while queue:
            v = queue.popleft() # Get the next node 'v' from the queue
            stack.append(v)     # Push 'v' onto the stack for the backward pass
            
            # Iterate over all neighbors 'w' of 'v'
            for w in G.neighbors(v):
                
                # Case 1: 'w' is found for the first time
                if distance[w] < 0:
                    distance[w] = distance[v] + 1 # Set distance
                    queue.append(w)               # Add 'w' to the queue
                
                # Case 2: 'w' is part of a shortest path from 's'
                if distance[w] == distance[v] + 1:
                    sigma[w] += sigma[v]      # Add the path count of 'v' to 'w'
                    predecessors[w].append(v) # 'v' is a predecessor of 'w'
                    
        # --- 4. Backward Pass (Accumulation) ---
        
        # This dictionary holds the "dependency" of 's' on each node 'w'
        dependency = {w: 0.0 for w in G.nodes()}
        
        # Process nodes in reverse order of their discovery (LIFO from the stack)
        while stack:
            w = stack.pop() # Get the node 'w' (farthest nodes first)
            
            # For each predecessor 'v' of 'w'
            for v in predecessors[w]:
                
                # Calculate the "credit" 'v' gets from 'w'
                # This is (paths through v / total paths) * (1 + dependency of w)
                credit = (sigma[v] / sigma[w]) * (1.0 + dependency[w])
                
                # Add this credit to the dependency of 'v'
                dependency[v] += credit
            
            # If 'w' is not the source 's', add its dependency to its final centrality score
            # (We don't count the source 's' in the paths s -> ... -> t)
            if w != s:
                centrality[w] += dependency[w]

    # --- 5. Finalization and Normalization ---
    
    # Get the number of nodes
    n = len(centrality)
    
    # Handle trivial graphs (0, 1, or 2 nodes)
    if n <= 2:
        return centrality

    # Normalization factor for undirected graphs
    # We divide by (n-1)*(n-2)
    # The '2' is already handled because Brandes' algorithm sums dependency
    # from both directions (s -> t and t -> s), effectively double-counting.
    # So we don't need to divide by 2 *again*.
    if normalized:
        scale = 1.0 / ((n - 1) * (n - 2))
        for node in centrality:
            centrality[node] *= scale
            
    return centrality

if __name__ == "__main__":
    # Load the complete graph using the function from 'graph.py'
    print("Loading graph from dataset...")
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(
        dataset_path=os.path.join("dataset") # Assuming 'dataset' is in the 'aad' folder
    )
    print("Graph loaded:", G)

    # Calculate Betweenness Centrality using our custom function
    print("\nCalculating Betweenness Centrality (custom implementation)...")
    bc = betweenness_centrality(G, normalized=True)
    
    print("\n--- Top 10 Nodes by Betweenness Centrality ---")
    
    # Sort the results
    sorted_bc = sorted(bc.items(), key=lambda item: item[1], reverse=True)
    
    # Print the top 10
    for i, (node, score) in enumerate(sorted_bc[:10]):
        print(f"{i+1}. Node {node}: {score:.8f}")