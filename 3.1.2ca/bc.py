import os
import sys
from collections import deque # We need a double-ended queue for an efficient BFS

# Import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

def betweenness_centrality(G, normalized=True):
    """Computes the betweenness centrality for all nodes using Brandes' algorithm."""
    
    # 1. Initialize the centrality score for all nodes to 0.0
    centrality = {node: 0.0 for node in G.nodes()}
    
    # 2. We have to run the algorithm for every single node as a source 's'
    for s in G.nodes():
        
        # --- 3. Forward Pass: Find all shortest paths from 's' ---
        # These data structures are reset for each new source 's'
        
        # Used to process nodes in the correct order during the backward pass
        stack = []
        
        # predecessors[w] = list of nodes 'v' that are on a shortest path from s to w
        predecessors = {w: [] for w in G.nodes()}
        
        # sigma[w] = number of shortest paths from s to w
        sigma = {w: 0.0 for w in G.nodes()}
        sigma[s] = 1.0
        
        # distance[w] = distance from s to w
        distance = {w: -1 for w in G.nodes()}
        distance[s] = 0
        
        queue = deque([s])
        
        # Start the Breadth-First Search
        while queue:
            v = queue.popleft() # Get the next node 'v'
            stack.append(v)     # Add 'v' to the stack for the backward pass
            
            # Look at all neighbors 'w' of 'v'
            for w in G.neighbors(v):
                
                # Case 1: 'w' hasn't been seen yet.
                if distance[w] < 0:
                    distance[w] = distance[v] + 1 # Set distance
                    queue.append(w)               # Add 'w' to the queue
                
                # Case 2: This is a shortest path to 'w'
                if distance[w] == distance[v] + 1:
                    sigma[w] += sigma[v]      # Add 'v's path count to 'w's
                    predecessors[w].append(v) # 'v' is a predecessor of 'w'
                    
        # --- 4. Backward Pass: Accumulate dependencies ---
        
        # This dictionary stores the "dependency" of the source 's' on all other nodes
        dependency = {w: 0.0 for w in G.nodes()}
        
        # Process nodes in reverse order (farthest from 's' first)
        while stack:
            w = stack.pop()
            
            # Go through all predecessors 'v' of 'w'
            for v in predecessors[w]:
                
                # Calculate the "credit" 'v' gets for being on a shortest path to 'w'
                credit = (sigma[v] / sigma[w]) * (1.0 + dependency[w])
                
                # Add this credit to 'v's dependency score
                dependency[v] += credit
            
            # Add the dependency score to the final centrality
            # (as long as 'w' is not the source 's' itself)
            if w != s:
                centrality[w] += dependency[w]

    # --- 5. Finalization and Normalization ---
    
    n = len(centrality)
    
    # Handle small graphs
    if n <= 2:
        return centrality

    # Normalize the scores
    if normalized:
        # Scale by the number of all possible node pairs
        scale = 1.0 / ((n - 1) * (n - 2))
        for node in centrality:
            centrality[node] *= scale
            
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

    # Calculate Betweenness Centrality using our custom function
    print("\nCalculating Betweenness Centrality (custom implementation)...")
    bc = betweenness_centrality(G, normalized=True)
    
    print("\n--- Top 10 Nodes by Betweenness Centrality ---")
    
    # Sort the results
    sorted_bc = sorted(bc.items(), key=lambda item: item[1], reverse=True)
    
    # Print the top 10
    for i, (node, score) in enumerate(sorted_bc[:10]):
        print(f"{i+1}. Node {node}: {score:.8f}")