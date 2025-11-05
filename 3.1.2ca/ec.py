import os
import sys
import math # We need math.sqrt() for normalization

# This path insertion allows us to import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from graph import create_complete_graph

def eigenvector_centrality(G, max_iter=100, tol=1.0e-6):
    """
    Computes the eigenvector centrality of all nodes in a graph G.
    This is a custom implementation of the Power Iteration method.
    
    Args:
        G (networkx.Graph): The graph.
        max_iter (int): Maximum number of iterations in power method.
        tol (float): Tolerance for convergence.
        
    Returns:
        dict: A dictionary of nodes with their eigenvector centrality scores.
    """
    
    # 1. Handle trivial case (empty graph)
    if len(G) == 0:
        return {}
        
    # 2. Initialize the centrality vector. Start with all nodes having a score of 1.0.
    # This is our initial guess, x_0
    x = {node: 1.0 for node in G.nodes()}
    
    # 3. Start the Power Iteration
    for i in range(max_iter):
        
        # Keep a copy of the old scores to check for convergence
        x_old = x.copy()
        
        # This dictionary will hold the scores for the next iteration (x_{k+1})
        x_new = {node: 0.0 for node in G.nodes()}
        
        # This will be the squared sum of the new scores, for normalization
        norm_L2_squared = 0.0

        # --- 4. Core Calculation Step ---
        # x_new = A * x_old
        # For each node 'v', its new score is the sum of the old scores of its neighbors
        for v in G.nodes():
            for u in G.neighbors(v):
                x_new[v] += x_old[u]
            
            # Add the square of the new score to our running sum
            norm_L2_squared += x_new[v]**2

        # --- 5. Normalization Step ---
        
        # Calculate the L2 norm (square root of the sum of squares)
        norm = math.sqrt(norm_L2_squared)
        
        # Avoid division by zero if the graph has no edges (all scores are 0)
        if norm == 0:
            return x_new # All centralities are 0
            
        # Normalize all scores in x_new
        for v in G.nodes():
            x[v] = x_new[v] / norm
            
        # --- 6. Convergence Check ---
        
        # Calculate the L1 norm of the difference vector (sum of absolute differences)
        diff_norm_L1 = sum(abs(x[v] - x_old[v]) for v in G.nodes())
        
        # If the change is smaller than our tolerance, we have converged
        if diff_norm_L1 < tol:
            return x # Return the converged centrality scores
            
    # If we hit max_iter without converging, we raise an error.
    # This can happen on disconnected graphs if not handled.
    print(f"Warning: Eigenvector centrality did not converge in {max_iter} iterations.")
    return x # Return the best guess we have

if __name__ == "__main__":
    # Load the complete graph using the function from 'graph.py'
    print("Loading graph from dataset...")
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(
        dataset_path=os.path.join("dataset") # Assuming 'dataset' is in the 'aad' folder
    )
    print("Graph loaded:", G)

    # Calculate Eigenvector Centrality using our custom function
    print("\nCalculating Eigenvector Centrality (custom implementation)...")
    ec = eigenvector_centrality(G, max_iter=100, tol=1.0e-6)
    
    print("\n--- Top 10 Nodes by Eigenvector Centrality ---")
    
    # Sort the results
    sorted_ec = sorted(ec.items(), key=lambda item: item[1], reverse=True)
    
    # Print the top 10
    for i, (node, score) in enumerate(sorted_ec[:10]):
        print(f"{i+1}. Node {node}: {score:.8f}")