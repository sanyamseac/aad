import os
import sys
import math # math.sqrt() for normalization

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

def eigenvector_centrality(G, max_iter=100, tol=1.0e-6):
    """
    Computes the eigenvector centrality for all nodes using the Power Iteration method
    
    Parameters:
    G (networkx.Graph): The input graph for which to compute eigenvector centrality
    max_iter (int): Maximum number of iterations for convergence (Default is 100)
    tol (float): Convergence tolerance for stopping criterion (Default is 1.0e-6)
    
    Returns:
    dict: A dictionary mapping each node to its eigenvector centrality score
    """

    if len(G) == 0:
        return {}
        
    # Initialize the centrality vector ==> start with all nodes having a score of 1
    # this is our initial guess, x_0
    x = {node: 1.0 for node in G.nodes()}
    
    # 3) start the Power Iteration
    for i in range(max_iter):
        
        # keep a copy of the old scores to check for convergence
        x_old = x.copy()
        
        # dictionary will hold the scores for the next iteration (x_{k+1})
        x_new = {node: 0.0 for node in G.nodes()}
        
        # stores the sum of squares for L2 normalization
        norm_sq = 0.0
        
        # Update Scores
        # main matrix-vector multiplication (A * x_old)
        for v in G.nodes():
            # Calculate the new score for node 'v'
            for u in G.neighbors(v):
                # 'v's new score is the sum of its neighbors' old scores
                x_new[v] += x_old[u]
            
            # Add to the sum of squares
            norm_sq += x_new[v]**2
            
        # Normalize Scores
        norm = math.sqrt(norm_sq)
        if norm == 0:
            # handles a graph with no edges
            return {node: 0.0 for node in G.nodes()}

        # Divide all scores by the L2 norm (the "length" of the vector)
        for v in G.nodes():
            x[v] = x_new[v] / norm
            
        # Convergence Check
        
        # Calculate the L1 norm of the difference vector (sum of absolute differences)
        diff_norm_L1 = sum(abs(x[v] - x_old[v]) for v in G.nodes())
        
        # If the change is smaller than our tolerance ==> we have converged
        if diff_norm_L1 < tol:
            return x        # Return the converged centrality scores
            
    # If we hit max_i-ter without converging ==> we raise an error
    # This can happen on disconnected graphs if not handled
    print(f"Warning: Eigenvector centrality did not converge in {max_iter} iterations.")
    return x        # Return the best guess we have

if __name__ == "__main__":
    # Load the complete graph using the function from 'graph.py'
    print("Loading graph from dataset...")
    # We need to find the 'dataset' folder, which is one level up from here
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(
        dataset_path=dataset_path
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