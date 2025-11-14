import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

# 1) SETUP ---

# Import the 4 centrality functions from your files in this folder
from dc import degree_centrality
from bc import betweenness_centrality
from cc import closeness_centrality
from ec import eigenvector_centrality

# Import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

def print_top_nodes(df, column, top_n=10):
    """Helper function to print a sorted list of top nodes."""
    print(f"\n--- Top {top_n} Nodes by {column} ---")
    # Sort the DataFrame by the specified column and print the top N
    top_nodes = df.sort_values(by=column, ascending=False).head(top_n)
    print(top_nodes)

def main():
    # 2) LOAD GRAPH ---
    
    print("Loading complete graph from dataset...")
    

    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    
    # This loads the ONE combined graph
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(
        dataset_path=dataset_path  # Pass the correct path here
    )
    print(f"Graph loaded successfully.")
    print(f"Total Nodes: {G.number_of_nodes():,}")
    print(f"Total Edges: {G.number_of_edges():,}")
    
    # 3) RUN & TIME ALGORITHMS (Deliverable: Runtime Analysis) ---
    
    print("\nStarting centrality calculations...")
    runtimes = {}
    
    # Calculate Degree Centrality
    start_time = time.time()
    dc = degree_centrality(G)
    runtimes['Degree'] = time.time() - start_time
    print(f"Degree Centrality finished in {runtimes['Degree']:.4f}s")
    
    # Calculate Betweenness Centrality
    start_time = time.time()
    bc = betweenness_centrality(G)
    runtimes['Betweenness'] = time.time() - start_time
    print(f"Betweenness Centrality finished in {runtimes['Betweenness']:.4f}s")
    
    # Calculate Closeness Centrality
    start_time = time.time()
    cc = closeness_centrality(G)
    runtimes['Closeness'] = time.time() - start_time
    print(f"Closeness Centrality finished in {runtimes['Closeness']:.4f}s")
    
    # Calculate Eigenvector Centrality
    start_time = time.time()
    ec = eigenvector_centrality(G)
    runtimes['Eigenvector'] = time.time() - start_time
    print(f"Eigenvector Centrality finished in {runtimes['Eigenvector']:.4f}s")

    # 4) CONSOLIDATE RESULTS ---
    
    print("\nConsolidating results into DataFrame...")
    # Create a pandas DataFrame from the centrality dictionaries
    df = pd.DataFrame({
        'Degree': dc,
        'Betweenness': bc,
        'Closeness': cc,
        'Eigenvector': ec
    })
    
    # Fill any missing values (e.g., from disconnected components) with 0
    df = df.fillna(0)
    
    # 5) DELIVERABLE: Top Influential Users ---
    
    # Use the helper function to print the top 10 for each metric
    print_top_nodes(df, 'Degree', top_n=10)
    print_top_nodes(df, 'Betweenness', top_n=10)
    print_top_nodes(df, 'Closeness', top_n=10)
    print_top_nodes(df, 'Eigenvector', top_n=10)
    
    # 6) DELIVERABLE: Correlation Analysis ---
    
    print("\n--- Correlation Matrix ---")
    # Calculate the Pearson correlation between all 4 centrality measures
    correlation_matrix = df.corr()
    print(correlation_matrix)
    
    # 7) DELIVERABLE: Centrality Distribution ---
    
    print("\nGenerating centrality distribution plots...")
    # Create a 2x2 grid of histograms
    # .hist() returns a 2x2 array of plot 'axes'
    axes = df.hist(bins=50, figsize=(12, 10), log=True, grid=False)
    
    # axes[0,0] is top-left, axes[0,1] is top-right, etc.
    axes[0,0].set_title("Degree Centrality")
    axes[0,0].set_xlabel("Normalized Degree Score")
    axes[0,0].set_ylabel("Number of Nodes (Frequency)")
    
    axes[0,1].set_title("Betweenness Centrality")
    axes[0,1].set_xlabel("Normalized Betweenness Score")
    axes[0,1].set_ylabel("Number of Nodes (Frequency)")
    
    axes[1,0].set_title("Closeness Centrality")
    axes[1,0].set_xlabel("Normalized Closeness Score")
    axes[1,0].set_ylabel("Number of Nodes (Frequency)")
    
    axes[1,1].set_title("Eigenvector Centrality")
    axes[1,1].set_xlabel("Normalized Eigenvector Score")
    axes[1,1].set_ylabel("Number of Nodes (Frequency)")
    
    # Add an overall title
    plt.suptitle("Centrality Distributions (Log Scale)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    # Save the plot to a file
    output_fig = "centrality_distributions.png"
    plt.savefig(output_fig)
    print(f"Saved distribution plots to '{output_fig}'")
    
    # 8) DELIVERABLE: Runtime Analysis (Summary Table) ---
    
    print("\n--- Runtime Analysis Summary ---")
    print(f"{'Metric':<18} | {'Time (seconds)':<15}")
    print("-" * 35)
    for metric, runtime in runtimes.items():
        print(f"{metric:<18} | {runtime:<15.4f}")

if __name__ == "__main__":
    main()