import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt


# Import the 4 centrality functions 
from dc import degree_centrality
from bc import betweenness_centrality
from cc import closeness_centrality
from ec import eigenvector_centrality

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

def print_top_nodes(df, column, top_n=10):
    """Helper function to print a sorted list of top nodes."""
    print(f"\n--- Top {top_n} Nodes by {column} ---")
    
    # Sort the DataFrame by the specified column and print the top N
    top_nodes = df.sort_values(by=column, ascending=False).head(top_n)
    print(top_nodes)

def main():
    # LOAD GRAPH 
    
    print("Loading complete graph from dataset...")
    
    # Get the absolute path to the dataset folder
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    
    # This loads the ONE combined graph
    # We correctly unpack all 4 values returned by the function
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(
        dataset_path=dataset_path  
    )
    print(f"Graph loaded successfully.")
    print(f"Total Nodes: {G.number_of_nodes():,}")
    print(f"Total Edges: {G.number_of_edges():,}")
    
    # RUN & TIME ALGORITHMS 
    
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

    # CONSOLIDATE RESULTS 
    
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
    
    # DELIVERABLE : Top Influential Users 
    
    print("\n" + "="*50)
    print(" DELIVERABLE: TOP INFLUENTIAL USERS")
    print("="*50)
    print("(These are the 'most popular', 'best brokers', etc.)")
    print_top_nodes(df, 'Degree', top_n=10)
    print_top_nodes(df, 'Betweenness', top_n=10)
    print_top_nodes(df, 'Closeness', top_n=10)
    print_top_nodes(df, 'Eigenvector', top_n=10)
    
    # DELIVERABLE : Correlation Analysis 
    
    print("\n" + "="*50)
    print(" DELIVERABLE: CORRELATION ANALYSIS")
    print("="*50)
    # Calculating the Pearson correlation between all 4 centrality measures
    correlation_matrix = df.corr()
    print(correlation_matrix)
    
    # Centrality Distribution 
    
    print("\n" + "="*50)
    print(" DELIVERABLE: CENTRALITY DISTRIBUTION")
    print("="*50)
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
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])           # Adjust for suptitle
    
    # Saving plot to a file
    output_fig = "centrality_distributions.png"
    plt.savefig(output_fig)
    print(f"Saved distribution plots to '{output_fig}'")
    
    # Runtime Analysis (Summary Table) 
    
    print("\n" + "="*50)
    print(" DELIVERABLE: RUNTIME ANALYSIS (FULL GRAPH)")
    print("="*50)
    print(f"{'Metric':<18} | {'Time (seconds)':<15}")
    print("-" * 35)
    for metric, runtime in runtimes.items():
        print(f"{metric:<18} | {runtime:<15.4f}")

    # REAL-WORLD PERSONA ANALYSIS 
    
    print("\n" + "="*50)
    print(" DELIVERABLE: REAL-WORLD PERSONA ANALYSIS")
    print("="*50)
    
    # Define our thresholds using quantiles
    # "High" = Top 25% (0.75 quantile)
    # "Low" = Bottom 50% (0.50 quantile)
    high_degree = df['Degree'].quantile(0.75)
    low_degree = df['Degree'].quantile(0.50)
    high_betweenness = df['Betweenness'].quantile(0.75)
    low_betweenness = df['Betweenness'].quantile(0.50)
    high_closeness = df['Closeness'].quantile(0.75)
    
    # Persona 1: "The Hub" (or "Local Celebrity")
    # High Degree (popular) but Low Betweenness (not a bridge)
    # They are the center of a single, dense group.
    hubs = df[
        (df['Degree'] > high_degree) &
        (df['Betweenness'] < low_betweenness)
    ]
    print(f"\nFound {len(hubs)} 'Hubs' (High Degree, Low Betweenness):")
    print(hubs.sort_values(by='Degree', ascending=False).head(10))
    
    # Persona 2: "The Broker" (or "Gatekeeper")
    # High Betweenness (a bridge) but Low Degree (not popular)
    # They are critical for connecting different groups.
    brokers = df[
        (df['Betweenness'] > high_betweenness) &
        (df['Degree'] < low_degree)
    ]
    print(f"\nFound {len(brokers)} 'Brokers' (High Betweenness, Low Degree):")
    print(brokers.sort_values(by='Betweenness', ascending=False).head(10))
    
    # Persona 3: "The Leader" (or "Network Super-Star")
    # High in ALL key metrics.
    leaders = df[
        (df['Degree'] > high_degree) &
        (df['Betweenness'] > high_betweenness) &
        (df['Closeness'] > high_closeness)
    ]
    print(f"\nFound {len(leaders)} 'Leaders' (High Degree, Betweenness, & Closeness):")
    print(leaders.sort_values(by='Degree', ascending=False).head(10))
    
    print("\n" + "="*50)
    print("\nFull graph analysis finished.")

if __name__ == "__main__":
    main()