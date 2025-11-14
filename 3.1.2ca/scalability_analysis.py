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

def run_all_centralities(G):
    """Runs all 4 centrality functions and returns their runtimes."""
    
    runtimes = {}
    
    # Time Degree Centrality
    start_time = time.time()
    degree_centrality(G)
    runtimes['Degree'] = time.time() - start_time
    
    # Time Betweenness Centrality
    start_time = time.time()
    betweenness_centrality(G)
    runtimes['Betweenness'] = time.time() - start_time
    
    # Time Closeness Centrality
    start_time = time.time()
    closeness_centrality(G)
    runtimes['Closeness'] = time.time() - start_time
    
    # Time Eigenvector Centrality
    start_time = time.time()
    eigenvector_centrality(G)
    runtimes['Eigenvector'] = time.time() - start_time
    
    return runtimes

def main():
    # 2) SETUP GRAPH LOADING ---
    
    # Get the absolute path to the dataset folder
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    
    # Define how many graphs to test.
    # num_files = 1 will be the smallest, num_files = 10 will be the largest
    graph_sizes_to_test = range(1, 11)
    
    all_results = []

    print("Starting Scalability Analysis...")

    # 3) LOOP AND RUN ANALYSIS ---
    
    for num_files in graph_sizes_to_test:
        print(f"\n--- Analyzing Graph with {num_files} ego-network(s) ---")
        
        # Load the graph with the specified number of files
        # We pass BOTH num_files and the correct dataset_path
        G, _, _, _ = create_complete_graph(num_files=num_files, dataset_path=dataset_path)
        
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        if num_nodes == 0:
            print("Warning: Graph is empty. Skipping.")
            continue
            
        print(f"Graph loaded: {num_nodes:,} Nodes, {num_edges:,} Edges")
        
        # Run all 4 centrality algorithms and time them
        runtimes = run_all_centralities(G)
        print(f"Runtimes: {runtimes}")
        
        # Store the results
        result_data = {
            'Num_Files': num_files,
            'Nodes': num_nodes,
            'Edges': num_edges,
            'Time_Degree': runtimes['Degree'],
            'Time_Betweenness': runtimes['Betweenness'],
            'Time_Closeness': runtimes['Closeness'],
            'Time_Eigenvector': runtimes['Eigenvector']
        }
        all_results.append(result_data)

    # 4) CREATE RESULTS TABLE ---
    
    print("\n--- Scalability Analysis Complete ---")
    
    # Convert the list of results into a pandas DataFrame
    df = pd.DataFrame(all_results)
    
    # Set 'Num_Files' as the index for clarity
    df = df.set_index('Num_Files')
    
    print("Runtime Results Table:")
    print(df)
    
    # Save the table to a CSV file for your report
    df.to_csv("scalability_results.csv")
    print("\nResults saved to 'scalability_results.csv'")

    # 5) CREATE PLOTS ---

    print("Generating scalability plots...")

    # Plot 1: Nodes vs. Time
    plt.figure(figsize=(12, 8))
    plt.plot(df['Nodes'], df['Time_Degree'], 'o-', label='Degree Centrality')
    plt.plot(df['Nodes'], df['Time_Betweenness'], 's-', label='Betweenness Centrality')
    plt.plot(df['Nodes'], df['Time_Closeness'], '^-', label='Closeness Centrality')
    plt.plot(df['Nodes'], df['Time_Eigenvector'], 'D-', label='Eigenvector Centrality')
    
    plt.title('Scalability Analysis: Nodes vs. Runtime', fontsize=16)
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.savefig('scalability_vs_nodes.png')
    print("Saved 'scalability_vs_nodes.png'")

    # Plot 2: Edges vs. Time
    plt.figure(figsize=(12, 8))
    plt.plot(df['Edges'], df['Time_Degree'], 'o-', label='Degree Centrality')
    plt.plot(df['Edges'], df['Time_Betweenness'], 's-', label='Betweenness Centrality')
    plt.plot(df['Edges'], df['Time_Closeness'], '^-', label='Closeness Centrality')
    plt.plot(df['Edges'], df['Time_Eigenvector'], 'D-', label='Eigenvector Centrality')
    
    plt.title('Scalability Analysis: Edges vs. Runtime', fontsize=16)
    plt.xlabel('Number of Edges', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.savefig('scalability_vs_edges.png')
    print("Saved 'scalability_vs_edges.png'")
    
    print("\nAll analysis finished.")

if __name__ == "__main__":
    main()