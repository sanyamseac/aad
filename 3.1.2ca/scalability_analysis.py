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

def run_all_centralities(G):
    """Runs all 4 centrality functions and returns their runtimes"""
    
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
    # SETUP GRAPH LOADING
    
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    
    # Define the no of graphs to test
    graph_sizes_to_test = range(1, 11)
    
    all_results = []

    print("Starting Scalability Analysis...")

    # LOOP AND RUN ANALYSIS 
    
    for num_files in graph_sizes_to_test:
        print(f"\n--- Analyzing Graph with {num_files} ego-network(s) ---")
        
        # Load the graph with the specified number of files
        # We MUST unpack the 4-item tuple here
        G, _, _, _ = create_complete_graph(num_files=num_files, dataset_path=dataset_path)
        
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        if num_nodes == 0:
            print(f"Warning: Graph for {num_files} files is empty. Skipping.")
            continue
            
        print(f"Graph loaded: {num_nodes:,} Nodes, {num_edges:,} Edges")
        
        # Run all 4 centrality algorithms and time them
        runtimes = run_all_centralities(G)
        print(f"Runtimes: {runtimes}")
        
        # Calculate the value of the theoretical complexities
        complex_V = num_nodes
        complex_E = num_edges
        complex_V_VE = num_nodes * (num_nodes + num_edges)
        
        # Store all the results
        result_data = {
            'Num_Files': num_files,
            'Nodes': num_nodes,
            'Edges': num_edges,
            'Complex_V': complex_V,
            'Complex_E': complex_E,
            'Complex_V_VE': complex_V_VE,
            'Time_Degree': runtimes['Degree'],
            'Time_Betweenness': runtimes['Betweenness'],
            'Time_Closeness': runtimes['Closeness'],
            'Time_Eigenvector': runtimes['Eigenvector']
        }
        all_results.append(result_data)

    # --- 4. CREATE RESULTS TABLE ---
    
    print("\n--- Scalability Analysis Complete ---")
    
    # Convert the list of results into a pandas DataFrame
    df = pd.DataFrame(all_results)
    
    if not df.empty:
        df = df.set_index('Num_Files')
    
    print("Runtime Results Table (with Complexity Values):")
    print(df)
    
    # Save the table to a CSV file for your report
    df.to_csv("scalability_results.csv")
    print("\nResults saved to 'scalability_results.csv'")

    # CREATE PLOTS (TIME vs. SIZE) 
    
    if df.empty:
        print("No data to plot. Exiting.")
        return

    print("Generating Time vs. Size plots...")

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
    
    # CREATE PLOTS (TIME vs. COMPLEXITY) 

    print("Generating Time vs. Complexity plots...")

    # Create a 2x2 grid for our four plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Runtime vs. Theoretical Complexity', fontsize=18)

    # Plot 3: Degree Centrality (Time vs. V)
    axes[0, 0].plot(df['Complex_V'], df['Time_Degree'], 'o-b')
    axes[0, 0].set_title('Degree Centrality', fontsize=14)
    axes[0, 0].set_xlabel('Theoretical Complexity: O(V)', fontsize=12)
    axes[0, 0].set_ylabel('Time (seconds)', fontsize=12)
    axes[0, 0].grid(True)

    # Plot 4: Closeness Centrality (Time vs. V*(V+E))
    axes[0, 1].plot(df['Complex_V_VE'], df['Time_Closeness'], '^-r')
    axes[0, 1].set_title('Closeness Centrality', fontsize=14)
    axes[0, 1].set_xlabel('Theoretical Complexity: O(V(V+E))', fontsize=12)
    axes[0, 1].set_ylabel('Time (seconds)', fontsize=12)
    axes[0, 1].grid(True)

    # Plot 5: Betweenness Centrality (Time vs. V*(V+E))
    axes[1, 0].plot(df['Complex_V_VE'], df['Time_Betweenness'], 's-g')
    axes[1, 0].set_title('Betweenness Centrality', fontsize=14)
    axes[1, 0].set_xlabel('Theoretical Complexity: O(V(V+E))', fontsize=12)
    axes[1, 0].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 0].grid(True)

    # Plot 6: Eigenvector Centrality (Time vs. E)
    # We plot against E, as O(k*E) ( assuming k to be roughly constant)
    axes[1, 1].plot(df['Complex_E'], df['Time_Eigenvector'], 'D-m')
    axes[1, 1].set_title('Eigenvector Centrality', fontsize=14)
    axes[1, 1].set_xlabel('Theoretical Complexity: O(k*E)', fontsize=12)
    axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 1].grid(True)

    # Clean up the layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the combined plot
    plt.savefig('scalability_vs_complexity.png')
    print("Saved 'scalability_vs_complexity.png'")
    
    print("\nAll analysis finished.")

if __name__ == "__main__":
    main()