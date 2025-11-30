import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Current folder path
DATA_FOLDER = '.' 

# File names (Make sure these match your actual files)
EDGE_FILE = 'graph_edges.csv'
ALGO_FILES = {
    'Girvan-Newman': 'girvan_newman_results.csv',
    'Louvain': 'lm_custom_results.csv',
    'Leiden': 'leiden_results.csv'
}

# Output settings
OUTPUT_DPI = 300  # High resolution for reports

# ==========================================
# 1. LOAD GRAPH & COMPUTE FIXED LAYOUT
# ==========================================
print("Loading graph data...")
edges_df = pd.read_csv(os.path.join(DATA_FOLDER, EDGE_FILE))

# Create Graph
G = nx.from_pandas_edgelist(edges_df, source='source', target='target')

print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
print("Computing fixed layout (Spring/Force-Directed)...")

# CRITICAL STEP: Calculate layout ONCE for all algorithms
# 'k' adjusts the spacing. None uses default (1/sqrt(n))
# 'seed' ensures the layout is the same every time you run this script
fixed_pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_community_data(filepath):
    """Reads CSV and converts to a dictionary {node: community_id}"""
    df = pd.read_csv(filepath)
    # Ensure node_id matches the type in G (usually string or int)
    # We try to align types based on the graph nodes
    first_node = list(G.nodes())[0]
    if isinstance(first_node, int):
        df['node_id'] = df['node_id'].astype(int)
    else:
        df['node_id'] = df['node_id'].astype(str)
        
    return pd.Series(df.community_id.values, index=df.node_id).to_dict()

def plot_network(graph, layout, partition, title, ax):
    """Draws the network with nodes colored by community"""
    # Get list of community IDs for coloring
    communities = [partition.get(n, 0) for n in graph.nodes()]
    
    # Draw edges (thin and transparent to not clutter)
    nx.draw_networkx_edges(graph, layout, alpha=0.1, edge_color='gray', ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, layout, 
                           nodelist=graph.nodes(),
                           node_color=communities, 
                           node_size=50, 
                           cmap=cm.tab20, # A colormap with many distinct colors
                           alpha=0.9, 
                           ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

def plot_distribution(partition, title, ax):
    """Plots the size distribution of communities"""
    counts = pd.Series(list(partition.values())).value_counts()
    
    # Plot histogram of sizes
    ax.hist(counts, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title(f"{title}\n(Community Size Dist.)", fontsize=10)
    ax.set_xlabel("Size of Community")
    ax.set_ylabel("Frequency")

# ==========================================
# 3. GENERATE PLOTS
# ==========================================
print("Generating visualizations...")

# Figure 1: Network Maps (The Visual Comparison)
fig_net, axes_net = plt.subplots(1, 3, figsize=(24, 8))
fig_dist, axes_dist = plt.subplots(1, 3, figsize=(18, 5))

results_summary = []

for i, (algo_name, filename) in enumerate(ALGO_FILES.items()):
    path = os.path.join(DATA_FOLDER, filename)
    
    if os.path.exists(path):
        print(f"Processing {algo_name}...")
        partition = load_community_data(path)
        
        # 1. Plot Network Map
        plot_network(G, fixed_pos, partition, algo_name, axes_net[i])
        
        # 2. Plot Size Distribution
        plot_distribution(partition, algo_name, axes_dist[i])
        
        # 3. Calculate Modularity (Optional Metric)
        # Note: Needs community partition formatted as list of sets
        from networkx.algorithms.community import modularity
        # Reverse dict: {0: [node1, node2], 1: [node3]}
        comm_sets = {}
        for node, comm_id in partition.items():
            comm_sets.setdefault(comm_id, []).append(node)
        
        try:
            mod_score = modularity(G, list(comm_sets.values()))
            results_summary.append({'Algorithm': algo_name, 'Modularity': mod_score})
        except Exception as e:
            print(f"Could not calc modularity for {algo_name}: {e}")
            
    else:
        print(f"WARNING: File {filename} not found.")
        axes_net[i].text(0.5, 0.5, "File Not Found", ha='center')

# Save Network Map
fig_net.tight_layout()
save_path_net = os.path.join(DATA_FOLDER, 'comparison_network_maps.png')
fig_net.savefig(save_path_net, dpi=OUTPUT_DPI)
print(f"Saved network maps to: {save_path_net}")

# Save Distributions
fig_dist.tight_layout()
save_path_dist = os.path.join(DATA_FOLDER, 'comparison_size_distributions.png')
fig_dist.savefig(save_path_dist, dpi=OUTPUT_DPI)
print(f"Saved distributions to: {save_path_dist}")

# Save Metrics to CSV
if results_summary:
    metrics_df = pd.DataFrame(results_summary)
    metrics_path = os.path.join(DATA_FOLDER, 'calculated_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print("Modularity scores saved to calculated_metrics.csv")

print("Done!")