import networkx as nx
import matplotlib.pyplot as plt
from graph import create_complete_facebook_graph
import numpy as np
import os
import glob

def visualize_complete_graph(G, ego_nodes, max_nodes=5000):
    """Visualize the complete graph (sample if too large)"""
    plt.figure(figsize=(20, 20))
    
    # Sample nodes if graph is too large
    if G.number_of_nodes() > max_nodes:
        print(f"Graph too large ({G.number_of_nodes()} nodes). Sampling {max_nodes} nodes for visualization...")
        nodes_to_draw = list(ego_nodes) + list(G.nodes())[:max_nodes - len(ego_nodes)]
        subG = G.subgraph(nodes_to_draw)
    else:
        subG = G
    
    # Use spring layout
    pos = nx.spring_layout(subG, k=1, iterations=30, seed=42)
    
    # Draw ego nodes
    nx.draw_networkx_nodes(subG, pos, nodelist=[n for n in ego_nodes if n in subG], 
                          node_color='red', node_size=300, 
                          label='Ego Nodes', alpha=0.8)
    
    # Draw alter nodes
    alter_nodes = [n for n in subG.nodes() if n not in ego_nodes]
    nx.draw_networkx_nodes(subG, pos, nodelist=alter_nodes,
                          node_color='lightblue', node_size=50,
                          label='Alter Nodes', alpha=0.6)
    
    # Draw edges
    nx.draw_networkx_edges(subG, pos, alpha=0.1, width=0.5)
    
    plt.title(f'Complete Facebook Ego Network Graph\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('d:\\aad\\complete_facebook_network.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_complete_graph(G, ego_nodes, circles, features):
    """Print comprehensive graph statistics"""
    print("\n" + "=" * 60)
    print("COMPLETE FACEBOOK GRAPH STATISTICS")
    print("=" * 60)
    print(f"Total number of nodes: {G.number_of_nodes():,}")
    print(f"Total number of edges: {G.number_of_edges():,}")
    print(f"Number of ego networks: {len(ego_nodes)}")
    print(f"Graph density: {nx.density(G):.6f}")
    
    # Calculate average degree
    degrees = [d for n, d in G.degree()]
    print(f"Average degree: {np.mean(degrees):.2f}")
    print(f"Max degree: {np.max(degrees)}")
    print(f"Min degree: {np.min(degrees)}")
    
    # Connected components
    num_components = nx.number_connected_components(G)
    print(f"\nNumber of connected components: {num_components}")
    
    if num_components > 1:
        largest_cc = max(nx.connected_components(G), key=len)
        print(f"Largest component size: {len(largest_cc):,} nodes")
    
    # Clustering coefficient (on sample if too large)
    if G.number_of_nodes() <= 5000:
        avg_clustering = nx.average_clustering(G)
        print(f"Average clustering coefficient: {avg_clustering:.4f}")
    else:
        print("Graph too large for clustering coefficient calculation")
    
    print("\n" + "=" * 60)
    print("EGO NETWORKS SUMMARY")
    print("=" * 60)
    for ego_id in ego_nodes:
        ego_neighbors = list(G.neighbors(ego_id))
        print(f"Ego {ego_id}: {len(ego_neighbors)} connections")
        
        if ego_id in circles:
            print(f"  Circles: {len(circles[ego_id])}")
    
    print("\n" + "=" * 60)
    print("FEATURE INFORMATION")
    print("=" * 60)
    print(f"Ego nodes with features: {len(features)}")
    if features:
        feature_lengths = [len(f) for f in features.values()]
        print(f"Feature vector length: {feature_lengths[0]} (all ego nodes)")

# Main execution
if __name__ == "__main__":
    G, all_ego_nodes, all_circles, all_features = create_complete_facebook_graph("d:\\aad\\dataset\\")
    print(G)
    visualize_complete_graph(G, all_ego_nodes)

    output_file = "d:\\aad\\facebook_graph.gpickle"
    nx.write_gpickle(G, output_file)
    print(f"Graph saved to: {output_file}")
    
    print("\nDone!")
