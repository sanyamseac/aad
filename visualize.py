import networkx as nx
import matplotlib.pyplot as plt
from graph import create_complete_graph

def visualize_complete_graph(G, ego_nodes):
    """Visualize the complete graph (sample if too large)"""
    plt.figure(figsize=(20, 20))
    
    pos = nx.spring_layout(G, k=1, iterations=30, seed=42)

    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in ego_nodes if n in G], node_color='red', node_size=300, label='Ego Nodes', alpha=0.8)

    alter_nodes = [n for n in G.nodes() if n not in ego_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=alter_nodes, node_color='lightblue', node_size=50, label='Alter Nodes', alpha=0.6)

    nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5)
    
    plt.title(f'Complete Facebook Ego Network Graph\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('complete_facebook_network.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    G, all_ego_nodes, all_circles, all_features = create_complete_graph()
    visualize_complete_graph(G, all_ego_nodes)

    output_file = "complete_facebook_graph.gpickle"
    nx.write_gpickle(G, output_file)
    print(f"Graph saved to: {output_file}")
