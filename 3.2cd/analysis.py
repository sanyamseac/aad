import os
import sys
import matplotlib.pyplot as plt
import networkx as nx

# Ensure imports from parent if graph.py is placed there
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph import create_complete_graph

# ------ Correct imports as you requested ------
from gn import girvan_newman        # Girvan-Newman
from lm import louvain_algorithm    # Louvain Modularity Method
from la import leiden_algorithm     # Leiden Algorithm
# ----------------------------------------------


def girvan_result(G, num_communities=5, max_iterations=20):
    """Convert Girvan-Newman result history to node->community mapping."""
    result = girvan_newman(G, num_communities=num_communities, max_iterations=max_iterations)

    if not result:  # No split happened
        return {node: 0 for node in G.nodes()}

    _, final_components, _, _ = result[-1]  # (iteration, components, removed_edge, betweenness)

    mapping = {}
    for cid, comp in enumerate(final_components):
        for node in comp:
            mapping[node] = cid

    return mapping


def draw_graph(G, mapping, ax, title):
    """Plot communities with different colors."""
    pos = nx.spring_layout(G, seed=42)  # stable layout

    comm_ids = sorted(set(mapping.values()))
    cmap = plt.cm.get_cmap("tab20", len(comm_ids))
    node_colors = [cmap(mapping[n]) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=40, node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)

    ax.set_title(title)
    ax.axis("off")


def main():
    print("Loading graph ...")
    G, _, _, _ = create_complete_graph(1)  # choose ego network index manually

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n--- Running Algorithms ---\n")

    print("[1] Girvan-Newman...")
    gn_comm = girvan_result(G, num_communities=5)

    print("[2] Louvain Modularity...")
    lv_comm, modularity = louvain_algorithm(G)
    print(f"   -> Modularity = {modularity:.5f}")

    print("[3] Leiden Algorithm...")
    ld_comm = leiden_algorithm(G)

    # Plot side-by-side comparison
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Community Structure Visualization", fontsize=16)

    draw_graph(G, gn_comm,  axs[0], "Girvan–Newman")
    draw_graph(G, lv_comm,  axs[1], "Louvain")
    draw_graph(G, ld_comm,  axs[2], "Leiden")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
