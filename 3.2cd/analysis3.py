import os
import sys
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Make parent directory (with graph.py) importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your graph creator and algorithms from the existing files
from graph import create_complete_graph
from gn import girvan_newman
from lm import louvain_algorithm
from la import leiden_algorithm, modularity  # modularity not strictly needed, but available


def community_sizes_from_gn_history(history):
    """
    Takes the history list from girvan_newman() and returns a list of community sizes
    for the final partition (last iteration).
    """
    if not history:
        return []

    # history is a list of tuples: (iteration, components, removed_edge, betweenness_score)
    final_iteration, final_components, _, _ = history[-1]
    sizes = [len(component) for component in final_components]
    return sizes


def community_sizes_from_louvain(communities_dict):
    """
    communities_dict: node -> community_id
    Returns a list of community sizes.
    """
    comm_to_nodes = defaultdict(list)
    for node, comm in communities_dict.items():
        comm_to_nodes[comm].append(node)
    sizes = [len(nodes) for nodes in comm_to_nodes.values()]
    return sizes


def community_sizes_from_leiden(partition_dict):
    """
    partition_dict: node -> community_id (from leiden_algorithm)
    Returns a list of community sizes.
    """
    comm_to_nodes = defaultdict(list)
    for node, comm in partition_dict.items():
        comm_to_nodes[comm].append(node)
    sizes = [len(nodes) for nodes in comm_to_nodes.values()]
    return sizes


def get_size_distribution(sizes):
    """
    Given a list of sizes, returns (sorted_sizes, counts_list),
    where sorted_sizes are the unique community sizes and
    counts_list[i] is the number of communities with size sorted_sizes[i].
    """
    counter = Counter(sizes)
    sorted_sizes = sorted(counter.keys())
    counts = [counter[s] for s in sorted_sizes]
    return sorted_sizes, counts


def main():
    # -----------------------------
    # 1. Load the dataset graph
    # -----------------------------
    print("Loading graph from dataset...")
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

    # Use the same graph for all three algorithms for fair comparison.
    # You can change the parameter (e.g., 1, 2, etc.) to pick which ego graph you want.
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(1)
    print(f"Graph loaded: {G}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}\n")

    # -----------------------------
    # 2. Run Girvan-Newman
    # -----------------------------
    print("=" * 60)
    print("Running Girvan-Newman Algorithm")
    print("=" * 60)
    # You can tune num_communities / max_iterations if you want
    gn_history = girvan_newman(G, num_communities=None, max_iterations=20)
    gn_sizes = community_sizes_from_gn_history(gn_history)
    print(f"Girvan-Newman final community sizes: {gn_sizes}\n")

    # -----------------------------
    # 3. Run Louvain
    # -----------------------------
    print("=" * 60)
    print("Running Louvain Algorithm")
    print("=" * 60)
    louvain_communities, louvain_modularity = louvain_algorithm(G)
    lv_sizes = community_sizes_from_louvain(louvain_communities)
    print(f"Louvain final modularity: {louvain_modularity:.6f}")
    print(f"Louvain community sizes: {lv_sizes}\n")

    # -----------------------------
    # 4. Run Leiden
    # -----------------------------
    print("=" * 60)
    print("Running Leiden Algorithm")
    print("=" * 60)
    leiden_partition = leiden_algorithm(G, max_iterations=10)
    ld_sizes = community_sizes_from_leiden(leiden_partition)
    print(f"Leiden community sizes: {ld_sizes}\n")

    # -----------------------------
    # 5. Compute size distributions
    # -----------------------------
    gn_x, gn_y = get_size_distribution(gn_sizes) if gn_sizes else ([], [])
    lv_x, lv_y = get_size_distribution(lv_sizes) if lv_sizes else ([], [])
    ld_x, ld_y = get_size_distribution(ld_sizes) if ld_sizes else ([], [])

    # -----------------------------
    # 6. Plot on the same graph
    # -----------------------------
    plt.figure(figsize=(10, 6))

    if gn_x:
        plt.plot(gn_x, gn_y, marker='o', label='Girvan-Newman')
    if lv_x:
        plt.plot(lv_x, lv_y, marker='s', label='Louvain')
    if ld_x:
        plt.plot(ld_x, ld_y, marker='^', label='Leiden')

    plt.xlabel("Community Size (number of nodes)")
    plt.ylabel("Number of Communities")
    plt.title("Community Size Distribution Comparison\n(Girvan-Newman vs Louvain vs Leiden)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
# ---- NEW CODE START ----
    # Define your filename
    file_name = "community_size_distribution.png" 
    
    # Save the figure to the current directory
    plt.savefig(file_name)
    print(f"Graph successfully saved as {file_name}")
    # ---- NEW CODE END ----


if __name__ == "__main__":
    main()
