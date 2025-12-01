# runtime_analysis.py

import os
import sys
import time
import matplotlib.pyplot as plt

# Make sure we can import graph.py from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

# Import algorithms from the files you already have in the same folder
from gn import girvan_newman
from lm import louvain_algorithm
from la import leiden_algorithm


def measure_runtime_girvan_newman(G, max_iterations=5, num_communities=None):
    """
    Measure runtime of Girvan-Newman on graph G.
    We pass a small max_iterations to avoid extremely long runtimes.
    """
    start = time.perf_counter()
    girvan_newman(G, num_communities=num_communities, max_iterations=max_iterations)
    end = time.perf_counter()
    return end - start


def measure_runtime_louvain(G, max_iterations=50):
    """Measure runtime of Louvain algorithm on graph G."""
    start = time.perf_counter()
    louvain_algorithm(G, max_iterations=max_iterations)
    end = time.perf_counter()
    return end - start


def measure_runtime_leiden(G, max_iterations=10):
    """Measure runtime of Leiden algorithm on graph G."""
    start = time.perf_counter()
    leiden_algorithm(G, max_iterations=max_iterations)
    end = time.perf_counter()
    return end - start


def main():
    # ---- 1. Load a base graph from your dataset ----
    print("Loading graph from dataset...")
    # Choose whichever ego-id you want to analyse (1, 2, etc.)
    G_full, all_ego_nodes, all_circles, all_features = create_complete_graph(1)
    print(f"Graph loaded: {G_full}")
    print(f"Total nodes: {G_full.number_of_nodes()}, edges: {G_full.number_of_edges()}")

    # ---- 2. Decide which subgraph sizes to test ----
    max_nodes = G_full.number_of_nodes()

    # We'll take 5 sizes from small to full graph
    # e.g., [20%, 40%, 60%, 80%, 100%] of nodes (rounded and deduplicated)
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    node_sizes = sorted(set(max(5, int(max_nodes * f)) for f in fractions if int(max_nodes * f) >= 5))

    print(f"\nNode sizes to test: {node_sizes}\n")

    # ---- 3. Containers for runtimes ----
    gn_times = []
    louvain_times = []
    leiden_times = []

    # ---- 4. Measure runtimes for each subgraph size ----
    nodes_list = list(G_full.nodes())

    for n in node_sizes:
        print(f"--- Measuring for subgraph with {n} nodes ---")
        sub_nodes = nodes_list[:n]
        G_sub = G_full.subgraph(sub_nodes).copy()

        # Girvan-Newman (with limited iterations so it finishes in reasonable time)
        t_gn = measure_runtime_girvan_newman(G_sub, max_iterations=5)
        print(f"Girvan-Newman time: {t_gn:.4f} s")
        gn_times.append(t_gn)

        # Louvain
        t_louvain = measure_runtime_louvain(G_sub, max_iterations=50)
        print(f"Louvain time:       {t_louvain:.4f} s")
        louvain_times.append(t_louvain)

        # Leiden
        t_leiden = measure_runtime_leiden(G_sub, max_iterations=10)
        print(f"Leiden time:        {t_leiden:.4f} s\n")
        leiden_times.append(t_leiden)

    # ---- 5. Plot the runtimes on the same graph ----
    plt.figure(figsize=(8, 6))

    # Different colours so it's easy to analyse
    plt.plot(node_sizes, gn_times, marker='o', label='Girvan-Newman', color='tab:blue')
    plt.plot(node_sizes, louvain_times, marker='s', label='Louvain', color='tab:orange')
    plt.plot(node_sizes, leiden_times, marker='^', label='Leiden', color='tab:green')

    plt.xlabel("Number of nodes in subgraph")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Performance Comparison:\nGirvan-Newman vs Louvain vs Leiden")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
