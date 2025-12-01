import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

# Make parent folder (for graph.py) importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local imports (same folder as this script)
from graph import create_complete_graph

# Girvan-Newman
from gn import girvan_newman

# Louvain pieces
from lm import calculate_modularity, calculate_modularity_gain

# Leiden pieces
from la import move_nodes_fast, refine_partition


# ---------- GIRVAN–NEWMAN: MODULARITY TRAJECTORY ----------

def compute_girvan_newman_modularity(G, num_communities=5, max_iterations=20):
    """
    Run Girvan-Newman and compute modularity (using Louvain's calculate_modularity)
    for the partition at each iteration.

    Returns:
        iters_gn: list of iteration indices
        mods_gn: list of modularity values
    """
    history = girvan_newman(G, num_communities=num_communities, max_iterations=max_iterations)

    iters_gn = []
    mods_gn = []

    for iteration, components, removed_edge, betweenness_score in history:
        # components: list[set] -> node -> community_id mapping
        communities = {}
        for cid, comp in enumerate(components):
            for node in comp:
                communities[node] = cid

        Q = calculate_modularity(G, communities)
        iters_gn.append(iteration)
        mods_gn.append(Q)

    return iters_gn, mods_gn


# ---------- LOUVAIN: STEPWISE VERSION (USING lm.py HELPERS) ----------

def compute_louvain_modularity_trajectory(G, max_iterations=20):
    """
    Reimplements the outer loop of louvain_algorithm, but records modularity
    after each iteration.

    Uses calculate_modularity and calculate_modularity_gain from lm.py.

    Returns:
        iters_lv: list of iteration indices (starting from 1)
        mods_lv: list of modularity values
    """
    m = G.number_of_edges()
    if m == 0:
        return [0], [0.0]

    # Initial: each node in its own community
    communities = {node: node for node in G.nodes()}

    iters_lv = []
    mods_lv = []

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # Phase 1: same as in lm.py but we’re inside this function
        for node in G.nodes():
            current_community = communities[node]
            best_community = current_community
            best_gain = 0.0

            old_community = communities[node]

            # Neighboring communities
            neighbor_communities = set()
            for neighbor in G.neighbors(node):
                neighbor_communities.add(communities[neighbor])

            # Try moving node to each neighboring community
            for community in neighbor_communities:
                communities[node] = community
                gain = calculate_modularity_gain(G, node, old_community,
                                                 community, communities, m)

                if gain > best_gain:
                    best_gain = gain
                    best_community = community

            # Final move decision
            if best_community != old_community:
                communities[node] = best_community
                improved = True
            else:
                communities[node] = old_community

        # Record modularity after this full pass
        Q = calculate_modularity(G, communities)
        iters_lv.append(iteration)
        mods_lv.append(Q)

        # Optional: stop if no improvement in modularity
        if iteration > 1 and abs(mods_lv[-1] - mods_lv[-2]) < 1e-9:
            break

    return iters_lv, mods_lv


# ---------- LEIDEN: STEPWISE VERSION (USING la.py HELPERS) ----------

def compute_leiden_modularity_trajectory(G, max_iterations=10):
    """
    Stepwise Leiden-like process using move_nodes_fast and refine_partition
    from la.py, but tracking modularity after each iteration.

    NOTE: This is a simplified version (no graph aggregation step) but gives
    a non-constant modularity curve for comparison.

    Returns:
        iters_ld: list of iteration indices (starting from 0)
        mods_ld: list of modularity values
    """
    # Start: each node has its own community id
    partition = {node: i for i, node in enumerate(G.nodes())}

    iters_ld = []
    mods_ld = []

    # Record initial modularity (iteration 0)
    Q0 = calculate_modularity(G, partition)
    iters_ld.append(0)
    mods_ld.append(Q0)

    for iteration in range(1, max_iterations + 1):
        # Phase 1: Fast local moving
        partition = move_nodes_fast(G, partition, randomize=True)

        # Phase 2: Refinement
        partition = refine_partition(G, partition)

        # Convert partition (node -> comm_id) directly to modularity
        Q = calculate_modularity(G, partition)

        iters_ld.append(iteration)
        mods_ld.append(Q)

        # Convergence check
        if abs(mods_ld[-1] - mods_ld[-2]) < 1e-6:
            break

    return iters_ld, mods_ld


# ---------- MAIN: RUN & PLOT ALL THREE ----------

def main():
    # Load the same graph for all algorithms
    print("Loading graph from dataset...")
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(1)
    print(f"Graph loaded: {G}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}\n")

    # Girvan–Newman
    print("=" * 60)
    print("Girvan-Newman trajectory")
    print("=" * 60)
    gn_iters, gn_mods = compute_girvan_newman_modularity(
        G,
        num_communities=5,
        max_iterations=20
    )

    # Louvain
    print("\n" + "=" * 60)
    print("Louvain trajectory")
    print("=" * 60)
    lv_iters, lv_mods = compute_louvain_modularity_trajectory(G, max_iterations=20)

    # Leiden
    print("\n" + "=" * 60)
    print("Leiden trajectory (simplified)")
    print("=" * 60)
    ld_iters, ld_mods = compute_leiden_modularity_trajectory(G, max_iterations=10)

    # Print final modularities for sanity
    if gn_mods:
        print(f"\nGirvan-Newman final modularity: {gn_mods[-1]:.6f}")
    if lv_mods:
        print(f"Louvain final modularity: {lv_mods[-1]:.6f}")
    if ld_mods:
        print(f"Leiden final modularity: {ld_mods[-1]:.6f}")

    # ---- Plot all on the same graph ----
    plt.figure(figsize=(9, 5))

    # Use algorithm-specific iteration indices on x-axis
    plt.plot(gn_iters, gn_mods, marker='o', label="Girvan-Newman")
    plt.plot(lv_iters, lv_mods, marker='s', label="Louvain")
    plt.plot(ld_iters, ld_mods, marker='^', label="Leiden")

    plt.xlabel("Iteration")
    plt.ylabel("Modularity Q")
    plt.title("Modularity Trajectories: Girvan-Newman vs Louvain vs Leiden")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
