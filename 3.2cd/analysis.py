import os
import sys
import time
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Setup results directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Increase recursion depth (from analysis5.py)
sys.setrecursionlimit(3000)

# Make parent folder (for graph.py) importable - Setup once for all sections
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local imports (Assuming these exist in the same folder or path)
from graph import create_complete_graph
from gn import girvan_newman
from lm import louvain_algorithm, calculate_modularity, calculate_modularity_gain
from la import leiden_algorithm, move_nodes_fast, refine_partition


# =============================================================================
# SECTION 1: MODULARITY TRAJECTORY (Originally analysis2.py)
# =============================================================================

def compute_girvan_newman_modularity(G, num_communities=5, max_iterations=20):
    """
    Run Girvan-Newman and compute modularity (using Louvain's calculate_modularity)
    for the partition at each iteration.
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


def compute_louvain_modularity_trajectory(G, max_iterations=20):
    """
    Reimplements the outer loop of louvain_algorithm, but records modularity
    after each iteration.
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


def compute_leiden_modularity_trajectory(G, max_iterations=10):
    """
    Stepwise Leiden-like process using move_nodes_fast and refine_partition
    from la.py, but tracking modularity after each iteration.
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


def run_modularity_comparison():
    print("\n" + "=" * 60)
    print("SECTION 1: Modularity Trajectories Comparison")
    print("=" * 60)
    
    # Load the same graph for all algorithms
    print("Loading graph from dataset...")
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(1)
    print(f"Graph loaded: {G}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}\n")

    # Girvan–Newman
    print("Calculating Girvan-Newman trajectory...")
    gn_iters, gn_mods = compute_girvan_newman_modularity(
        G,
        num_communities=5,
        max_iterations=20
    )

    # Louvain
    print("Calculating Louvain trajectory...")
    lv_iters, lv_mods = compute_louvain_modularity_trajectory(G, max_iterations=20)

    # Leiden
    print("Calculating Leiden trajectory (simplified)...")
    ld_iters, ld_mods = compute_leiden_modularity_trajectory(G, max_iterations=10)

    # Print final modularities for sanity
    if gn_mods:
        print(f"Girvan-Newman final modularity: {gn_mods[-1]:.6f}")
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

    # Save the figure to results directory
    file_name = os.path.join(RESULTS_DIR, "modularity_scores_comparison.png")
    plt.savefig(file_name)
    plt.close() # Close to prevent overlap
    print(f"Graph successfully saved as {file_name}")


# =============================================================================
# SECTION 2: COMMUNITY SIZE DISTRIBUTION (Originally analysis3.py)
# =============================================================================

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
    Given a list of sizes, returns (sorted_sizes, counts_list).
    """
    counter = Counter(sizes)
    sorted_sizes = sorted(counter.keys())
    counts = [counter[s] for s in sorted_sizes]
    return sorted_sizes, counts


def run_community_size_comparison():
    print("\n" + "=" * 60)
    print("SECTION 2: Community Size Distribution Comparison")
    print("=" * 60)

    # 1. Load the dataset graph
    print("Loading graph from dataset...")
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(1)
    
    # 2. Run Girvan-Newman
    print("Running Girvan-Newman Algorithm...")
    gn_history = girvan_newman(G, num_communities=None, max_iterations=20)
    gn_sizes = community_sizes_from_gn_history(gn_history)
    print(f"Girvan-Newman final community sizes: {gn_sizes}")

    # 3. Run Louvain
    print("Running Louvain Algorithm...")
    louvain_communities, louvain_modularity = louvain_algorithm(G)
    lv_sizes = community_sizes_from_louvain(louvain_communities)
    print(f"Louvain final modularity: {louvain_modularity:.6f}")
    print(f"Louvain community sizes: {lv_sizes}")

    # 4. Run Leiden
    print("Running Leiden Algorithm...")
    leiden_partition = leiden_algorithm(G, max_iterations=10)
    ld_sizes = community_sizes_from_leiden(leiden_partition)
    print(f"Leiden community sizes: {ld_sizes}")

    # 5. Compute size distributions
    gn_x, gn_y = get_size_distribution(gn_sizes) if gn_sizes else ([], [])
    lv_x, lv_y = get_size_distribution(lv_sizes) if lv_sizes else ([], [])
    ld_x, ld_y = get_size_distribution(ld_sizes) if ld_sizes else ([], [])

    # 6. Plot on the same graph
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

    # Save the figure to results directory
    file_name = os.path.join(RESULTS_DIR, "community_size_distribution.png")
    plt.savefig(file_name)
    plt.close()
    print(f"Graph successfully saved as {file_name}")


# =============================================================================
# SECTION 3: VISUALIZATION (Originally analysis.py)
# =============================================================================

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


def draw_graph_vis(G, mapping, ax, title):
    """Plot communities with different colors."""
    pos = nx.spring_layout(G, seed=42)  # stable layout

    comm_ids = sorted(set(mapping.values()))
    cmap = plt.cm.get_cmap("tab20", len(comm_ids))
    node_colors = [cmap(mapping[n]) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=40, node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)

    ax.set_title(title)
    ax.axis("off")


def run_visual_structure_analysis():
    print("\n" + "=" * 60)
    print("SECTION 3: Community Structure Visualization")
    print("=" * 60)
    
    print("Loading graph ...")
    G, _, _, _ = create_complete_graph(1)  # choose ego network index manually

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Running Algorithms for visualization...")

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

    draw_graph_vis(G, gn_comm,  axs[0], "Girvan–Newman")
    draw_graph_vis(G, lv_comm,  axs[1], "Louvain")
    draw_graph_vis(G, ld_comm,  axs[2], "Leiden")

    plt.tight_layout()
    
    # Save the figure to results directory
    file_name = os.path.join(RESULTS_DIR, "community_structure_visualisation.png")
    plt.savefig(file_name)
    plt.close()
    print(f"Graph successfully saved as {file_name}")


# =============================================================================
# SECTION 4: RUNTIME ANALYSIS (Originally analysis4.py)
# =============================================================================

def measure_runtime_girvan_newman(G, max_iterations=5, num_communities=None):
    """Measure runtime of Girvan-Newman on graph G."""
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


def run_runtime_analysis():
    print("\n" + "=" * 60)
    print("SECTION 4: Runtime Performance Comparison")
    print("=" * 60)
    
    # 1. Load a base graph from your dataset
    print("Loading graph from dataset...")
    G_full, all_ego_nodes, all_circles, all_features = create_complete_graph(1)
    
    # 2. Decide which subgraph sizes to test
    max_nodes = G_full.number_of_nodes()

    # We'll take 5 sizes from small to full graph
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    node_sizes = sorted(set(max(5, int(max_nodes * f)) for f in fractions if int(max_nodes * f) >= 5))

    print(f"\nNode sizes to test: {node_sizes}\n")

    # 3. Containers for runtimes
    gn_times = []
    louvain_times = []
    leiden_times = []

    # 4. Measure runtimes for each subgraph size
    nodes_list = list(G_full.nodes())

    for n in node_sizes:
        print(f"--- Measuring for subgraph with {n} nodes ---")
        sub_nodes = nodes_list[:n]
        G_sub = G_full.subgraph(sub_nodes).copy()

        # Girvan-Newman (with limited iterations)
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

    # 5. Plot the runtimes on the same graph
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

    # Save the figure to results directory
    file_name = os.path.join(RESULTS_DIR, "runtime_performance_analysis.png")
    plt.savefig(file_name)
    plt.close()
    print(f"Graph successfully saved as {file_name}")


# =============================================================================
# SECTION 5: COMPLEXITY BENCHMARK (Originally analysis5.py)
# =============================================================================

def generate_random_graph(n, p=0.2):
    """
    Generates a random Erdos-Renyi graph.
    """
    while True:
        G = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(G):
            return G

def benchmark_and_plot(alg_func, sizes, complexity_lambda, alg_name, complexity_label, ax, **kwargs):
    """
    Runs benchmark and plots Theoretical vs Actual time with connected lines.
    """
    actual_times = []
    theoretical_values = []
    
    print(f"Benchmarking {alg_name}...")
    print(f"  [Progress]: ", end="", flush=True)

    for i, n in enumerate(sizes):
        # Create a graph
        G = generate_random_graph(n)
        m = G.number_of_edges()
        
        # Measure Actual Time
        start_time = time.time()
        
        # Call the algorithm
        alg_func(G, **kwargs)
            
        end_time = time.time()
        duration = end_time - start_time
        actual_times.append(duration)
        
        # Calculate Theoretical Complexity Value
        theo_val = complexity_lambda(n, m)
        theoretical_values.append(theo_val)
        
        # Print a dot for progress so you know it's not frozen
        print(".", end="", flush=True)
        
    print(" Done!")

    # --- Plotting ---
    
    # Sort points by theoretical value
    sorted_pairs = sorted(zip(theoretical_values, actual_times))
    if not sorted_pairs:
        return
    theo_sorted, time_sorted = zip(*sorted_pairs)
    
    # Plot Data Trace (Connected Dots)
    ax.plot(theo_sorted, time_sorted, color='blue', alpha=0.6, marker='o', linestyle='-', linewidth=1, markersize=4, label='Data Trace')
    
    # Linear Regression Trend Line
    if len(theoretical_values) > 1:
        slope, intercept = np.polyfit(theoretical_values, actual_times, 1)
        line_x = np.array([min(theoretical_values), max(theoretical_values)])
        line_y = slope * line_x + intercept
        
        # Calculate R-squared
        correlation_matrix = np.corrcoef(theoretical_values, actual_times)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy ** 2
        
        ax.plot(line_x, line_y, 'r--', label=f'Trend Fit (R²={r_squared:.3f})')

    ax.set_title(f'{alg_name}', fontweight='bold')
    ax.set_xlabel(f'Theoretical\n{complexity_label}')
    ax.set_ylabel('Actual Time (s)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

def run_complexity_benchmark():
    print("\n" + "=" * 60)
    print("SECTION 5: Theoretical Complexity vs Actual Time")
    print("=" * 60)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Theoretical Complexity vs Actual Execution Time', fontsize=16, y=1.05)
    
    # 1. Girvan-Newman (The Bottleneck)
    gn_sizes = range(5, 22, 1) 
    
    benchmark_and_plot(
        alg_func=girvan_newman,
        sizes=gn_sizes,
        complexity_lambda=lambda n, m: (m**2) * n,
        alg_name="Girvan-Newman",
        complexity_label=r"$O(m^2 n)$",
        ax=ax1,
        num_communities=2 # Stop after first split to keep it fast
    )

    # 2. Louvain (Fast)
    lm_sizes = range(10, 201, 5)
    
    benchmark_and_plot(
        alg_func=louvain_algorithm,
        sizes=lm_sizes,
        complexity_lambda=lambda n, m: n * np.log(n),
        alg_name="Louvain",
        complexity_label=r"$O(n \log n)$",
        ax=ax2
    )

    # 3. Leiden (Fast)
    la_sizes = range(10, 201, 5)
    
    benchmark_and_plot(
        alg_func=leiden_algorithm,
        sizes=la_sizes,
        complexity_lambda=lambda n, m: n * np.log(n),
        alg_name="Leiden",
        complexity_label=r"$O(n \log n)$",
        ax=ax3
    )

    plt.tight_layout()
    file_name = os.path.join(RESULTS_DIR, 'final_benchmark_plot.png')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    print(f"\nBenchmark Complete! Results saved to '{file_name}'")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Execute all analyses in sequence
    run_modularity_comparison()
    run_community_size_comparison()
    run_visual_structure_analysis()
    run_runtime_analysis()
    run_complexity_benchmark()