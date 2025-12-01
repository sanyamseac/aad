import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Include requested imports for dataset access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

# Import local algorithms
from bfs import bfs_traversal
from dfs import dfs_traversal
from ufa_by_rank import run_ufa_rank
from ufa_by_size import run_ufa_size

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, 'plots')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis')
for d in [PLOT_DIR, ANALYSIS_DIR]: os.makedirs(d, exist_ok=True)
for d in ['bfs', 'dfs', 'ufa_rank', 'ufa_size']: os.makedirs(os.path.join(PLOT_DIR, d), exist_ok=True)

# =====================================================
# METRIC FUNCTIONS FOR SPECIAL INSIGHTS INTO THE GRAPH
# =====================================================

def calculate_density(V, E):
    """Calculates graph density (2E / V(V-1)).
    Args:
        V: Number of nodes.
        E: Number of edges.
    Returns:
        Density in [0,1]. 0 if V < 2.
    """
    if V < 2: return 0
    return (2 * E) / (V * (V - 1))

def calculate_clustering_coefficient(G):
    """Computes average clustering coefficient from scratch.
    Args:
        G: Graph object with adjacency list G.adj.
    Returns:
        Average clustering coefficient across all nodes.
    """
    total_coef = 0
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0: return 0
    
    for node in nodes:
        neighbors = list(G.adj[node])
        k = len(neighbors)
        if k < 2:
            total_coef += 0
            continue
            
        links = 0
        # Check connections between neighbors
        for i in range(k):
            for j in range(i + 1, k):
                if neighbors[j] in G.adj[neighbors[i]]:
                    links += 1
        
        total_coef += (2.0 * links) / (k * (k - 1))
        
    return total_coef / n

def calculate_path_metrics(G, component_nodes):
    """Estimates diameter and average shortest path via sampled BFS.
    Args:
        G: Graph object.
        component_nodes: Iterable of nodes forming the component.
    Returns:
        (int, float): (Estimated diameter, estimated average path length).
    """
    nodes = list(component_nodes)
    n = len(nodes)
    if n < 2: return 0, 0

    # Sample source nodes to estimate metrics (Full O(V*(V+E)) is too slow)
    sample_size = min(100, n)
    sources = random.sample(nodes, sample_size)
    
    max_eccentricity = 0
    total_path_sum = 0
    total_pairs = 0
    
    for start_node in sources:
        # BFS for Shortest Paths from start_node
        q = deque([(start_node, 0)])
        visited = {start_node}
        local_max_dist = 0
        
        while q:
            curr, dist = q.popleft()
            local_max_dist = max(local_max_dist, dist)
            total_path_sum += dist 
            
            for neighbor in G.adj[curr]:
                if neighbor in component_nodes and neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, dist + 1))
        
        max_eccentricity = max(max_eccentricity, local_max_dist)
        total_pairs += (len(visited) - 1)
    
    # Avg Path = Total Distance / Pairs Checked
    avg_path = total_path_sum / total_pairs if total_pairs > 0 else 0
    
    return max_eccentricity, avg_path

def get_giant_component_nodes(G):
    """Returns list of nodes in an approximate giant component.
    Args:
        G: Graph object.
    Returns:
        List of node ids for largest discovered component (sampled BFS).
    """
    visited = set()
    best_comp = set()
    # Sample heuristic for speed
    sample = list(G.nodes())[:50] if len(G.nodes()) > 50 else list(G.nodes())
    for node in sample:
        if node not in visited:
            comp, _, _ = bfs_traversal(G, node, set())
            visited.update(comp)
            if len(comp) > len(best_comp): best_comp = comp
    return list(best_comp)

# =====================================================
# ANALYSIS A: Start Node Invariance
# =====================================================
def run_analysis_A():
    """Runs Analysis A: evaluates start-node invariance for BFS/DFS.
    What: Measure how traversal start node within the giant component affects runtime.
    How:
        1. Build aggregated graph from 10 dataset files.
        2. Approximate giant component nodes via sampled BFS expansions.
        3. Perform 100 runs each for BFS and DFS choosing random start nodes.
        4. Record per-run times; compute mean, std dev, coefficient of variation.
        5. Plot time series with mean line -> plots/<algo>/A_Invariance.png.
        6. Write summary table -> analysis/A_Invariance.md.
    Args: None
    Returns: None (side effects: creates plot files and markdown summary).
    Results: Demonstrates low variance ⇒ algorithmic stability w.r.t. start node.
    """
    print("\n[A] Running Start Node Invariance (100 runs)...")
    G, _, _, _ = create_complete_graph(num_files=10)
    giant_nodes = get_giant_component_nodes(G)
    
    results = {"bfs": [], "dfs": []}
    NUM_RUNS = 100
    
    for algo in ["bfs", "dfs"]:
        func = bfs_traversal if algo == "bfs" else dfs_traversal
        times = []
        for _ in range(NUM_RUNS):
            start = random.choice(giant_nodes)
            t0 = time.perf_counter()
            func(G, start, visited=set())
            times.append(time.perf_counter() - t0)
        results[algo] = times
        
        plt.figure(figsize=(10, 5))
        plt.plot(times, 'o-', alpha=0.6, markersize=4, label='Run Time')
        plt.axhline(np.mean(times), color='r', linestyle='--', label='Mean')
        plt.title(f'Analysis A: {algo.upper()} Start Node Invariance')
        plt.xlabel('Run ID'); plt.ylabel('Time (s)')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(PLOT_DIR, algo, 'A_Invariance.png'))
        plt.close()

    with open(os.path.join(ANALYSIS_DIR, 'A_Invariance.md'), 'w') as f:
        f.write("# Analysis A: Start Node Invariance\n\n")
        f.write("| Algo | Mean (s) | StdDev (s) | Coeff Var |\n|---|---|---|---|\n")
        for algo in ["bfs", "dfs"]:
            mu, sigma = np.mean(results[algo]), np.std(results[algo])
            f.write(f"| {algo.upper()} | {mu:.5f} | {sigma:.5f} | {(sigma/mu)*100:.2f}% |\n")

# =====================================================
# ANALYSIS B: Time Complexity
# =====================================================
def run_analysis_B():
    """Runs Analysis B: examines scaling of algorithms as graph grows.
    What: Compare BFS, DFS, Union-Find (rank/size) runtime over incremental file loads.
    How:
        For i = 1..10 files:
          1. Construct cumulative graph.
          2. Capture V (nodes) and E (edges).
          3. Time BFS, DFS, union-find by rank, union-find by size once each.
          4. Store per-file stats.
        Generate 2x2 metric grids (Time vs V, E, V+E, V*E) in plots/<algo>/B_Complexity_Grid.png.
        Write consolidated table to analysis/B_Complexity.md.
    Args: None
    Returns: None (side effects: metric grid plots + markdown table).
    Results: Empirically validates near-linear behavior in V+E for traversals; ~linear edge pass for union-find.
    """
    print("\n[B] Running Time Complexity (1-10 files)...")
    stats = []
    for i in range(1, 11):
        G, _, _, _ = create_complete_graph(num_files=i)
        V, E = G.number_of_nodes(), G.number_of_edges()
        start = list(G.nodes())[0]
        
        times = {}
        # BFS / DFS
        for algo, func in [('bfs', bfs_traversal), ('dfs', dfs_traversal)]:
            t0 = time.perf_counter()
            func(G, start, set())
            times[algo] = time.perf_counter() - t0
        
        # UFA Rank
        t0 = time.perf_counter()
        run_ufa_rank(G)
        times['ufa_rank'] = time.perf_counter() - t0

        # UFA Size
        t0 = time.perf_counter()
        run_ufa_size(G)
        times['ufa_size'] = time.perf_counter() - t0
        
        stats.append({'V': V, 'E': E, 'VE': V+E, 'VxE': V*E, **times})

    metrics = [('V', 'Nodes'), ('E', 'Edges'), ('VE', 'V + E'), ('VxE', 'V * E')]
    for algo in ['bfs', 'dfs', 'ufa_rank', 'ufa_size']:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{algo.upper()} Time Complexity', fontsize=16)
        y_vals = [s[algo] for s in stats]
        
        for idx, (key, label) in enumerate(metrics):
            ax = axs[idx//2, idx%2]
            x_vals = [s[key] for s in stats]
            xy = sorted(zip(x_vals, y_vals))
            ax.plot([p[0] for p in xy], [p[1] for p in xy], 'o-')
            ax.set_title(f'Time vs {label}'); ax.grid(True)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(PLOT_DIR, algo, 'B_Complexity_Grid.png'))
        plt.close()

    with open(os.path.join(ANALYSIS_DIR, 'B_Complexity.md'), 'w') as f:
        f.write("# Analysis B: Time Complexity\n\n| Files | V | E | BFS(s) | DFS(s) | UFA Rank(s) | UFA Size(s) |\n|---|---|---|---|---|---|---|\n")
        for i, s in enumerate(stats):
            f.write(f"| {i+1} | {s['V']} | {s['E']} | {s['bfs']:.4f} | {s['dfs']:.4f} | {s['ufa_rank']:.4f} | {s['ufa_size']:.4f} |\n")

# =====================================================
# ANALYSIS C: Order Invariance
# =====================================================
def run_analysis_C():
    """Runs Analysis C: tests order invariance under shuffled adjacency/edge lists.
    What: Determine sensitivity of BFS, DFS, and union-find variants to input ordering.
    How:
        1. Build full graph (10 files).
        2. Pre-compute normal vs shuffled adjacency (and edge list for union-find) to isolate algorithm cost.
        3. Run each algorithm 10 repetitions with normal and shuffled ordering; average times.
        4. Produce bar chart per algorithm -> plots/<algo>/C_Order_Invariance.png.
        5. Log table with normal, shuffled, absolute diff -> analysis/C_Order.md.
    Args: None
    Returns: None (side effects: bar plots + markdown table).
    Results: Small deltas indicate robust order invariance (expected for these algorithms beyond cache effects).
    """
    print("\n[C] Running Order Invariance...")
    G, _, _, _ = create_complete_graph(num_files=10)
    start = list(G.nodes())[0]
    
    # Pre-calculate shuffled structures to exclude overhead
    adj_norm = {n: list(G.adj[n]) for n in G}
    adj_shuf = {n: random.sample(list(G.adj[n]), len(G.adj[n])) for n in G}
    edges = []
    for u in G.adj:
        for v in G.adj[u]:
            if u < v: edges.append((u,v))
    edges_shuf = random.sample(edges, len(edges))
    
    results = {}
    # BFS / DFS
    for name, func in [('bfs', bfs_traversal), ('dfs', dfs_traversal)]:
        t0 = time.perf_counter(); [func(G, start, set(), custom_adj=adj_norm) for _ in range(10)]
        t_n = (time.perf_counter()-t0)/10
        t0 = time.perf_counter(); [func(G, start, set(), custom_adj=adj_shuf) for _ in range(10)]
        t_s = (time.perf_counter()-t0)/10
        results[name] = (t_n, t_s)

    # UFA Variants
    for name, func in [('ufa_rank', run_ufa_rank), ('ufa_size', run_ufa_size)]:
        t0 = time.perf_counter(); [func(G, custom_edges=edges) for _ in range(10)]
        t_n = (time.perf_counter()-t0)/10
        t0 = time.perf_counter(); [func(G, custom_edges=edges_shuf) for _ in range(10)]
        t_s = (time.perf_counter()-t0)/10
        results[name] = (t_n, t_s)

    for algo, (n, s) in results.items():
        plt.figure(figsize=(5, 4))
        plt.bar(['Normal', 'Shuffled'], [n, s], color=['#1f77b4', '#ff7f0e'])
        plt.title(f'{algo.upper()} Order Invariance')
        plt.savefig(os.path.join(PLOT_DIR, algo, 'C_Order_Invariance.png'))
        plt.close()

    with open(os.path.join(ANALYSIS_DIR, 'C_Order.md'), 'w') as f:
        f.write("# Analysis C: Order Invariance\n\n| Algo | Normal(s) | Shuffled(s) | Diff |\n|---|---|---|---|\n")
        for k, (n, s) in results.items():
            f.write(f"| {k.upper()} | {n:.5f} | {s:.5f} | {abs(n-s):.5f}s |\n")

# =====================================================
# ANALYSIS D: COMPREHENSIVE CONNECTIVITY METRICS
# =====================================================
def run_full_connectivity_analysis():
    """Runs Analysis D: tracks evolving connectivity & structural metrics (1..10 files).
    What: Observe how global + giant component metrics change as more ego networks are merged.
    How:
        For i = 1..10 files:
          1. Build cumulative graph.
          2. Compute density, average degree, clustering coefficient (scratch implementation).
          3. Enumerate connected components via BFS; identify giant component.
          4. Compute coverage %, internal edges, component counts.
          5. Estimate diameter & average path length using sampled BFS on GC.
          6. Append structured log to analysis/D_Connectivity_Analysis.md.
        After loop: generate evolution plots for each metric -> plots/Metric_<name>.png.
    Args: None
    Returns: None (side effects: 10 metric plots + detailed markdown log).
    Results: Provides empirical growth trends informing friend recommendation & network structure insights.
    """
    print("\n[D] Running Comprehensive Network Metrics Analysis (1-10 files)...")
    
    history = {
        "files": [], "nodes": [], "edges": [], "density": [], "avg_degree": [],
        "clustering": [], "num_components": [], "gc_size_nodes": [], "gc_size_edges": [],
        "gc_coverage": [], "diameter": [], "avg_path": []
    }

    report_path = os.path.join(ANALYSIS_DIR, 'D_Connectivity_Analysis.md')
    with open(report_path, 'w') as f:
        f.write("# Comprehensive Network Connectivity Analysis\n\n")
        f.write("## Metric Definitions\n")
        f.write("- **Density:** Ratio of actual edges to possible edges.\n")
        f.write("- **Clustering Coefficient:** Measure of the degree to which nodes cluster together.\n")
        f.write("- **Giant Component (GC):** The largest connected subgraph.\n")
        f.write("- **Diameter:** Longest shortest path in the GC.\n\n")
        f.write("## Detailed Data Log\n\n")

    for i in range(1, 11):
        print(f"  > Processing File {i}/10...")
        G, _, _, _ = create_complete_graph(num_files=i)
        
        V = G.number_of_nodes()
        E = G.number_of_edges()
        
        # Calculate Global Metrics (From Scratch)
        density = calculate_density(V, E)
        avg_degree = (2 * E) / V if V > 0 else 0
        clustering = calculate_clustering_coefficient(G)
        
        # Component Identification (BFS)
        nodes = list(G.nodes())
        visited = set()
        components = []
        for node in nodes:
            if node not in visited:
                comp, edges, _ = bfs_traversal(G, node, visited)
                components.append({'nodes': len(comp), 'edges': edges, 'node_set': comp})
        
        num_comp = len(components)
        
        # Giant Component Metrics
        if components:
            components.sort(key=lambda x: x['nodes'], reverse=True)
            giant = components[0]
            gc_size_n = giant['nodes']
            gc_size_e = giant['edges']
            gc_cov = (gc_size_n / V) * 100 if V > 0 else 0
            
            # Path Metrics (on GC via Sampling BFS)
            diameter, avg_path = calculate_path_metrics(G, giant['node_set'])
        else:
            gc_size_n, gc_size_e, gc_cov, diameter, avg_path = 0, 0, 0, 0, 0

        # Record History
        history["files"].append(i)
        history["nodes"].append(V)
        history["edges"].append(E)
        history["density"].append(density)
        history["avg_degree"].append(avg_degree)
        history["clustering"].append(clustering)
        history["num_components"].append(num_comp)
        history["gc_size_nodes"].append(gc_size_n)
        history["gc_size_edges"].append(gc_size_e)
        history["gc_coverage"].append(gc_cov)
        history["diameter"].append(diameter)
        history["avg_path"].append(avg_path)

        with open(report_path, 'a') as f:
            f.write(f"### Step {i} (Files 1-{i})\n")
            f.write(f"- **Structure:** {V:,} Nodes, {E:,} Edges\n")
            f.write(f"- **Connectivity:** {num_comp} Component(s), GC covers {gc_cov:.2f}%\n")
            f.write(f"- **Density:** {density:.6f} | **Avg Degree:** {avg_degree:.2f}\n")
            f.write(f"- **Clustering Coeff:** {clustering:.4f}\n")
            f.write(f"- **GC Diameter:** {diameter} | **GC Avg Path:** {avg_path:.4f}\n\n")

    print("  > Generating Metric Plots...")
    plot_specs = [
        ("nodes", "Nodes (V)", "Count", "linear"),
        ("edges", "Edges (E)", "Count", "linear"),
        ("density", "Graph Density", "Density", "linear"),
        ("avg_degree", "Average Degree", "Connections", "linear"),
        ("clustering", "Clustering Coefficient", "Value", "linear"),
        ("num_components", "Connected Components", "Count", "linear"),
        ("gc_size_nodes", "Giant Component Size (Nodes)", "Nodes", "linear"),
        ("gc_coverage", "GC Coverage", "Percent", "linear"),
        ("diameter", "Network Diameter (Est)", "Network Diameter", "linear"), # Updated Label
        ("avg_path", "Avg Path Length (Est)", "Avg Path Length", "linear")   # Updated Label
    ]

    for key, title, ylabel, scale in plot_specs:
        plt.figure(figsize=(8, 6))
        plt.plot(history["files"], history[key], 'o-', linewidth=2, color='teal')
        plt.title(f'Evolution of {title}')
        plt.xlabel('Number of Dataset Files')
        plt.ylabel(ylabel)
        plt.yscale(scale)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(range(1, 11))
        plt.savefig(os.path.join(PLOT_DIR, f'Metric_{key}.png'))
        plt.close()

    print(f"  > Saved 10 metric plots to {PLOT_DIR}")
    print(f"  > Saved detailed report to {report_path}")

if __name__ == "__main__":
    run_analysis_A()
    run_analysis_B()
    run_analysis_C()
    run_full_connectivity_analysis()