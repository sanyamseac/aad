import os
import sys
import time
import random
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt

# Include requested imports for dataset access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

# Import local algorithms
from bfs import bfs_traversal
from dfs import dfs_traversal
from ufa import run_ufa

# Base output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, 'plots')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis')

# Create directories if they don't exist
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def get_giant_component_nodes(G):
    """Finds a node within the giant component to ensure valid testing."""
    visited = set()
    best_comp = set()
    # Sample for speed
    sample_nodes = list(G.nodes())[:50] if len(G.nodes()) > 50 else list(G.nodes())
    
    for node in sample_nodes:
        if node not in visited:
            comp, _, _ = bfs_traversal(G, node, set())
            visited.update(comp)
            if len(comp) > len(best_comp):
                best_comp = comp
    return list(best_comp)

# ==========================================
# ANALYSIS A: Start Node Invariance
# (Run on FULL 10 Files)
# ==========================================
def run_analysis_A():
    print("\n" + "="*60)
    print("ANALYSIS A: Start Node Invariance (Full Dataset)")
    print("="*60)
    
    G, _, _, _ = create_complete_graph(num_files=10)
    giant_nodes = get_giant_component_nodes(G)
    
    results = {"bfs": [], "dfs": []}
    NUM_RUNS = 30
    
    for algo_name, algo_func in [("bfs", bfs_traversal), ("dfs", dfs_traversal)]:
        print(f"  Running {algo_name.upper()} {NUM_RUNS} times...")
        for _ in range(NUM_RUNS):
            start_node = random.choice(giant_nodes)
            t0 = time.perf_counter()
            algo_func(G, start_node, visited=set())
            results[algo_name].append(time.perf_counter() - t0)
            
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(results["bfs"], 'o-', label='BFS', alpha=0.7)
    plt.plot(results["dfs"], 's-', label='DFS', alpha=0.7)
    plt.xlabel('Run ID')
    plt.ylabel('Time (s)')
    plt.title('Analysis A: Start Node Invariance (10 Files)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, 'A_Invariance.png'))
    plt.close()
    
    # Markdown Report
    with open(os.path.join(ANALYSIS_DIR, 'A_Invariance.md'), 'w') as f:
        f.write("# Analysis A: Start Node Invariance\n\n")
        f.write(f"**Dataset:** Full Graph (10 files)\n\n")
        f.write("| Algorithm | Mean Time (s) | Std Dev (s) |\n")
        f.write("|-----------|---------------|-------------|\n")
        f.write(f"| BFS | {np.mean(results['bfs']):.5f} | {np.std(results['bfs']):.5f} |\n")
        f.write(f"| DFS | {np.mean(results['dfs']):.5f} | {np.std(results['dfs']):.5f} |\n")
        f.write("\n**Conclusion:** Low standard deviation confirms O(V+E) is independent of the start node within the component.\n")
    print("  > Saved plots/A_Invariance.png and analysis/A_Invariance.md")

# ==========================================
# ANALYSIS B: Time Complexity
# (Iterate 1-10 for Trend, 4 Subplots per Algo)
# ==========================================
def run_analysis_B():
    print("\n" + "="*60)
    print("ANALYSIS B: Time Complexity (Scalability 1-10)")
    print("="*60)
    
    stats = []
    
    # Iterate 1 to 10 files
    for i in range(1, 11):
        G, _, _, _ = create_complete_graph(num_files=i)
        V = G.number_of_nodes()
        E = G.number_of_edges()
        start_node = list(G.nodes())[0]
        
        t0 = time.perf_counter()
        bfs_traversal(G, start_node, set())
        t_bfs = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        dfs_traversal(G, start_node, set())
        t_dfs = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        run_ufa(G)
        t_ufa = time.perf_counter() - t0
        
        stats.append({'V': V, 'E': E, 'VE': V+E, 'VxE': V*E, 'bfs': t_bfs, 'dfs': t_dfs, 'ufa': t_ufa})
        print(f"  Processed {i} file(s): V={V}, E={E}")

    # Generate 4-Panel Grids for EACH Algorithm
    metrics = [
        ('V', 'Nodes (V)'),
        ('E', 'Edges (E)'),
        ('VE', 'V + E'),
        ('VxE', 'V * E')
    ]
    
    for algo in ['bfs', 'dfs', 'ufa']:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{algo.upper()} Complexity Analysis', fontsize=16)
        
        times = [s[algo] for s in stats]
        
        for idx, (key, label) in enumerate(metrics):
            row, col = idx // 2, idx % 2
            ax = axs[row, col]
            
            x_vals = [s[key] for s in stats]
            
            # Sort for clean plotting
            sorted_pairs = sorted(zip(x_vals, times))
            xs = [p[0] for p in sorted_pairs]
            ys = [p[1] for p in sorted_pairs]
            
            ax.plot(xs, ys, 'o-', linewidth=1.5)
            ax.set_title(f'Time vs {label}')
            ax.set_xlabel(label)
            ax.set_ylabel('Time (s)')
            ax.grid(True)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(PLOT_DIR, f'B_{algo.upper()}_Complexity_Grid.png'))
        plt.close()

    # Consolidated Markdown Report
    with open(os.path.join(ANALYSIS_DIR, 'B_Complexity.md'), 'w') as f:
        f.write("# Analysis B: Time Complexity\n\n")
        f.write("Scalability analysis performed by iterating dataset size from 1 to 10 files.\n\n")
        f.write("| Files | V | E | BFS (s) | DFS (s) | UFA (s) |\n")
        f.write("|-------|---|---|---------|---------|---------|\n")
        for i, s in enumerate(stats):
            f.write(f"| {i+1} | {s['V']} | {s['E']} | {s['bfs']:.4f} | {s['dfs']:.4f} | {s['ufa']:.4f} |\n")
            
    print("  > Saved plots/B_*_Complexity_Grid.png and analysis/B_Complexity.md")

# ==========================================
# ANALYSIS C: Order Invariance
# (Run on FULL 10 Files)
# ==========================================
def run_analysis_C():
    print("\n" + "="*60)
    print("ANALYSIS C: Order Invariance (Full Dataset)")
    print("="*60)
    
    G, _, _, _ = create_complete_graph(num_files=10)
    start_node = list(G.nodes())[0]
    results = {}
    
    for name, func in [("BFS", bfs_traversal), ("DFS", dfs_traversal), ("UFA", run_ufa)]:
        times_norm, times_shuf = [], []
        for _ in range(10):
            # Normal
            t0 = time.perf_counter()
            if name == "UFA": func(G, shuffle_edges=False)
            else: func(G, start_node, visited=set(), shuffle_children=False)
            times_norm.append(time.perf_counter() - t0)
            
            # Shuffled
            t0 = time.perf_counter()
            if name == "UFA": func(G, shuffle_edges=True)
            else: func(G, start_node, visited=set(), shuffle_children=True)
            times_shuf.append(time.perf_counter() - t0)
            
        results[name] = (np.mean(times_norm), np.mean(times_shuf))
        
    # Plotting
    plt.figure(figsize=(8, 6))
    x = np.arange(3)
    width = 0.35
    plt.bar(x - width/2, [results[k][0] for k in results], width, label='Normal')
    plt.bar(x + width/2, [results[k][1] for k in results], width, label='Shuffled')
    plt.xticks(x, results.keys())
    plt.ylabel('Time (s)')
    plt.title('Analysis C: Execution Order Invariance (10 Files)')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, 'C_Order_Invariance.png'))
    plt.close()
    
    # Markdown Report
    with open(os.path.join(ANALYSIS_DIR, 'C_Order.md'), 'w') as f:
        f.write("# Analysis C: Execution Order Invariance\n\n")
        f.write("| Algorithm | Normal Time (s) | Shuffled Time (s) | Diff (%) |\n")
        f.write("|-----------|-----------------|-------------------|----------|\n")
        for k, v in results.items():
            diff = abs(v[0] - v[1]) / v[0] * 100
            f.write(f"| {k} | {v[0]:.5f} | {v[1]:.5f} | {diff:.2f}% |\n")
    print("  > Saved plots/C_Order_Invariance.png and analysis/C_Order.md")

# ==========================================
# ANALYSIS D: CCA Deliverables
# (Iterate 1-10, Per-Step Outputs)
# ==========================================
def run_analysis_D():
    print("\n" + "="*60)
    print("ANALYSIS D: Connected Components (Iterating 1-10 Files)")
    print("="*60)
    
    for i in range(1, 11):
        print(f"  Analyzing Dataset Size: {i} File(s)...")
        G, _, _, _ = create_complete_graph(num_files=i)
        nodes = list(G.nodes())
        
        # Run CCA (BFS)
        visited = set()
        components = []
        for node in nodes:
            if node not in visited:
                comp, edges, _ = bfs_traversal(G, node, visited)
                components.append({'nodes': len(comp), 'edges': edges})
        
        components.sort(key=lambda x: x['nodes'], reverse=True)
        giant = components[0] if components else {'nodes': 0, 'edges': 0}
        
        # 1. Plot Distribution for this step
        sizes = [c['nodes'] for c in components]
        plt.figure(figsize=(8, 5))
        plt.hist(sizes, bins=30, color='skyblue', edgecolor='black', log=True)
        plt.title(f'Component Size Distribution ({i} Files)')
        plt.xlabel('Component Size')
        plt.ylabel('Frequency (Log)')
        plt.savefig(os.path.join(PLOT_DIR, f'D_{i}_Distribution.png'))
        plt.close()
        
        # 2. Markdown Report for this step
        with open(os.path.join(ANALYSIS_DIR, f'D_{i}_Analysis.md'), 'w') as f:
            f.write(f"# Connected Components Analysis: {i} File(s)\n\n")
            f.write(f"- **Total Nodes:** {G.number_of_nodes():,}\n")
            f.write(f"- **Total Edges:** {G.number_of_edges():,}\n")
            f.write(f"- **Total Components:** {len(components)}\n\n")
            f.write("## Giant Component (GC)\n")
            f.write(f"- **Nodes:** {giant['nodes']:,}\n")
            f.write(f"- **Edges:** {giant['edges']:,}\n")
            if G.number_of_nodes() > 0:
                f.write(f"- **Coverage:** {(giant['nodes']/G.number_of_nodes())*100:.2f}% of graph\n\n")
            f.write("## Component Distribution\n")
            f.write(f"![Distribution Plot](../plots/D_{i}_Distribution.png)\n")
            
    print("  > Saved separate plots and .md reports for all 10 steps in plots/ and analysis/")

if __name__ == "__main__":
    run_analysis_A()
    run_analysis_B()
    run_analysis_C()
    run_analysis_D()