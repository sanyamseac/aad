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

# ==========================================
# HELPERS
# ==========================================
def get_giant_component_nodes(G):
    """Finds a node within the giant component to ensure valid testing."""
    visited = set()
    best_comp = set()
    # Sample a subset of nodes for speed if graph is huge
    sample_nodes = list(G.nodes())[:50] if len(G.nodes()) > 50 else list(G.nodes())
    
    for node in sample_nodes:
        if node not in visited:
            comp, _, _ = bfs_traversal(G, node, visited)
            if len(comp) > len(best_comp):
                best_comp = comp
    return list(best_comp)

# ==========================================
# ANALYSIS A: Start Node Invariance
# Valid for: BFS, DFS (UFA is global, not rooted)
# ==========================================
def run_analysis_A():
    print("\n" + "="*60)
    print("ANALYSIS A: Start Node Invariance (BFS & DFS)")
    print("="*60)
    print("Proof: Choosing any random node gives same runtime/space.")
    
    G, _, _, _ = create_complete_graph(num_files=5)
    giant_nodes = get_giant_component_nodes(G)
    print(f"Testing on Giant Component ({len(giant_nodes)} nodes)...")
    
    results = {"BFS": [], "DFS": []}
    NUM_RUNS = 30
    
    for algo_name, algo_func in [("BFS", bfs_traversal), ("DFS", dfs_traversal)]:
        print(f"  Running {algo_name} {NUM_RUNS} times...")
        for _ in range(NUM_RUNS):
            start_node = random.choice(giant_nodes)
            tracemalloc.start()
            t0 = time.perf_counter()
            
            algo_func(G, start_node, visited=set())
            
            dt = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            results[algo_name].append({"time": dt, "space": peak})
            
        times = [r["time"] for r in results[algo_name]]
        spaces = [r["space"] for r in results[algo_name]]
        
        print(f"    {algo_name} Time: Avg={np.mean(times):.4f}s, StdDev={np.std(times):.4f}")
        print(f"    {algo_name} Space: Avg={np.mean(spaces)/1024:.2f}KB, StdDev={np.std(spaces)/1024:.2f}KB")

# ==========================================
# ANALYSIS B: Time Complexity vs Graph Size
# Valid for: BFS, DFS, UFA
# ==========================================
def run_analysis_B():
    print("\n" + "="*60)
    print("ANALYSIS B: Time Complexity vs V+E")
    print("="*60)
    print("Proof: Graph of Time vs (V+E) should be a straight line.")
    
    data = {"BFS": [], "DFS": [], "UFA": []}
    graph_sizes = [] # Stores V+E
    
    # Increment graph size from 1 to 10 files
    for i in range(1, 11):
        G, _, _, _ = create_complete_graph(num_files=i)
        V = G.number_of_nodes()
        E = G.number_of_edges()
        ve = V + E
        graph_sizes.append(ve)
        
        print(f"  Files={i}, V+E={ve}...")
        start_node = list(G.nodes())[0]
        
        # BFS
        t0 = time.perf_counter()
        bfs_traversal(G, start_node, set())
        data["BFS"].append(time.perf_counter() - t0)
        
        # DFS
        t0 = time.perf_counter()
        dfs_traversal(G, start_node, set())
        data["DFS"].append(time.perf_counter() - t0)
        
        # UFA
        t0 = time.perf_counter()
        run_ufa(G)
        data["UFA"].append(time.perf_counter() - t0)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(graph_sizes, data["BFS"], 'o-', label='BFS')
    plt.plot(graph_sizes, data["DFS"], 's-', label='DFS')
    plt.plot(graph_sizes, data["UFA"], '^-', label='UFA')
    
    plt.xlabel('Graph Size (Nodes + Edges)')
    plt.ylabel('Time (seconds)')
    plt.title('Time Complexity Analysis (V+E)')
    plt.legend()
    plt.grid(True)
    plt.savefig('analysis_B_complexity.png')
    print("  Plot saved to analysis_B_complexity.png")

# ==========================================
# ANALYSIS C: Order Invariance
# Valid for: BFS, DFS, UFA
# ==========================================
def run_analysis_C():
    print("\n" + "="*60)
    print("ANALYSIS C: Visit/Edge Order Invariance")
    print("="*60)
    print("Proof: Shuffling children/edges does not affect complexity.")
    
    G, _, _, _ = create_complete_graph(num_files=5)
    start_node = list(G.nodes())[0]
    
    def test_algo(name, func, *args):
        times_norm = []
        times_shuf = []
        for _ in range(10):
            t0 = time.perf_counter()
            func(*args, shuffle_children=False) if "UFA" not in name else func(*args, shuffle_edges=False)
            times_norm.append(time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            func(*args, shuffle_children=True) if "UFA" not in name else func(*args, shuffle_edges=True)
            times_shuf.append(time.perf_counter() - t0)
            
        print(f"  {name}: Normal={np.mean(times_norm):.4f}s, Shuffled={np.mean(times_shuf):.4f}s")

    test_algo("BFS", bfs_traversal, G, start_node, set())
    test_algo("DFS", dfs_traversal, G, start_node, set())
    test_algo("UFA", run_ufa, G)

# ==========================================
# ANALYSIS D: CCA Deliverables
# Valid for: BFS, DFS, UFA (Comparison)
# ==========================================
def run_analysis_D():
    print("\n" + "="*60)
    print("ANALYSIS D: CCA Deliverables (Counts, Giant Comp, Distribution)")
    print("="*60)
    
    print("Loading FULL 10-file graph...")
    G, _, _, _ = create_complete_graph(num_files=10)
    nodes = list(G.nodes())
    
    # Run BFS for CCA
    print("Running BFS CCA...")
    visited = set()
    bfs_comps = []
    for node in nodes:
        if node not in visited:
            comp, edges, _ = bfs_traversal(G, node, visited)
            bfs_comps.append({"nodes": len(comp), "edges": edges})
            
    # Run UFA for CCA (Validation)
    print("Running UFA CCA (Validation)...")
    ufa_comps_raw, _ = run_ufa(G)
    
    print(f"  BFS found {len(bfs_comps)} components.")
    print(f"  UFA found {len(ufa_comps_raw)} components.")
    
    # Deliverables
    bfs_comps.sort(key=lambda x: x['nodes'], reverse=True)
    giant = bfs_comps[0]
    
    print("\n[DELIVERABLE] Giant Component Analysis:")
    print(f"  Nodes: {giant['nodes']:,}")
    print(f"  Edges: {giant['edges']:,}")
    print(f"  Percentage: {(giant['nodes']/G.number_of_nodes())*100:.2f}%")
    
    print("\n[DELIVERABLE] Component Size Distribution:")
    sizes = [c['nodes'] for c in bfs_comps]
    
    plt.figure(figsize=(10, 6))
    plt.hist(sizes, bins=np.logspace(np.log10(1), np.log10(max(sizes)), 50), color='teal', edgecolor='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Component Size')
    plt.ylabel('Frequency')
    plt.title('Connected Component Size Distribution (Log-Log)')
    plt.savefig('analysis_D_distribution.png')
    print("  Plot saved to analysis_D_distribution.png")

if __name__ == "__main__":
    # Run all analyses
    run_analysis_A()
    run_analysis_B()
    run_analysis_C()
    run_analysis_D()