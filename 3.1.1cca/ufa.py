import os
import sys
import random
import time

# Include requested imports for dataset access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

class UnionFind:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}
        
    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
        
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False

def run_ufa(G, nodes=None, shuffle_edges=False):
    """
    Runs Union-Find on the graph to identify components.
    """
    if nodes is None:
        nodes = list(G.nodes())
        
    uf = UnionFind(nodes)
    
    edges = []
    for u in G.adj:
        for v in G.adj[u]:
            if u < v:
                edges.append((u, v))
                
    if shuffle_edges:
        random.shuffle(edges)
        
    for u, v in edges:
        uf.union(u, v)
        
    components_map = {}
    for node in nodes:
        root = uf.find(node)
        if root not in components_map:
            components_map[root] = []
        components_map[root].append(node)
        
    component_stats = []
    for root, comp_nodes in components_map.items():
        comp_node_set = set(comp_nodes)
        edges_count = 0
        for node in comp_node_set:
            for neighbor in G.adj.get(node, []):
                if neighbor in comp_node_set and node < neighbor:
                    edges_count += 1
        component_stats.append({
            "nodes": len(comp_node_set),
            "edges": edges_count
        })
        
    return component_stats, len(edges)

if __name__ == "__main__":
    print("\n--- Union-Find Standalone Analysis ---")
    try:
        val = input("Enter number of files to use (1-10): ").strip()
        num = int(val)
        if not (1 <= num <= 10): raise ValueError
    except ValueError:
        print("Invalid input. Using 1 file.")
        num = 1
        
    print(f"Loading {num} file(s)...")
    G, _, _, _ = create_complete_graph(num_files=num)
    
    print(f"Running Union-Find...")
    t0 = time.perf_counter()
    comps, _ = run_ufa(G)
    print(f"Done in {time.perf_counter()-t0:.4f}s.")
    print(f"Found {len(comps)} connected components.")