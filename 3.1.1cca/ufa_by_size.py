import os
import sys
import random
import time

random.seed(67)

# Include imports for dataset access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

class UnionFindBySize:
    """Disjoint Set Union with union by size + path compression."""
    def __init__(self, nodes):
        """Initializes parent and size maps.
        Args:
            nodes: Iterable of node identifiers.
        """
        self.parent = {node: node for node in nodes}
        self.size = {node: 1 for node in nodes}
        
    def find(self, i):
        """Finds representative of set containing i (path compression).
        Args:
            i: Node id.
        Returns:
            Root representative of i.
        """
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
        
    def union(self, i, j):
        """Unites sets containing i and j using size heuristic.
        Args:
            i: First node.
            j: Second node.
        Returns:
            True if a merge occurred, False otherwise.
        """
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i != root_j:
            if self.size[root_i] < self.size[root_j]:
                self.parent[root_i] = root_j
                self.size[root_j] += self.size[root_i]
            else:
                self.parent[root_j] = root_i
                self.size[root_i] += self.size[root_j]
            return True
        return False

def run_ufa_size(G, nodes=None, shuffle_edges=False, custom_edges=None):
    """Runs union-find by size over edges and returns component stats.
    Args:
        G: Graph object with adjacency in G.adj.
        nodes: Optional iterable of nodes (defaults to all nodes).
        shuffle_edges: If True, randomizes edge order.
        custom_edges: Optional precomputed edge list to use.
    Returns:
        (list, int): List of component dicts {'nodes': count, 'edges': internal_edges},
        and total number of processed undirected edges.
    """
    if nodes is None: nodes = list(G.nodes())
    uf = UnionFindBySize(nodes)
    
    if custom_edges is not None:
        edges = custom_edges
    else:
        edges = []
        for u in G.adj:
            for v in G.adj[u]:
                if u < v: edges.append((u, v))
        if shuffle_edges: random.shuffle(edges)
        
    for u, v in edges:
        uf.union(u, v)
        
    components_map = {}
    for node in nodes:
        root = uf.find(node)
        if root not in components_map: components_map[root] = []
        components_map[root].append(node)
        
    component_stats = []
    for root, comp_nodes in components_map.items():
        comp_node_set = set(comp_nodes)
        edges_count = 0
        for node in comp_node_set:
            for neighbor in G.adj.get(node, []):
                if neighbor in comp_node_set and node < neighbor:
                    edges_count += 1
        component_stats.append({"nodes": len(comp_node_set), "edges": edges_count})
        
    return component_stats, len(edges)

if __name__ == "__main__":
    print("\n--- Union-Find (Size) Standalone ---")
    try:
        val = input("Files (1-10): ").strip()
        num = int(val)
        if not (1 <= num <= 10): raise ValueError
    except: num = 1
    G, _, _, _ = create_complete_graph(num_files=num)
    t0 = time.perf_counter()
    comps, _ = run_ufa_size(G)
    print(f"Done in {time.perf_counter()-t0:.4f}s. {len(comps)} components.")