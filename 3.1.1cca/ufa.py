import os
import sys
import random

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
    
    Args:
        G: NetworkX graph object.
        nodes: Optional list of nodes (if None, taken from G).
        shuffle_edges (bool): Whether to process edges in random order.
    
    Returns:
        (list, int): (List of component dicts, number of edges processed)
    """
    if nodes is None:
        nodes = list(G.nodes())
        
    uf = UnionFind(nodes)
    
    # Extract unique edges (u < v)
    edges = []
    for u in G.adj:
        for v in G.adj[u]:
            if u < v:
                edges.append((u, v))
                
    if shuffle_edges:
        random.shuffle(edges)
        
    for u, v in edges:
        uf.union(u, v)
        
    # Group by root
    components_map = {}
    for node in nodes:
        root = uf.find(node)
        if root not in components_map:
            components_map[root] = []
        components_map[root].append(node)
        
    # Format results
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
            "edges": edges_count,
            "node_set": comp_node_set 
        })
        
    return component_stats, len(edges)

if __name__ == "__main__":
    print("Running UFA standalone test...")
    G, _, _, _ = create_complete_graph(num_files=1)
    print(f"Graph loaded: {G}")
    
    comps, _ = run_ufa(G)
    print(f"UFA found {len(comps)} components.")