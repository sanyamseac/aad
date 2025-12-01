import os
import sys
from collections import deque
import random
import time

random.seed(67)

# Include imports for dataset access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

def bfs_traversal(G, start_node, visited, shuffle_children=False, custom_adj=None):
    """
    Performs Breadth-First Search to find a connected component.
    
    Args:
        G: NetworkX graph object (used for adj if custom_adj is None).
        start_node: Node to start traversal from.
        visited: Set to track visited nodes (modified in-place).
        shuffle_children: Whether to shuffle neighbors (for Order Invariance Analysis).
        custom_adj: Optional pre-shuffled adjacency list (for Order Invariance Analysis).
    
    Returns:
        (set, int, int): (Nodes in component, Edges in component, Max Queue Size)
    """
    q = deque([start_node])
    visited.add(start_node)
    component_nodes = {start_node}
    max_queue_size = 1

    while q:
        curr = q.popleft()
        
        # Use custom_adj if provided (for Analysis C), else G.adj
        if custom_adj:
            neighbors = custom_adj.get(curr, [])
        else:
            neighbors = list(G.adj.get(curr, []))
            
        if shuffle_children:
            random.shuffle(neighbors)
            
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                component_nodes.add(neighbor)
                q.append(neighbor)
        
        max_queue_size = max(max_queue_size, len(q))

    # Count internal edges strictly within the component
    # We count edge (u, v) only if u < v to avoid double counting undirected edges
    edges_in_component = 0
    for node in component_nodes:
        for neighbor in G.adj.get(node, []):
            if neighbor in component_nodes and node < neighbor:
                edges_in_component += 1
                
    return component_nodes, edges_in_component, max_queue_size

if __name__ == "__main__":
    print("\n--- BFS Standalone Traversal ---")
    try:
        val = input("Enter number of files to use (1-10): ").strip()
        num = int(val)
        if not (1 <= num <= 10): raise ValueError
    except ValueError:
        print("Invalid input. Defaulting to 1 file.")
        num = 1
        
    print(f"Loading {num} file(s)...")
    G, _, _, _ = create_complete_graph(num_files=num)
    
    if len(G.nodes()) > 0:
        start = list(G.nodes())[0]
        print(f"Traversing from node {start}...")
        t0 = time.perf_counter()
        comp, edges, _ = bfs_traversal(G, start, set())
        dt = time.perf_counter() - t0
        print(f"Done in {dt:.4f}s.")
        print(f"Component: {len(comp)} nodes, {edges} edges.")
    else:
        print("Graph is empty.")