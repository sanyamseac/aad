import os
import sys
from collections import deque
import random
import time

# Include requested imports for dataset access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

def bfs_traversal(G, start_node, visited, shuffle_children=False):
    """
    Performs Breadth-First Search to find a connected component.
    """
    q = deque([start_node])
    visited.add(start_node)
    component_nodes = {start_node}
    
    max_queue_size = 1

    while q:
        curr = q.popleft()
        
        neighbors = list(G.adj.get(curr, []))
        if shuffle_children:
            random.shuffle(neighbors)
            
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                component_nodes.add(neighbor)
                q.append(neighbor)
        
        max_queue_size = max(max_queue_size, len(q))

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
        print("Invalid input. Using 1 file.")
        num = 1
        
    print(f"Loading {num} file(s)...")
    G, _, _, _ = create_complete_graph(num_files=num)
    start = list(G.nodes())[0]
    
    print(f"Traversing from node {start}...")
    t0 = time.perf_counter()
    comp, edges, _ = bfs_traversal(G, start, set())
    print(f"Done in {time.perf_counter()-t0:.4f}s.")
    print(f"Component: {len(comp)} nodes, {edges} edges.")