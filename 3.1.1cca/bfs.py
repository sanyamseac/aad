import os
import sys
from collections import deque
import random

# Include requested imports for dataset access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

def bfs_traversal(G, start_node, visited, shuffle_children=False):
    """
    Performs Breadth-First Search to find a connected component.
    
    Args:
        G: NetworkX graph object (uses G.adj for adjacency).
        start_node: Node to start from.
        visited (set): Global visited set (modified in-place).
        shuffle_children (bool): Whether to randomize neighbor order.
        
    Returns:
        (set, int, int): (component_nodes, component_edges_count, max_queue_size)
    """
    q = deque([start_node])
    visited.add(start_node)
    component_nodes = {start_node}
    
    max_queue_size = 1

    while q:
        curr = q.popleft()
        
        # Access neighbors via G.adj for performance
        neighbors = list(G.adj.get(curr, []))
        if shuffle_children:
            random.shuffle(neighbors)
            
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                component_nodes.add(neighbor)
                q.append(neighbor)
        
        max_queue_size = max(max_queue_size, len(q))

    # Count edges strictly within the component (u < v to avoid double counting)
    edges_in_component = 0
    for node in component_nodes:
        for neighbor in G.adj.get(node, []):
            if neighbor in component_nodes and node < neighbor:
                edges_in_component += 1
                
    return component_nodes, edges_in_component, max_queue_size

if __name__ == "__main__":
    # Standalone test as requested
    print("Running BFS standalone test...")
    G, _, _, _ = create_complete_graph(num_files=1) # Load small sample
    print(f"Graph loaded: {G}")
    
    start_node = list(G.nodes())[0]
    visited = set()
    comp, edges, _ = bfs_traversal(G, start_node, visited)
    print(f"BFS from node {start_node}: Found component with {len(comp)} nodes and {edges} edges.")