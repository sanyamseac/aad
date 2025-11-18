import os
import sys
import random

# Include requested imports for dataset access
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

def dfs_traversal(G, start_node, visited, shuffle_children=False):
    """
    Performs Depth-First Search to find a connected component.
    Uses an iterative stack to prevent recursion depth errors on large graphs.
    
    Args:
        G: NetworkX graph object.
        start_node: Node to start from.
        visited (set): Global visited set (modified in-place).
        shuffle_children (bool): Whether to randomize neighbor order.
        
    Returns:
        (set, int, int): (component_nodes, component_edges_count, max_stack_size)
    """
    stack = [start_node]
    visited.add(start_node)
    component_nodes = {start_node}
    
    max_stack_size = 1

    while stack:
        curr = stack.pop()
        
        neighbors = list(G.adj.get(curr, []))
        if shuffle_children:
            random.shuffle(neighbors)
            
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                component_nodes.add(neighbor)
                stack.append(neighbor)
        
        max_stack_size = max(max_stack_size, len(stack))

    edges_in_component = 0
    for node in component_nodes:
        for neighbor in G.adj.get(node, []):
            if neighbor in component_nodes and node < neighbor:
                edges_in_component += 1
                
    return component_nodes, edges_in_component, max_stack_size

if __name__ == "__main__":
    print("Running DFS standalone test...")
    G, _, _, _ = create_complete_graph(num_files=1)
    print(f"Graph loaded: {G}")
    
    start_node = list(G.nodes())[0]
    visited = set()
    comp, edges, _ = dfs_traversal(G, start_node, visited)
    print(f"DFS from node {start_node}: Found component with {len(comp)} nodes and {edges} edges.")