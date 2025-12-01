"""Girvan-Newman Community Detection Algorithm.

This module implements the Girvan-Newman algorithm for detecting communities in networks.
The algorithm iteratively removes edges with the highest betweenness centrality,
gradually separating the network into communities.

Time Complexity: O(m²n) where m=edges, n=nodes
Space Complexity: O(n + m)
"""

import os
import sys
from collections import deque
import copy

# Import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph


def calculate_edge_betweenness(G):
    """Calculate edge betweenness centrality for all edges in the graph.
    
    Uses BFS from each node to compute shortest paths, then accumulates
    betweenness scores for each edge based on the number of shortest paths
    passing through it.
    
    Args:
        G (networkx.Graph): The graph structure.
        
    Returns:
        dict: Dictionary mapping edges (u, v) to their betweenness scores.
        
    Time Complexity: O(mn) where m=edges, n=nodes
    """
    edge_betweenness = {edge: 0.0 for edge in G.edges()}
    
    # Run BFS from each node as source
    for source in G.nodes():
        # Initialize data structures
        stack = []
        predecessors = {node: [] for node in G.nodes()}
        sigma = {node: 0.0 for node in G.nodes()}
        sigma[source] = 1.0
        distance = {node: -1 for node in G.nodes()}
        distance[source] = 0
        
        queue = deque([source])
        
        # Forward pass: BFS to find shortest paths
        while queue:
            v = queue.popleft()
            stack.append(v)
            
            for w in G.neighbors(v):
                # First time we see this node
                if distance[w] < 0:
                    distance[w] = distance[v] + 1
                    queue.append(w)
                
                # This is a shortest path to w
                if distance[w] == distance[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)
        
        # Backward pass: accumulate edge betweenness
        dependency = {node: 0.0 for node in G.nodes()}
        
        while stack:
            w = stack.pop()
            
            for v in predecessors[w]:
                # Calculate the contribution of this edge
                credit = (sigma[v] / sigma[w]) * (1.0 + dependency[w])
                
                # Add to both possible edge representations
                edge1 = (v, w)
                edge2 = (w, v)
                
                if edge1 in edge_betweenness:
                    edge_betweenness[edge1] += credit
                elif edge2 in edge_betweenness:
                    edge_betweenness[edge2] += credit
                
                dependency[v] += credit
    
    return edge_betweenness


def get_connected_components(G):
    """Find all connected components in the graph.
    
    Uses BFS to identify separate connected components in the graph.
    
    Args:
        G (networkx.Graph): The graph structure.
        
    Returns:
        list: List of sets, where each set contains nodes in one component.
        
    Time Complexity: O(n + m)
    """
    """
    Find all connected components in the graph.
    Returns a list of sets, where each set contains nodes in one component.
    """
    visited = set()
    components = []
    
    for node in G.nodes():
        if node not in visited:
            # BFS to find all nodes in this component
            component = set()
            queue = deque([node])
            visited.add(node)
            
            while queue:
                current = queue.popleft()
                component.add(current)
                
                for neighbor in G.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            components.append(component)
    
    return components


def girvan_newman(G, num_communities=None, max_iterations=None):
    """Apply the Girvan-Newman algorithm for community detection.
    
    Iteratively removes edges with highest betweenness centrality until
    the desired number of communities is reached or max iterations exceeded.
    
    Args:
        G (networkx.Graph): The input graph.
        num_communities (int, optional): Stop when this many communities are found.
        max_iterations (int, optional): Maximum number of edge removals.
        
    Returns:
        list: List of tuples (iteration, communities, removed_edge) tracking
              the algorithm's progress at each step.
              
    Time Complexity: O(m²n) for the full algorithm
    """
    # Create a copy of the graph to avoid modifying the original
    G_copy = copy.deepcopy(G)
    
    history = []
    iteration = 0
    
    print(f"Starting Girvan-Newman algorithm on graph with {G_copy.number_of_nodes()} nodes and {G_copy.number_of_edges()} edges\n")
    
    while G_copy.number_of_edges() > 0:
        # Check stopping conditions
        components = get_connected_components(G_copy)
        
        if num_communities and len(components) >= num_communities:
            print(f"Reached {num_communities} communities. Stopping.")
            break
        
        if max_iterations and iteration >= max_iterations:
            print(f"Reached maximum iterations ({max_iterations}). Stopping.")
            break
        
        # Calculate edge betweenness for all edges
        edge_betweenness = calculate_edge_betweenness(G_copy)
        
        if not edge_betweenness:
            break
        
        # Find the edge with highest betweenness
        max_edge = max(edge_betweenness.items(), key=lambda x: x[1])
        edge_to_remove = max_edge[0]
        betweenness_score = max_edge[1]
        
        # Remove the edge
        G_copy.remove_edge(*edge_to_remove)
        
        # Get current communities
        components = get_connected_components(G_copy)
        
        iteration += 1
        history.append((iteration, components, edge_to_remove, betweenness_score))
        
        print(f"Iteration {iteration}: Removed edge {edge_to_remove} (betweenness: {betweenness_score:.4f})")
        print(f"  Current number of communities: {len(components)}")
        print(f"  Remaining edges: {G_copy.number_of_edges()}\n")
    
    return history


if __name__ == "__main__":
    # Load the complete graph using the function from 'graph.py'
    print("Loading graph from dataset...")
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(2)
    print(f"Graph loaded: {G}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}\n")

    # Run Girvan-Newman algorithm
    # You can specify num_communities or max_iterations as stopping criteria
    print("=" * 60)
    print("Running Girvan-Newman Algorithm")
    print("=" * 60 + "\n")
    
    # Example: Stop after finding 5 communities or 20 iterations
    history = girvan_newman(G, num_communities=5, max_iterations=20)
    
    # Display final communities
    print("\n" + "=" * 60)
    print("FINAL COMMUNITIES")
    print("=" * 60)
    
    if history:
        final_iteration, final_communities, _, _ = history[-1]
        for i, community in enumerate(final_communities, 1):
            print(f"\nCommunity {i} (size: {len(community)}):")
            print(f"  Nodes: {sorted(list(community))[:10]}{'...' if len(community) > 10 else ''}")
    else:
        print("No communities detected.")