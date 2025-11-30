import os
import sys
import random
from collections import defaultdict, Counter

# Import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph


def modularity(G, communities):
    """Calculate modularity of a partition."""
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    
    Q = 0.0
    for community in communities.values():
        nodes = list(community)
        for i in nodes:
            for j in nodes:
                if G.has_edge(i, j):
                    Q += 1
                # Subtract expected edges
                Q -= (G.degree(i) * G.degree(j)) / (2 * m)
    
    return Q / (2 * m)


def move_nodes_fast(G, partition, randomize=True):
    """Fast local moving of nodes (Phase 1 of Leiden)."""
    nodes = list(G.nodes())
    if randomize:
        random.shuffle(nodes)
    
    improved = True
    while improved:
        improved = False
        
        for node in nodes:
            current_comm = partition[node]
            
            # Get neighboring communities
            neighbor_comms = defaultdict(int)
            for neighbor in G.neighbors(node):
                neighbor_comms[partition[neighbor]] += 1
            
            # Calculate current modularity contribution
            best_comm = current_comm
            best_delta = 0.0
            
            # Try moving to each neighboring community
            for comm, weight in neighbor_comms.items():
                if comm == current_comm:
                    continue
                
                # Calculate modularity gain
                k_i = G.degree(node)
                k_i_in_new = weight
                k_i_in_old = sum(1 for n in G.neighbors(node) if partition[n] == current_comm)
                
                m = G.number_of_edges()
                sum_in_new = sum(G.degree(n) for n in G.nodes() if partition[n] == comm)
                sum_in_old = sum(G.degree(n) for n in G.nodes() if partition[n] == current_comm)
                
                delta = (k_i_in_new - k_i_in_old) / m - k_i * (sum_in_new - sum_in_old + k_i) / (2 * m * m)
                
                if delta > best_delta:
                    best_delta = delta
                    best_comm = comm
            
            # Move node if improvement found
            if best_comm != current_comm:
                partition[node] = best_comm
                improved = True
    
    return partition


def refine_partition(G, partition):
    """Refine partition by splitting communities (Phase 2 of Leiden)."""
    # Create subgraph for each community
    communities = defaultdict(set)
    for node, comm in partition.items():
        communities[comm].add(node)
    
    new_partition = {}
    next_comm_id = max(partition.values()) + 1
    
    for comm_id, nodes in communities.items():
        if len(nodes) <= 1:
            for node in nodes:
                new_partition[node] = comm_id
            continue
        
        # Create subgraph
        subgraph = G.subgraph(nodes)
        
        # Try to split: assign each node to its own community initially
        sub_partition = {node: i for i, node in enumerate(nodes)}
        
        # Local moving within subgraph
        sub_partition = move_nodes_fast(subgraph, sub_partition, randomize=True)
        
        # Map back to global partition
        sub_comms = defaultdict(set)
        for node, sub_comm in sub_partition.items():
            sub_comms[sub_comm].add(node)
        
        # Assign new community IDs
        for sub_comm_nodes in sub_comms.values():
            for node in sub_comm_nodes:
                new_partition[node] = next_comm_id
            next_comm_id += 1
    
    return new_partition


def aggregate_graph(G, partition):
    """Create aggregate graph where each community becomes a super-node."""
    import networkx as nx
    
    # Create mapping of communities
    communities = defaultdict(set)
    for node, comm in partition.items():
        communities[comm].add(node)
    
    # Create new graph
    agg_G = nx.Graph()
    
    # Add nodes (one per community)
    for comm_id in communities.keys():
        agg_G.add_node(comm_id)
    
    # Add edges between communities
    edge_weights = defaultdict(int)
    for u, v in G.edges():
        comm_u = partition[u]
        comm_v = partition[v]
        if comm_u != comm_v:
            edge = tuple(sorted([comm_u, comm_v]))
            edge_weights[edge] += 1
    
    for (u, v), weight in edge_weights.items():
        agg_G.add_edge(u, v, weight=weight)
    
    return agg_G, communities


def leiden_algorithm(G, max_iterations=10, resolution=1.0):
    """
    Leiden algorithm for community detection.
    
    Parameters:
    - G: NetworkX graph (unweighted, undirected)
    - max_iterations: Maximum number of iterations
    - resolution: Resolution parameter (higher = more communities)
    
    Returns:
    - partition: Dictionary mapping nodes to community IDs
    """
    # Initialize: each node in its own community
    partition = {node: i for i, node in enumerate(G.nodes())}
    
    prev_modularity = modularity(G, defaultdict(set, {i: {node} for node, i in partition.items()}))
    
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}...")
        
        # Phase 1: Fast local moving
        partition = move_nodes_fast(G, partition, randomize=True)
        
        # Phase 2: Refinement
        partition = refine_partition(G, partition)
        
        # Calculate modularity
        communities = defaultdict(set)
        for node, comm in partition.items():
            communities[comm].add(node)
        
        current_modularity = modularity(G, communities)
        print(f"  Modularity: {current_modularity:.6f}")
        
        # Check convergence
        if abs(current_modularity - prev_modularity) < 1e-6:
            print("Converged!")
            break
        
        prev_modularity = current_modularity
        
        # Phase 3: Aggregate network
        if iteration < max_iterations - 1:
            agg_G, comm_mapping = aggregate_graph(G, partition)
            if agg_G.number_of_nodes() == G.number_of_nodes():
                print("No further aggregation possible.")
                break
    
    # Renumber communities to be consecutive
    unique_comms = sorted(set(partition.values()))
    comm_map = {old: new for new, old in enumerate(unique_comms)}
    partition = {node: comm_map[comm] for node, comm in partition.items()}
    
    return partition


if __name__ == "__main__":
    # Load the complete graph using the function from 'graph.py'
    print("Loading graph from dataset...")
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(1)
    print("Graph loaded:", G)

    # Run Leiden algorithm
    print("\nRunning Leiden Algorithm...")
    partition = leiden_algorithm(G, max_iterations=10)
    
    # Analyze results
    communities = defaultdict(set)
    for node, comm in partition.items():
        communities[comm].add(node)
    
    print(f"\n--- Results ---")
    print(f"Number of communities detected: {len(communities)}")
    print(f"Modularity: {modularity(G, communities):.6f}")
    
    print("\n--- Community Sizes ---")
    sorted_comms = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (comm_id, nodes) in enumerate(sorted_comms[:10]):
        print(f"Community {comm_id}: {len(nodes)} nodes")
    
    print("\n--- Sample Nodes from Top 3 Communities ---")
    for i, (comm_id, nodes) in enumerate(sorted_comms[:3]):
        sample_nodes = list(nodes)[:5]
        print(f"Community {comm_id}: {sample_nodes}")