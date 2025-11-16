#!/usr/bin/env python3
import os
import sys
from collections import deque

# ensure parent package 'aad' is importable (same trick you used)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

# optional: networkx-based approach (recommended if networkx is available)
try:
    import networkx as nx
    from networkx.algorithms.community import girvan_newman as nx_girvan_newman
except Exception:
    nx = None
    nx_girvan_newman = None

def load_graph_from_dataset():
    """Load the graph using create_complete_graph from your graph module."""
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(dataset_path=dataset_path)
    return G, all_ego_nodes, all_circles, all_features

def girvan_newman_networkx(G, num_communities=2):
    """
    Use networkx's girvan_newman generator and return the partition
    with `num_communities` communities (as a list of sets).
    """
    if nx_girvan_newman is None:
        raise RuntimeError("networkx.girvan_newman not available. Install networkx or use the custom implementation.")
    comp_generator = nx_girvan_newman(G)
    for communities in comp_generator:
        if len(communities) >= num_communities:
            return [set(c) for c in communities]
    # if we exit loop, return the last
    return [set(c) for c in communities]

def girvan_newman_custom(G, num_communities=2, edge_betweenness_func=None):
    """
    A simple custom implementation:
    - Repeatedly computes edge betweenness (using networkx if available)
    - Removes the edge(s) with highest betweenness
    - Stops when the graph splits into >= num_communities components
    Returns a list of sets (the communities).
    NOTE: This modifies a copy of G, not the original graph.
    """
    if num_communities <= 1:
        return [set(G.nodes())]

    # work on a copy so we don't modify original
    H = G.copy()

    # use networkx's edge betweenness if available else fallback
    use_nx = (nx is not None)
    while True:
        # number of connected components
        components = list(nx.connected_components(H)) if use_nx else list(_connected_components_simple(H))
        if len(components) >= num_communities:
            return [set(c) for c in components]

        # compute edge betweenness
        if use_nx:
            eb = nx.edge_betweenness_centrality(H)
        else:
            # fallback: very slow naive betweenness using BFS Brandes-like
            eb = _edge_betweenness_naive(H)

        # find maximum betweenness value
        if not eb:
            return [set(c) for c in nx.connected_components(H)] if use_nx else [set(c) for c in _connected_components_simple(H)]
        max_val = max(eb.values())

        # remove all edges that share the maximum value (this matches typical GN implementation)
        edges_to_remove = [e for e, val in eb.items() if abs(val - max_val) < 1e-12]

        H.remove_edges_from(edges_to_remove)

        # continue loop until desired number of components reached

# --- tiny fallback helpers (only used if networkx not installed) ---
def _connected_components_simple(G):
    """Return connected components for a dict-based adjacency graph.
       This assumes G has .nodes() and .neighbors(node) methods (like yours)."""
    visited = set()
    comps = []
    for n in G.nodes():
        if n in visited:
            continue
        comp = set()
        stack = [n]
        visited.add(n)
        while stack:
            v = stack.pop()
            comp.add(v)
            for w in G.neighbors(v):
                if w not in visited:
                    visited.add(w)
                    stack.append(w)
        comps.append(comp)
    return comps

def _edge_betweenness_naive(G):
    """
    Naive edge-betweenness approximation using Brandes for all pairs shortest paths.
    This is intentionally simple and will be slow for large graphs.
    Returns dict mapping (u,v) tuples (with u<v ordering) -> betweenness value.
    """
    # We'll reuse the Brandes algorithm but accumulate for edges
    # Ensure canonical edge key (min,max)
    edge_bet = {}
    for u in G.nodes():
        # initialize
        stack = []
        predecessors = {w: [] for w in G.nodes()}
        sigma = dict.fromkeys(G.nodes(), 0.0)
        sigma[u] = 1.0
        dist = dict.fromkeys(G.nodes(), -1)
        dist[u] = 0
        queue = deque([u])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in G.neighbors(v):
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)
        delta = dict.fromkeys(G.nodes(), 0.0)
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                # canonical edge key
                e = (v, w) if (v, w) in edge_bet or (v <= w) else (w, v)
                edge_key = tuple(sorted((v, w)))
                edge_bet[edge_key] = edge_bet.get(edge_key, 0.0) + c
                delta[v] += c
    return edge_bet

# ------------------- Example usage -------------------
if __name__ == "__main__":
    print("Loading graph from dataset...")
    G, all_ego_nodes, all_circles, all_features = load_graph_from_dataset()
    print("Graph loaded. Nodes:", G.number_of_nodes() if hasattr(G, "number_of_nodes") else len(list(G.nodes())))
    print("Edges:", G.number_of_edges() if hasattr(G, "number_of_edges") else "unknown")

    # --- Option A: networkx built-in girvan_newman (recommended if available) ---
    if nx is not None and nx_girvan_newman is not None:
        print("\nRunning networkx.girvan_newman and taking partition with 3 communities (example)...")
        communities = girvan_newman_networkx(G, num_communities=3)
        for i, c in enumerate(communities, 1):
            print(f"Community {i} (size {len(c)}): {sorted(list(c))[:10]}{'...' if len(c)>10 else ''}")
    else:
        print("\nnetworkx not available: skipping nx.girvan_newman demo. You can `pip install networkx` to use it.")

    # --- Option B: custom implementation (removes max-betweenness edges until k components) ---
    print("\nRunning custom Girvan-Newman to get 3 communities (this may be slow for large graphs)...")
    communities_custom = girvan_newman_custom(G, num_communities=3)
    for i, c in enumerate(communities_custom, 1):
        print(f"Custom Community {i} (size {len(c)}): {sorted(list(c))[:10]}{'...' if len(c)>10 else ''}")

    print("\nDone.")
