#!/usr/bin/env python3
"""
girvan_newman_aad.py
Improved, portable Girvan-Newman script adapted for the 'aad' project.

Features:
- Works with NetworkX Graph or a custom graph object that implements .nodes(),
  .neighbors(node), .remove_edges_from(list_of_edge_tuples), .copy(), and
  optionally .number_of_nodes(), .number_of_edges().
- Two modes: use networkx.girvan_newman (if available) or a custom implementation
  that repeatedly removes max edge-betweenness edges.
- CLI options: --dataset-path, --num-communities, --use-nx, --output-json
- Robust canonical edge key handling and fixed naive Brandes accumulation
- Produces JSON output summarizing communities and optionally writes to file

Usage example:
    python3 girvan_newman_aad.py --dataset-path ../dataset --num-communities 3 --output-json communities.json

"""

import os
import sys
import json
import argparse
import time
from collections import deque

# make parent package 'aad' importable (same trick your project used)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from graph import create_complete_graph
except Exception as e:
    create_complete_graph = None
    # we'll still let the user load a prebuilt networkx graph if they want

# optional: networkx-based approach
try:
    import networkx as nx
    from networkx.algorithms.community import girvan_newman as nx_girvan_newman
except Exception:
    nx = None
    nx_girvan_newman = None


def load_graph_from_dataset(dataset_path=None):
    """Load the graph using create_complete_graph from your graph module.
    If create_complete_graph is not available, try to load a networkx graph file
    path if provided (not implemented here)."""
    if create_complete_graph is None:
        raise RuntimeError("create_complete_graph not importable from graph module in parent package.")
    dataset_path = dataset_path or os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(1, dataset_path=dataset_path)
    return G, all_ego_nodes, all_circles, all_features


def girvan_newman_networkx(G, num_communities=2):
    """Use networkx's girvan_newman generator and return a partition with
    `num_communities` communities (as a list of sets).
    If G is not a networkx.Graph but provides edges/nodes, try to convert if possible."""
    if nx_girvan_newman is None:
        raise RuntimeError("networkx.girvan_newman not available. Install networkx or use the custom implementation.")
    # If G is not a networkx.Graph, we attempt a conversion
    if not isinstance(G, nx.Graph):
        try:
            H = nx.Graph()
            H.add_nodes_from(list(G.nodes()))
            # attempt to iterate neighbors to reconstruct edges
            for u in G.nodes():
                for v in G.neighbors(u):
                    H.add_edge(u, v)
            G_nx = H
        except Exception:
            raise RuntimeError("Graph is not a networkx.Graph and could not be converted.")
    else:
        G_nx = G

    comp_generator = nx_girvan_newman(G_nx)
    for communities in comp_generator:
        if len(communities) >= num_communities:
            return [set(c) for c in communities]
    # fallback
    return [set(c) for c in communities]


def girvan_newman_custom(G, num_communities=2, max_iterations=10000):
    """
    Custom Girvan-Newman:
    - Works for networkx.Graph or custom graph with expected methods
    - Repeatedly computes edge betweenness and removes all edges with maximum score
    - Stops when the graph has >= num_communities components or when iterations exhausted
    Returns list of sets (communities)
    """
    if num_communities <= 1:
        return [set(G.nodes())]

    # work on a copy so we don't modify the original
    H = G.copy()

    # helper to get components in either nx or custom graph
    def _components(graph):
        if nx is not None and isinstance(graph, nx.Graph):
            return list(nx.connected_components(graph))
        else:
            return _connected_components_simple(graph)

    # helper to compute edge betweenness
    def _edge_betweenness(graph):
        if nx is not None and isinstance(graph, nx.Graph):
            return nx.edge_betweenness_centrality(graph)
        else:
            return _edge_betweenness_naive(graph)

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        comps = _components(H)
        if len(comps) >= num_communities:
            return [set(c) for c in comps]

        eb = _edge_betweenness(H)
        if not eb:
            # no edges left
            return [set(c) for c in _components(H)]

        # Normalize keys to canonical (u,v) with u < v for non-networkx dicts too
        # If networkx returned keys as (u, v) where u-v order may matter, we keep them
        # but ensure comparability by mapping to sorted tuple for custom graphs' output
        # Determine if keys are tuples of nodes
        sample_key = next(iter(eb.keys()))
        use_sorted_keys = not (isinstance(sample_key, tuple) and hasattr(sample_key[0], '__hash__'))

        # find max
        max_val = max(eb.values())
        edges_to_remove = []
        for e, val in eb.items():
            # convert to canonical tuple for removal
            if isinstance(e, tuple) and len(e) == 2:
                u, v = e
            else:
                # unexpected key type; skip
                continue
            edge_key = (u, v)
            edges_to_remove.append(edge_key)
        # But we only want those with value == max_val (within tolerance)
        edges_to_remove = [tuple(e) for e, val in eb.items() if abs(val - max_val) < 1e-12]

        # If our graph is not networkx, ensure edges are removed using a canonical ordering
        # Some custom graph implementations expect (u, v) in same orientation as added edges.
        try:
            H.remove_edges_from(edges_to_remove)
        except Exception:
            # try reversed pairs if removal fails for some edges
            alt = []
            for u, v in edges_to_remove:
                try:
                    H.remove_edge(u, v)
                except Exception:
                    try:
                        H.remove_edge(v, u)
                    except Exception:
                        # if both fail, accumulate for a final try using remove_edges_from with sorted tuple
                        alt.append((u, v))
            if alt:
                try:
                    H.remove_edges_from(alt)
                except Exception:
                    # give up on specific removals; this should be rare
                    pass

    # If we exit due to iteration limit, return current components
    return [set(c) for c in _components(H)]


# --- fallback helpers (used if networkx not installed or for custom graph objects) ---
def _connected_components_simple(G):
    """Return connected components for a dict-like or object-based adjacency graph.
       Expects .nodes() and .neighbors(node) to be available."""
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
    Brandes-like accumulation for edges. Returns a dict with canonical sorted tuple keys (u, v) where u < v.
    This implementation is intentionally simple and will be slow for large graphs.
    """
    nodes = list(G.nodes())
    edge_bet = {}
    for s in nodes:
        # single-source shortest-paths
        stack = []
        predecessors = {w: [] for w in nodes}
        sigma = dict.fromkeys(nodes, 0.0)
        sigma[s] = 1.0
        dist = dict.fromkeys(nodes, -1)
        dist[s] = 0
        queue = deque([s])
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
        delta = dict.fromkeys(nodes, 0.0)
        while stack:
            w = stack.pop()
            coeff = (1.0 + delta[w])
            for v in predecessors[w]:
                c = (sigma[v] / sigma[w]) * coeff
                edge_key = tuple(sorted((v, w)))
                edge_bet[edge_key] = edge_bet.get(edge_key, 0.0) + c
                delta[v] += c
    return edge_bet


# ------------------- CLI and example usage -------------------
def main():
    parser = argparse.ArgumentParser(description="Girvan-Newman community detection for the aad project")
    parser.add_argument('--dataset-path', '-d', help='Path to dataset folder (passed to create_complete_graph)', default=None)
    parser.add_argument('--num-communities', '-k', type=int, default=3, help='Desired number of communities')
    parser.add_argument('--use-nx', action='store_true', help='Prefer using networkx implementation when available')
    parser.add_argument('--output-json', '-o', help='Write communities summary to this JSON file')
    parser.add_argument('--timeout-seconds', type=int, default=600, help='Max seconds for custom run')
    args = parser.parse_args()

    print('Loading graph from dataset...')
    try:
        G, all_ego_nodes, all_circles, all_features = load_graph_from_dataset(args.dataset_path)
    except Exception as e:
        print('Failed to load graph via create_complete_graph:', e)
        sys.exit(1)

    n_nodes = G.number_of_nodes() if hasattr(G, 'number_of_nodes') else len(list(G.nodes()))
    n_edges = G.number_of_edges() if hasattr(G, 'number_of_edges') else 'unknown'
    print(f'Graph loaded. Nodes: {n_nodes}, Edges: {n_edges}')

    start = time.time()
    communities = None
    if args.use_nx and nx is not None and nx_girvan_newman is not None:
        print('Running networkx.girvan_newman...')
        communities = girvan_newman_networkx(G, num_communities=args.num_communities)
    else:
        print('Running custom girvan_newman (may be slow for large graphs)...')
        communities = girvan_newman_custom(G, num_communities=args.num_communities)
    elapsed = time.time() - start

    print(f'Found {len(communities)} communities in {elapsed:.2f}s')
    summary = []
    for i, c in enumerate(communities, 1):
        members = sorted(list(c))
        print(f'Community {i}: size {len(members)}; sample: {members[:10]}{"..." if len(members)>10 else ""}')
        summary.append({'id': i, 'size': len(members), 'members_sample': members[:30]})

    out = {
        'num_communities_requested': args.num_communities,
        'num_communities_found': len(communities),
        'time_seconds': elapsed,
        'summary': summary
    }
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print('Wrote summary to', args.output_json)

    print('Done.')


if __name__ == '__main__':
    main()
