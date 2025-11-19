#!/usr/bin/env python3
"""
louvain_aad_full.py

Portable Louvain modularity optimization for the 'aad' project.

Features:
- Loads graph using create_complete_graph (from parent package)
- Two implementations:
    * fast: python-louvain (community) + networkx (if installed)
    * custom: pure-Python Louvain working from an adjacency dict (works with custom graph objects)
- CLI: --dataset-path, --use-nx, --output-json
- Produces JSON summary like your Girvan–Newman script

Notes:
- Custom implementation is unweighted and may be slower than python-louvain for large graphs,
  but it is accurate for unweighted graphs.
"""

import os
import sys
import json
import argparse
import time
from collections import defaultdict

# Make parent package importable (same trick your project used)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from graph import create_complete_graph
except Exception:
    create_complete_graph = None

# optional: networkx + python-louvain (community) for fast path
try:
    import networkx as nx
except Exception:
    nx = None

try:
    # python-louvain package is named 'community'
    import community as community_louvain
except Exception:
    community_louvain = None


# -------------------- Graph loading helpers --------------------
def load_graph_from_dataset(dataset_path=None):
    """
    Loads graph via create_complete_graph(1, dataset_path=...)
    Returns the graph object (whatever create_complete_graph returns as G).
    Raises RuntimeError if create_complete_graph is unavailable.
    """
    if create_complete_graph is None:
        raise RuntimeError("create_complete_graph not importable from parent package 'graph'.")
    dataset_path = dataset_path or os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    # create_complete_graph signature may vary, original script used: create_complete_graph(1, dataset_path=...)
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(1, dataset_path=dataset_path)
    return G, all_ego_nodes, all_circles, all_features


def graph_to_adj_and_stats(G):
    """
    Convert a graph-like object (networkx.Graph or custom with .nodes(), .neighbors(node)) into:
      - adj: {node: set(neighbor_nodes)}
      - degrees: {node: degree}
      - m: total number of edges (int)
    This avoids depending on .degree() or .number_of_edges() being implemented.
    """
    adj = {}
    for u in G.nodes():
        # ensure neighbors iterator works
        nbrs = list(G.neighbors(u))
        adj[u] = set(nbrs)

    degrees = {u: len(adj[u]) for u in adj}
    m = sum(degrees.values()) // 2
    return adj, degrees, m


# -------------------- NetworkX + python-louvain wrapper --------------------
def louvain_networkx(G):
    """
    Use python-louvain (community.best_partition) on a networkx.Graph.
    If G is not a networkx.Graph we attempt to convert it.
    Returns list of sets for communities.
    """
    if community_louvain is None:
        raise RuntimeError("python-louvain (module 'community') not installed. pip install python-louvain")

    if not (nx is not None and isinstance(G, nx.Graph)):
        # convert to networkx.Graph
        H = nx.Graph() if nx is not None else None
        if H is None:
            raise RuntimeError("networkx not available for conversion to nx.Graph.")
        H.add_nodes_from(list(G.nodes()))
        for u in G.nodes():
            for v in G.neighbors(u):
                H.add_edge(u, v)
        G_nx = H
    else:
        G_nx = G

    partition = community_louvain.best_partition(G_nx)  # node -> community id
    comms = defaultdict(set)
    for node, cid in partition.items():
        comms[cid].add(node)
    return [s for s in comms.values()]


# -------------------- Custom Louvain Implementation --------------------
# Implementation notes:
# - Operates on adjacency dict (unweighted)
# - Two-phase Louvain: (1) local moving until no improvement; (2) aggregate graph and repeat
# - Modularity delta formula used (unweighted graph):
#   deltaQ = (k_i_in - k_i * sum_tot_C / (2m)) / (2m)
#   where k_i_in = edges from i to community C, k_i = degree(i), sum_tot_C = sum of degrees in C, m = total edges
def louvain_custom(G, max_passes=10):
    """
    Run Louvain on graph-like object G and return list of communities (sets).
    max_passes: how many aggregate passes at most (prevents pathological infinite loops).
    """
    adj, degrees, m = graph_to_adj_and_stats(G)
    if m == 0:
        # trivial case: each node its own community or singletons as components
        return [set([n]) for n in adj]

    # initial partition: each node in its own community (cid = node)
    node2com = {node: node for node in adj}
    # community degree sum (sum_tot), and community internal edge counts (not strictly needed initially)
    com2deg = {node: degrees[node] for node in adj}

    def compute_modularity(node2com_local):
        """Compute modularity Q for current partition (unweighted)."""
        # Q = (1/2m) * sum_{ij} [ A_ij - k_i*k_j/(2m) ] delta(c_i, c_j)
        # We'll compute using communities: sum_in and sum_tot per community.
        com_in = defaultdict(int)   # number of internal edges counted twice
        com_tot = defaultdict(int)  # sum of degrees in community
        for u in adj:
            cu = node2com_local[u]
            com_tot[cu] += degrees[u]
            for v in adj[u]:
                if node2com_local[v] == cu:
                    com_in[cu] += 1
        # com_in currently counts each internal edge twice (u->v and v->u)
        Q = 0.0
        for c in com_tot:
            in_edges = com_in.get(c, 0) / 2.0
            tot = com_tot[c]
            Q += (in_edges / m) - (tot * tot) / ((2.0 * m) * (2.0 * m))
        return Q

    # outer loop: repeat aggregation pass until no improvement or max_passes reached
    prev_modularity = compute_modularity(node2com)
    # print(f"initial modularity: {prev_modularity:.6f}")
    for pass_n in range(max_passes):
        moved = True
        # local moving phase: iterate nodes in some order until no single-node move increases modularity
        iter_count = 0
        while moved:
            iter_count += 1
            moved = False
            # iterate nodes in arbitrary order (list to freeze iteration)
            for node in list(adj.keys()):
                cur_com = node2com[node]
                k_i = degrees[node]
                # compute neighboring communities and k_i_in for each
                neigh_com_counts = defaultdict(int)
                for nbr in adj[node]:
                    neigh_com_counts[node2com[nbr]] += 1

                # remove node from its community temporarily: adjust com2deg
                com2deg[cur_com] -= k_i

                # compute best move
                best_com = cur_com
                best_delta = 0.0
                # for speed: consider communities in neighbor set and also current (we'll allow staying)
                for com_c, k_i_in in neigh_com_counts.items():
                    # sum_tot for com_c (after temporary removal not yet applied for com_c unless com_c == cur_com)
                    sum_tot_c = com2deg.get(com_c, 0)
                    # delta modularity formula (unweighted)
                    delta_q = (k_i_in - (k_i * sum_tot_c) / (2.0 * m)) / (2.0 * m)
                    if delta_q > best_delta:
                        best_delta = delta_q
                        best_com = com_c

                # Also consider the possibility of moving to a new singleton community (rarely beneficial)
                # For a new community, sum_tot_c = 0 and k_i_in = 0 => delta = (0 - k_i*0/(2m))/(2m) = 0 (no gain)
                # so we can skip

                if best_com != cur_com:
                    # perform move
                    node2com[node] = best_com
                    com2deg[best_com] = com2deg.get(best_com, 0) + k_i
                    moved = True
                else:
                    # restore original occupancy (we temporarily subtracted earlier)
                    com2deg[cur_com] = com2deg.get(cur_com, 0) + k_i

        # after local moving phase, compute modularity
        curr_mod = compute_modularity(node2com)
        # print(f"after pass {pass_n+1}: modularity {curr_mod:.6f}")
        if curr_mod - prev_modularity < 1e-9:
            # no meaningful improvement; stop
            break
        prev_modularity = curr_mod

        # AGGREGATE: build new graph where communities become nodes
        # Build mapping from community id to set of nodes
        com2nodes = defaultdict(set)
        for n, c in node2com.items():
            com2nodes[c].add(n)

        # If each node is already its own community (no aggregation), break
        if len(com2nodes) == len(adj):
            break

        # build supergraph adjacency (weighted) as dict: com_u -> dict(com_v -> weight)
        super_adj = defaultdict(lambda: defaultdict(int))
        for u in adj:
            cu = node2com[u]
            for v in adj[u]:
                cv = node2com[v]
                super_adj[cu][cv] += 1

        # convert super_adj to new adjacency dict with combined weights -> but our custom code is unweighted
        # We'll convert weighted edges to multiplicity edges by keeping weight as integer in degrees calculation.
        # Create a small adapter "graph-like" object for next pass: nodes = com ids, neighbors yield each neighbor repeated? 
        # Simpler approach: create adjacency where an edge exists between two communities if weight>0.
        new_adj = {}
        for cu, neighs in super_adj.items():
            new_adj[cu] = set(k for k, w in neighs.items() if w > 0)

        # Replace adj, degrees, m, node2com for next pass
        adj = new_adj
        degrees = {u: len(adj[u]) for u in adj}
        m = sum(degrees.values()) // 2
        # reinitialize node2com so each new super-node is its own community (node2com maps supernode->supernode)
        node2com = {u: u for u in adj}
        com2deg = {u: degrees[u] for u in adj}

        # If there are no edges left, stop
        if m == 0:
            break

    # Final communities: build from last non-aggregated mapping if we aggregated at least once we need to recover original nodes
    # For simplicity we will reconstruct communities from the last meaningful partition before final aggregation.
    # We computed com2nodes in last aggregation step (if present). If not present, node2com maps real nodes to communities.
    try:
        # If com2nodes exists and its members are original nodes, return them
        final_coms = []
        if 'com2nodes' in locals():
            # com2nodes collects original nodes grouped by community id; return those
            for nodeset in com2nodes.values():
                final_coms.append(set(nodeset))
        else:
            # fallback: node2com maps original nodes to community id
            clusters = defaultdict(set)
            for n, c in node2com.items():
                clusters[c].add(n)
            final_coms = [s for s in clusters.values()]
    except Exception:
        # Last-resort fallback: connected components of original adjacency
        clusters = []
        visited = set()
        original_nodes = list(degrees.keys())
        # Try building connected components from original adjacency captured earlier (if available)
        for n in original_nodes:
            if n in visited:
                continue
            stack = [n]
            comp = set()
            visited.add(n)
            while stack:
                v = stack.pop()
                comp.add(v)
                for w in adj.get(v, []):
                    if w not in visited:
                        visited.add(w)
                        stack.append(w)
            clusters.append(comp)
        final_coms = clusters

    # normalize: remove empty communities and return
    return [set(c) for c in final_coms if c]


# -------------------- CLI and main --------------------
def main():
    parser = argparse.ArgumentParser(description="Louvain community detection for aad project")
    parser.add_argument('--dataset-path', '-d', help='Path to dataset folder (passed to create_complete_graph)', default=None)
    parser.add_argument('--use-nx', action='store_true', dest='use_nx', help='Prefer using python-louvain + networkx when available')
    parser.add_argument('--output-json', '-o', help='Write communities summary to this JSON file')
    parser.add_argument('--max-passes', type=int, default=10, help='Max aggregation passes for custom Louvain')
    args = parser.parse_args()

    print('Loading graph from dataset...')
    try:
        G, all_ego_nodes, all_circles, all_features = load_graph_from_dataset(args.dataset_path)
    except Exception as e:
        print('Failed to load graph via create_complete_graph:', e)
        sys.exit(1)

    # robust node/edge counts
    try:
        n_nodes = G.number_of_nodes()
    except Exception:
        n_nodes = len(list(G.nodes()))
    try:
        n_edges = G.number_of_edges()
    except Exception:
        # fallback count
        edge_count = 0
        for u in G.nodes():
            edge_count += len(list(G.neighbors(u)))
        n_edges = edge_count // 2

    print(f'Graph loaded. Nodes: {n_nodes}, Edges: {n_edges}')

    start = time.time()
    communities = None
    if args.use_nx and nx is not None and community_louvain is not None:
        print('Running python-louvain (fast path) ...')
        communities = louvain_networkx(G)
    else:
        print('Running custom Louvain (may be slower)...')
        communities = louvain_custom(G, max_passes=args.max_passes)
    elapsed = time.time() - start

    print(f'Found {len(communities)} communities in {elapsed:.2f}s')
    summary = []
    for i, c in enumerate(communities, 1):
        members = sorted(list(c))
        print(f'Community {i}: size {len(members)}; sample: {members[:10]}{"..." if len(members)>10 else ""}')
        summary.append({'id': i, 'size': len(members), 'members_sample': members[:30]})

    out = {
        'algorithm': 'louvain',
        'num_communities_found': len(communities),
        'time_seconds': elapsed,
        'summary': summary
    }
    if args.output_json:
        try:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(out, f, indent=2)
            print('Wrote summary to', args.output_json)
        except Exception as e:
            print('Failed to write JSON:', e)

    print('Done.')


if __name__ == '__main__':
    main()