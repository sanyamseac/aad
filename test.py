import networkx as nx
import sys
import time
# Import the function from the provided graph.py
from graph import create_complete_graph

def analyze_diameters():
    print(f"{'Files':<6} | {'Nodes':<6} | {'Edges':<8} | {'Diameter':<8} | {'# Pairs (Count)':<15}")
    print("-" * 60)

    # Iterate from num_files = 1 to 10
    for i in range(1, 10):
        try:
            # 1. Generate the graph
            # We capture the graph G; we ignore ego_nodes, circles, features for this specific task
            G, _, _, _ = create_complete_graph(num_files=i)
            
            # 2. Handle Connectivity
            # Diameter is strictly defined only for connected graphs. 
            # If the graph is disconnected, we analyze the Largest Connected Component (LCC).
            if nx.is_connected(G):
                subgraph = G
                conn_status = ""
            else:
                # Get the largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc).copy()
                conn_status = "*" # Mark that we used LCC
            
            # 3. Calculate Diameter
            # This calculates the maximum eccentricity
            diameter_val = nx.diameter(subgraph)
            
            # 4. Calculate "Number of Diameters" (Count of pairs with length == diameter)
            # We must compute all pairs shortest paths. 
            # Note: This is computationally intensive (O(V^2)) but feasible for these ego networks.
            diametral_path_count = 0
            
            # Get dictionary of all shortest paths: {source: {target: length}}
            #all_paths = dict(nx.all_pairs_shortest_path_length(subgraph))
            
            # for u in all_paths:
            #     for v, length in all_paths[u].items():
            #         # Ensure we only count unique pairs (u, v) and not (v, u), and avoid self-loops
            #         if u < v: 
            #             if length == diameter_val:
            #                 diametral_path_count += 1

            # 5. Output Results
            print(f"{i:<6} | {subgraph.number_of_nodes():<6} | {subgraph.number_of_edges():<8} | {diameter_val:<8}")

        except Exception as e:
            print(f"Error processing num_files={i}: {e}")

    print("-" * 60)
    print("* Indicates the graph was disconnected; metrics calculated on the Largest Connected Component.")

if __name__ == "__main__":
    analyze_diameters()