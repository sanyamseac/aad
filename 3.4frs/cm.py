import os
import sys
import networkx as nx
import random
import questionary

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

def compute_common_neighbors_score(G, u, v):
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    return len(neighbors_u.intersection(neighbors_v))

def recommend_friends(G, node, top_k=10):
    if node not in G:
        return []
    
    scores = []
    neighbors = set(G.neighbors(node))
    
    for candidate in G.nodes():
        if candidate != node and candidate not in neighbors:
            score = compute_common_neighbors_score(G, node, candidate)
            if score > 0:
                scores.append((candidate, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def cm_main(nx_G, nodes=None, top_k=5):
    if nodes is not None:
        for n in nodes:
            friends = recommend_friends(nx_G, n, top_k)
            return [rec_node for rec_node, _ in friends]

if __name__ == "__main__":
    nx_G, _, _, _ = create_complete_graph(1)
    
    choice = questionary.select(
        "How would you like to select nodes for recommendations?",
        choices=[
            "Use randomly selected sample nodes",
            "Enter a specific node ID"
        ]
    ).ask()
    
    if choice == "Use randomly selected sample nodes":
        sample_nodes = random.sample(list(nx_G.nodes()), min(3, nx_G.number_of_nodes()))
        print("\nSample Friend Recommendations:")
        
        for node in sample_nodes:
            recommendations = recommend_friends(nx_G, node, top_k=5)
            print(f"\nTop 5 friend recommendations for node {node}:")
            if recommendations:
                for rec_node, score in recommendations:
                    print(f"  → Node {rec_node}: {score} common neighbors")
            else:
                print(f"  No recommendations available for node {node}")
            
    elif choice == "Enter a specific node ID":
        node_input = questionary.text(f"Enter node ID:").ask()
        
        try:
            node = int(node_input)
            if node in nx_G.nodes():
                print("\nFriend Recommendations:")
                recommendations = recommend_friends(nx_G, node, top_k=5)
                print(f"\nTop 5 friend recommendations for node {node}:")
                if recommendations:
                    for rec_node, score in recommendations:
                        print(f"  → Node {rec_node}: {score} common neighbors")
                else:
                    print(f"  No recommendations available for node {node}")
            else:
                print(f"Error: Node {node} does not exist in the graph.")
        except ValueError:
            print("Error: Please enter a valid integer node ID.")
    
    else:
        print("No valid choice made. Exiting.")
        exit(1)