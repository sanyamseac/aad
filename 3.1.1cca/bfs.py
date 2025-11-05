import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

def bfs(G):
    pass

if __name__ == "__main__":
    G, all_ego_nodes, all_circles, all_features = create_complete_graph()
    print(G)
    bfs(G)