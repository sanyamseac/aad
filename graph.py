import networkx as nx
import numpy as np
import os
import sys
import glob

def load_ego_features(filepath):
    """Load ego node features from .egofeat file"""
    with open(filepath, 'r') as f:
        features = [int(x) for x in f.read().strip().split()]
    return np.array(features)

def load_feature_names(filepath):
    """Load feature names from .featnames file"""
    feature_names = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                idx, name = parts
                feature_names[int(idx)] = name
    return feature_names

def load_circles(filepath):
    """Load circles (communities) from .circles file"""
    circles = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                circle_name = parts[0]
                nodes = [int(x) for x in parts[1:]]
                circles[circle_name] = nodes
    return circles

def load_edges(filepath):
    """Load edges from .edges file"""
    edges = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                edges.append((int(parts[0]), int(parts[1])))
    return edges

def load_node_features(filepath):
    """Load node features from .feat file"""
    node_features = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                node_id = int(parts[0])
                features = np.array([int(x) for x in parts[1:]])
                node_features[node_id] = features
    return node_features

def create_complete_graph(num_files=None, dataset_path = os.path.join(sys.path[0], 'dataset')):
    """Create complete Facebook graph from all ego networks"""
    G = nx.Graph()
    all_ego_nodes = []
    all_circles = {}
    all_features = {}
    
    # Find all ego network files
    ego_files = glob.glob(os.path.join(dataset_path, "*.edges"))
    
    # Sort files alphabetically by filename
    ego_files = sorted(ego_files)
    
    # Select only the specified number of files if num_files is provided
    if num_files is not None and num_files > 0:
        ego_files = ego_files[:num_files]
    
    for ego_file in ego_files:
        # Extract ego node ID from filename
        ego_id = int(os.path.basename(ego_file).split('.')[0])
        all_ego_nodes.append(ego_id)
        
        # Load edges
        edges = load_edges(ego_file)
        
        # Add ego node
        G.add_node(ego_id, node_type='ego')
        
        # Track all alter nodes that appear in edges
        alter_nodes_in_network = set()
        
        # Add edges from .edges file (connections between alters)
        for node1, node2 in edges:
            G.add_edge(node1, node2)
            alter_nodes_in_network.add(node1)
            alter_nodes_in_network.add(node2)
        
        # Connect ego to ALL alter nodes (including those from .feat file)
        feat_file = ego_file.replace('.edges', '.feat')
        if os.path.exists(feat_file):
            node_features = load_node_features(feat_file)
            for node_id in node_features.keys():
                alter_nodes_in_network.add(node_id)
        
        # Now connect ego to all alters
        for alter_id in alter_nodes_in_network:
            G.add_edge(ego_id, alter_id)

        # Load ego features if available
        egofeat_file = ego_file.replace('.edges', '.egofeat')
        if os.path.exists(egofeat_file):
            ego_features = load_ego_features(egofeat_file)
            all_features[ego_id] = ego_features
            G.nodes[ego_id]['features'] = ego_features
        
        # Load node features if available
        if os.path.exists(feat_file):
            node_features = load_node_features(feat_file)
            for node_id, features in node_features.items():
                if node_id in G.nodes:
                    G.nodes[node_id]['features'] = features

        # Load circles if available
        circles_file = ego_file.replace('.edges', '.circles')
        if os.path.exists(circles_file):
            circles = load_circles(circles_file)
            all_circles[ego_id] = circles
    
    return G, all_ego_nodes, all_circles, all_features

if __name__ == "__main__":
    print("Loading complete graph from dataset...")
    G, ego_nodes, circles, features = create_complete_graph()
    print("graph", G)