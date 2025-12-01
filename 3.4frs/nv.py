"""Node2Vec: Scalable Feature Learning for Networks.

This module implements the Node2Vec algorithm for learning node embeddings
using biased random walks and Word2Vec. Node2Vec can capture both homophily
(nodes in communities) and structural equivalence (nodes with similar roles).

References:
    Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning
    for networks. KDD.

Time Complexity: O(r * n * l * d) for training, O(n * d) for inference
Space Complexity: O(n * d) for embeddings
"""

from gensim.models import Word2Vec
import os
import sys
import numpy as np
import random
import networkx as nx
import questionary
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

class Graph():
    """Graph wrapper for Node2Vec random walks.
    
    Attributes:
        G (networkx.Graph): The input graph.
        p (float): Return parameter (controls backtracking).
        q (float): In-out parameter (controls BFS vs DFS).
        alias_nodes (dict): Alias sampling tables for nodes.
        alias_edges (dict): Alias sampling tables for edges.
    """
    
    def __init__(self, nx_G, p, q):
        """Initialize Graph with biased walk parameters.
        
        Args:
            nx_G (networkx.Graph): Input graph.
            p (float): Return parameter (low p = stay local).
            q (float): In-out parameter (low q = BFS, high q = DFS).
        """
        self.G = nx_G
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        """Simulate a biased random walk starting from a node.
        
        Args:
            walk_length (int): Number of steps in the walk.
            start_node (int): Starting node ID.
            
        Returns:
            list: Sequence of node IDs visited.
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
                        alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        """Generate random walks from all nodes.
        
        Args:
            num_walks (int): Number of walks to start from each node.
            walk_length (int): Length of each walk.
            
        Returns:
            list: List of walks (each walk is a list of node IDs).
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for walk_iter in tqdm(range(num_walks), desc="Walk iterations"):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        """Compute biased transition probabilities for an edge.
        
        Args:
            src (int): Source node.
            dst (int): Destination node.
            
        Returns:
            tuple: (J, q) alias sampling tables.
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(1.0/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(1.0)
            else:
                unnormalized_probs.append(1.0/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """Precompute transition probabilities for efficient walk generation.
        
        Uses alias sampling for O(1) sampling from discrete distributions.
        Computes probabilities for all nodes and edges.
        """
        G = self.G

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [1.0 for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    """Build alias sampling tables for efficient discrete sampling.
    
    The alias method enables O(1) sampling from discrete distributions.
    
    Args:
        probs (list): Probability distribution (must sum to 1).
        
    Returns:
        tuple: (J, q) where J is alias table, q is probability table.
        
    Reference:
        https://hips.seas.harvard.edu/blog/2013/03/03/
        the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    """Sample from discrete distribution using alias method.
    
    Args:
        J (np.ndarray): Alias table.
        q (np.ndarray): Probability table.
        
    Returns:
        int: Sampled index.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

def learn_embeddings(walks, size=128, window=10, workers=8, iter=5):
	"""Train Word2Vec Skip-Gram model on random walks.
	
	Args:
		walks (list): List of walks (sequences of node IDs).
		size (int): Embedding dimensionality. Defaults to 128.
		window (int): Context window size. Defaults to 10.
		workers (int): Number of parallel workers. Defaults to 8.
		iter (int): Training epochs. Defaults to 5.
		
	Returns:
		Word2Vec: Trained model with node embeddings.
	"""
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(sentences=walks, vector_size=size, window=window, workers=workers, epochs=iter, min_count=0, sg=1) #sg 1 for skip gram and min_count 0 to consider all nodes
	
	return model

def get_edge_score(model, edge):
	"""Calculate similarity score for a potential edge.
	
	Uses cosine similarity between node embeddings.
	
	Args:
		model (Word2Vec): Trained Node2Vec model.
		edge (tuple): (u, v) node pair.
		
	Returns:
		float: Cosine similarity in range [-1, 1].
	"""
	u, v = edge
	try:
		u_emb = model.wv[str(u)]
		v_emb = model.wv[str(v)]
		score = np.dot(u_emb, v_emb) / (np.linalg.norm(u_emb) * np.linalg.norm(v_emb))
		return score
	except KeyError:
		return 0.0

def recommend_friends(model, node, G, top_k=10):
	"""Generate friend recommendations using Node2Vec embeddings.
	
	Args:
		model (Word2Vec): Trained Node2Vec model.
		node (int): Target node for recommendations.
		G (networkx.Graph): The graph structure.
		top_k (int, optional): Number of recommendations. Defaults to 10.
		
	Returns:
		list: List of (node_id, similarity_score) tuples sorted descending.
	"""
	try:
		node_emb = model.wv[str(node)]
	except KeyError:
		return []
	
	recommendations = []
	for other_node in G.nodes():
		if other_node != node and not G.has_edge(node, other_node):
			try:
				other_emb = model.wv[str(other_node)]
				score = np.dot(node_emb, other_emb) / (np.linalg.norm(node_emb) * np.linalg.norm(other_emb))
				recommendations.append((other_node, score))
			except KeyError:
				continue
	
	recommendations.sort(key=lambda x: x[1], reverse=True)
	return recommendations[:top_k]

def train_node2vec_model(G, p=0.7, q=0.7, num_walks=10, walk_length=80):
	"""Train complete Node2Vec model on a graph.
	
	Executes the full pipeline: preprocess transitions, generate walks,
	train Word2Vec embeddings.
	
	Args:
		G (networkx.Graph): Input graph.
		p (float, optional): Return parameter. Defaults to 0.7.
		    Low p = stay local, high p = explore.
		q (float, optional): In-out parameter. Defaults to 0.7.
		    Low q = BFS (communities), high q = DFS (roles).
		num_walks (int, optional): Walks per node. Defaults to 10.
		walk_length (int, optional): Steps per walk. Defaults to 80.
		
	Returns:
		Word2Vec: Trained model containing node embeddings.
	"""
	graph_obj = Graph(G, p, q)
	graph_obj.preprocess_transition_probs()
	walks = graph_obj.simulate_walks(num_walks, walk_length)
	model = learn_embeddings(walks)
	return model

if __name__ == "__main__":
    nx_G, _, _, _ = create_complete_graph(1)
    
    choice = questionary.select(
        "Which algo would you like to use?",
        choices=[
            "Node2Vec Algorithm",
            "DeepWalk Algorithm",
            "Advanced (Custom Parameters)"
        ]
    ).ask()
    
    if choice == "Node2Vec Algorithm":
        model = train_node2vec_model(nx_G, p=0.7, q=0.7, num_walks=10, walk_length=80)
    elif choice == "DeepWalk Algorithm":
        model = train_node2vec_model(nx_G, p=1.0, q=1.0, num_walks=10, walk_length=80)
    elif choice == "Advanced (Custom Parameters)":
        p = float(questionary.text(
            "Enter return parameter (p):",
            default="0.7",
            validate=lambda x: x.replace('.', '', 1).isdigit() and float(x) > 0
        ).ask())
        
        q = float(questionary.text(
            "Enter in-out parameter (q):",
            default="0.7",
            validate=lambda x: x.replace('.', '', 1).isdigit() and float(x) > 0
        ).ask())
        
        num_walks = int(questionary.text(
            "Enter number of walks per node:",
            default="10",
            validate=lambda x: x.isdigit() and int(x) > 0
        ).ask())
        
        walk_length = int(questionary.text(
            "Enter walk length:",
            default="80",
            validate=lambda x: x.isdigit() and int(x) > 0
        ).ask())
        
        print(f"\nTraining with custom parameters: p={p}, q={q}, num_walks={num_walks}, walk_length={walk_length}")
        model = train_node2vec_model(nx_G, p=p, q=q, num_walks=num_walks, walk_length=walk_length)
    else:
        print("No valid choice made. Exiting.")
        exit(1)
    
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
            recommendations = recommend_friends(model, node, nx_G, top_k=5)
            print(f"\nTop 5 friend recommendations for node {node}:")
            if recommendations:
                for rec_node, score in recommendations:
                    print(f"  → Node {rec_node}: similarity score = {score:.4f}")
            else:
                print(f"  No recommendations available for node {node}")
            
    elif choice == "Enter a specific node ID":
        node_input = questionary.text(f"Enter node ID:").ask()
        
        try:
            node = int(node_input)
            if node in nx_G.nodes():
                print("\nFriend Recommendations:")
                recommendations = recommend_friends(model, node, nx_G, top_k=5)
                print(f"\nTop 5 friend recommendations for node {node}:")
                if recommendations:
                    for rec_node, score in recommendations:
                        print(f"  → Node {rec_node}: similarity score = {score:.4f}")
                else:
                    print(f"  No recommendations available for node {node}")
            else:
                print(f"Error: Node {node} does not exist in the graph.")
        except ValueError:
            print("Error: Please enter a valid integer node ID.")
    
    else:
        print("No valid choice made. Exiting.")
        exit(1)
