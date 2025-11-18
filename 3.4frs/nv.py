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
    def __init__(self, nx_G, p, q):
        self.G = nx_G
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
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
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Simulating random walks...')
        for walk_iter in tqdm(range(num_walks), desc="Walk iterations"):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
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
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
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
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
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
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

def learn_embeddings(walks, size=128, window=10, workers=8, iter=5):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(sentences=walks, vector_size=size, window=window, workers=workers, epochs=iter, min_count=0, sg=1) #sg 1 for skip gram and min_count 0 to consider all nodes
	
	return model

def get_edge_score(model, edge):
	'''
	Compute cosine similarity between node embeddings.
	'''
	u, v = edge
	try:
		u_emb = model.wv[str(u)]
		v_emb = model.wv[str(v)]
		score = np.dot(u_emb, v_emb) / (np.linalg.norm(u_emb) * np.linalg.norm(v_emb))
		return score
	except KeyError:
		return 0.0

def recommend_friends(model, node, G, top_k=10):
	'''
	Recommend top-k friends for a given node.
	'''
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
	'''
	Train Node2Vec model on a graph.
	
	Args:
		G: NetworkX graph
		p: Return parameter
		q: In-out parameter
		num_walks: Number of walks per node
		walk_length: Length of each walk
		
	Returns:
		Trained Word2Vec model
	'''
	print(f"Training Node2Vec with p={p}, q={q}")
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
