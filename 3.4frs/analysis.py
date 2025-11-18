"""
Scalability Analysis for Friend Recommendation Algorithms
Tests performance of all heuristic algorithms across 10 different graph sizes.
"""

import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

from cm import recommend_friends as cm_recommend, compute_common_neighbors_score
from aa import recommend_friends as aa_recommend, compute_adamic_adar_score
from jc import recommend_friends as jc_recommend, compute_jaccard_coefficient
from pa import recommend_friends as pa_recommend, compute_preferential_attachment_score
from ra import recommend_friends as ra_recommend, compute_resource_allocation_score


def train_test_split(G, test_ratio=0.2, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    edges = list(G.edges())
    random.shuffle(edges)
    
    split_idx = int(len(edges) * (1 - test_ratio))
    train_edges = edges[:split_idx]
    test_edges_pos = edges[split_idx:]
    
    train_G = nx.Graph()
    train_G.add_nodes_from(G.nodes())
    train_G.add_edges_from(train_edges)
    
    test_edges_neg = []
    nodes = list(G.nodes())
    while len(test_edges_neg) < len(test_edges_pos):
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u != v and not G.has_edge(u, v) and (u, v) not in test_edges_neg and (v, u) not in test_edges_neg:
            test_edges_neg.append((u, v))
    
    return train_G, test_edges_pos, test_edges_neg


def evaluate_algorithm_on_graph(graph_data, algo_name, recommend_func, score_func, top_k=10, sample_size=50):
    """Evaluate a single algorithm on pre-loaded graph data"""
    graph_id, nx_G, train_G, test_edges_pos, test_edges_neg, sample_nodes = graph_data
    
    process = psutil.Process()
    process.memory_info()
    start_memory = process.memory_info().rss / 1024 / 1024
    start_time = time.time()
    
    all_predictions = []
    
    for node in sample_nodes:
        preds = recommend_func(train_G, node, top_k)
        all_predictions.extend([(node, pred_node) for pred_node, _ in preds])
    
    runtime = time.time() - start_time
    end_memory = process.memory_info().rss / 1024 / 1024
    memory_used = max(0.1, end_memory - start_memory)
    
    test_set = set(tuple(sorted(edge)) for edge in test_edges_pos)
    pred_set = set(tuple(sorted(edge)) for edge in all_predictions)
    
    true_positives = len(pred_set.intersection(test_set))
    
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = true_positives / len(test_set) if len(test_set) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    roc_auc = 0.0
    if score_func:
        try:
            test_sample = min(50, len(test_edges_pos))
            scores_pos = [score_func(train_G, u, v) for u, v in test_edges_pos[:test_sample]]
            scores_neg = [score_func(train_G, u, v) for u, v in test_edges_neg[:test_sample]]
            
            from sklearn.metrics import roc_auc_score
            y_true = [1] * len(scores_pos) + [0] * len(scores_neg)
            y_scores = scores_pos + scores_neg
            
            if len(set(y_true)) >= 2:
                roc_auc = roc_auc_score(y_true, y_scores)
        except:
            pass
    
    return {
        'graph_id': graph_id,
        'algorithm': algo_name,
        'nodes': nx_G.number_of_nodes(),
        'edges': nx_G.number_of_edges(),
        'train_edges': train_G.number_of_edges(),
        'test_edges': len(test_edges_pos),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'runtime': runtime,
        'memory_mb': memory_used,
        'true_positives': true_positives,
        'total_predictions': len(pred_set),
        'sample_size': len(sample_nodes)
    }


def load_and_prepare_graph(graph_id, sample_size=50):
    """Load a graph and prepare it for evaluation"""
    try:
        nx_G, _, _, _ = create_complete_graph(graph_id)
        train_G, test_edges_pos, test_edges_neg = train_test_split(nx_G, test_ratio=0.2)
        all_nodes = list(train_G.nodes())
        sample_nodes = random.sample(all_nodes, min(sample_size, len(all_nodes)))
        return (graph_id, nx_G, train_G, test_edges_pos, test_edges_neg, sample_nodes)
    except:
        return None


def plot_scalability_results(df, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    
    algorithms = df['algorithm'].unique()
    colors = {'Common Neighbors': 'blue', 'Adamic-Adar': 'green', 
              'Jaccard Coefficient': 'red', 'Preferential Attachment': 'purple',
              'Resource Allocation': 'orange'}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Scalability Analysis: Algorithm Performance Across Different Graph Sizes', fontsize=16, fontweight='bold')
    
    # 1. Precision vs Graph Size
    ax = axes[0, 0]
    for algo in algorithms:
        data = df[df['algorithm'] == algo].sort_values('nodes')
        ax.plot(data['nodes'], data['precision'], marker='o', label=algo, 
                color=colors.get(algo, 'black'), linewidth=2)
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision vs Graph Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Recall vs Graph Size
    ax = axes[0, 1]
    for algo in algorithms:
        data = df[df['algorithm'] == algo].sort_values('nodes')
        ax.plot(data['nodes'], data['recall'], marker='s', label=algo,
                color=colors.get(algo, 'black'), linewidth=2)
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Recall vs Graph Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. F1 Score vs Graph Size
    ax = axes[0, 2]
    for algo in algorithms:
        data = df[df['algorithm'] == algo].sort_values('nodes')
        ax.plot(data['nodes'], data['f1_score'], marker='^', label=algo,
                color=colors.get(algo, 'black'), linewidth=2)
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score vs Graph Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. ROC-AUC vs Graph Size
    ax = axes[1, 0]
    for algo in algorithms:
        data = df[df['algorithm'] == algo].sort_values('nodes')
        ax.plot(data['nodes'], data['roc_auc'], marker='d', label=algo,
                color=colors.get(algo, 'black'), linewidth=2)
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('ROC-AUC', fontsize=12)
    ax.set_title('ROC-AUC vs Graph Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Runtime vs Graph Size
    ax = axes[1, 1]
    for algo in algorithms:
        data = df[df['algorithm'] == algo].sort_values('nodes')
        ax.plot(data['nodes'], data['runtime'], marker='*', label=algo,
                color=colors.get(algo, 'black'), linewidth=2)
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Runtime vs Graph Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Graph Sizes (Nodes and Edges)
    ax = axes[1, 2]
    graph_data = df.groupby('graph_id').first().sort_values('nodes')
    ax.plot(graph_data['nodes'], graph_data['edges'], marker='o', 
            color='navy', linewidth=2, label='Edges')
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Number of Edges', fontsize=12)
    ax.set_title('Graph Structure (Nodes vs Edges)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'scalability_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()


def main():
    print("="*80)
    print("SCALABILITY ANALYSIS - FRIEND RECOMMENDATION ALGORITHMS")
    print("="*80)
    
    graph_ids = list(range(1, 11))
    
    algorithms = [
        ("Common Neighbors", cm_recommend, compute_common_neighbors_score),
        ("Adamic-Adar", aa_recommend, compute_adamic_adar_score),
        ("Jaccard Coefficient", jc_recommend, compute_jaccard_coefficient),
        ("Preferential Attachment", pa_recommend, compute_preferential_attachment_score),
        ("Resource Allocation", ra_recommend, compute_resource_allocation_score)
    ]
    
    print("\nLoading and preparing graphs...")
    graph_data_list = []
    for graph_id in graph_ids:
        graph_data = load_and_prepare_graph(graph_id, sample_size=50)
        if graph_data:
            graph_data_list.append(graph_data)
            print(f"  Graph {graph_id}: {graph_data[1].number_of_nodes()} nodes, {graph_data[1].number_of_edges()} edges")
    
    print(f"\nRunning {len(algorithms)} algorithms on {len(graph_data_list)} graphs in parallel...")
    
    all_results = []
    tasks = []
    for graph_data in graph_data_list:
        for algo_name, recommend_func, score_func in algorithms:
            tasks.append((graph_data, algo_name, recommend_func, score_func, 10, 50))
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(evaluate_algorithm_on_graph, *task): task for task in tasks}
        
        completed = 0
        total = len(futures)
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"  Progress: {completed}/{total} evaluations completed")
            except Exception as e:
                print(f"  Error processing {task[1]}: {e}")
    
    df = pd.DataFrame(all_results)
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    summary_cols = ['algorithm', 'graph_id', 'nodes', 'edges', 'precision', 'recall', 
                    'f1_score', 'roc_auc', 'runtime', 'memory_mb']
    print(df[summary_cols].to_string(index=False))
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, 'scalability_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {csv_path}")
    
    plot_scalability_results(df, results_dir)
    
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS BY ALGORITHM")
    print(f"{'='*80}\n")
    
    agg_stats = df.groupby('algorithm').agg({
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'runtime': ['mean', 'std']
    }).round(4)
    
    print(agg_stats)
    
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}")
    
    return df


if __name__ == "__main__":
    import networkx as nx
    results_df = main()
