"""
Comprehensive Analysis for Friend Recommendation Algorithms.

Evaluates algorithms across three dimensions:
- Performance: Precision, Recall, F1, ROC-AUC, MAP
- Scalability: Runtime and Memory usage across graph sizes
- Complexity: Theoretical vs Actual time complexity
"""

import os
import sys
import time
import random
import gc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph
from cm import recommend_friends as cm_recommend, compute_common_neighbors_score
from aa import recommend_friends as aa_recommend, compute_adamic_adar_score
from jc import recommend_friends as jc_recommend, compute_jaccard_coefficient
from pa import recommend_friends as pa_recommend, compute_preferential_attachment_score
from ra import recommend_friends as ra_recommend, compute_resource_allocation_score


def train_test_split(G, test_ratio=0.2, seed=92):
    """
    Split graph edges into training and test sets.
    
    Args:
        G: NetworkX graph
        test_ratio: Proportion of edges for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_graph, positive_test_edges, negative_test_edges)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    edges = list(G.edges())
    random.shuffle(edges)
    split_idx = int(len(edges) * (1 - test_ratio))
    
    train_G = nx.Graph()
    train_G.add_nodes_from(G.nodes())
    train_G.add_edges_from(edges[:split_idx])
    
    test_edges_pos = edges[split_idx:]
    test_edges_neg = []
    nodes = list(G.nodes())
    
    while len(test_edges_neg) < len(test_edges_pos):
        u, v = random.choice(nodes), random.choice(nodes)
        if u != v and not G.has_edge(u, v) and (u, v) not in test_edges_neg and (v, u) not in test_edges_neg:
            test_edges_neg.append((u, v))
    
    return train_G, test_edges_pos, test_edges_neg


def calculate_performance_metrics(predictions, test_edges_pos, test_edges_neg, 
                                  train_G, score_func, ranked_predictions, sample_nodes):
    """
    Calculate comprehensive performance metrics for link prediction.
    
    Args:
        predictions: List of predicted edges
        test_edges_pos: Positive test edges
        test_edges_neg: Negative test edges
        train_G: Training graph
        score_func: Scoring function for ROC-AUC calculation
        ranked_predictions: Ranked predictions per node for MAP
        sample_nodes: Sampled nodes for evaluation
        
    Returns:
        Dictionary containing precision, recall, F1, ROC-AUC, MAP, and true positives
    """
    test_set = set(tuple(sorted(edge)) for edge in test_edges_pos)
    pred_set = set(tuple(sorted(edge)) for edge in predictions)
    true_positives = len(pred_set.intersection(test_set))
    
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(test_set) if test_set else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    roc_auc = 0.0
    if score_func:
        try:
            test_sample = min(100, len(test_edges_pos))
            scores_pos = [score_func(train_G, u, v) for u, v in test_edges_pos[:test_sample]]
            scores_neg = [score_func(train_G, u, v) for u, v in test_edges_neg[:test_sample]]
            y_true = [1] * len(scores_pos) + [0] * len(scores_neg)
            y_scores = scores_pos + scores_neg
            if len(set(y_true)) >= 2:
                roc_auc = roc_auc_score(y_true, y_scores)
        except (ValueError, ZeroDivisionError, IndexError):
            roc_auc = 0.0
    
    test_dict = {}
    for u, v in test_edges_pos:
        test_dict.setdefault(u, set()).add(v)
        test_dict.setdefault(v, set()).add(u)
    
    average_precisions = []
    for i, node in enumerate(sample_nodes):
        if node not in test_dict or i >= len(ranked_predictions) or not ranked_predictions[i]:
            continue
        
        relevant_items = test_dict[node]
        hits, precision_sum = 0, 0.0
        
        for k, (pred_node, _) in enumerate(ranked_predictions[i], 1):
            if pred_node in relevant_items:
                hits += 1
                precision_sum += hits / k
        
        if hits > 0:
            average_precisions.append(precision_sum / len(relevant_items))
    
    return {
        'precision': precision, 'recall': recall, 'f1_score': f1_score,
        'roc_auc': roc_auc, 'map': np.mean(average_precisions) if average_precisions else 0.0,
        'true_positives': true_positives
    }


def get_theoretical_complexity(algo_name, n, d_avg):
    """
    Calculate theoretical time complexity for each algorithm.
    
    Args:
        algo_name: Algorithm name
        n: Number of nodes
        d_avg: Average degree
        
    Returns:
        Theoretical complexity value
    """
    complexities = {
        'Preferential Attachment': n,
        'Common Neighbors': n * d_avg**2,
        'Adamic-Adar': n * d_avg**2,
        'Jaccard Coefficient': n * d_avg**2,
        'Resource Allocation': n * d_avg**2
    }
    return complexities.get(algo_name, n * d_avg**2)


def evaluate_algorithm_on_graph(graph_data, algo_name, recommend_func, score_func, top_k=10, sample_size=50):
    """
    Evaluate a single algorithm on pre-loaded graph data.
    
    Args:
        graph_data: Tuple of (graph_id, nx_G, train_G, test_edges_pos, test_edges_neg, sample_nodes)
        algo_name: Name of the algorithm
        recommend_func: Recommendation function
        score_func: Scoring function
        top_k: Number of recommendations per node
        sample_size: Number of nodes to sample
        
    Returns:
        Dictionary containing performance and scalability metrics
    """
    import tracemalloc
    graph_id, nx_G, train_G, test_edges_pos, test_edges_neg, sample_nodes = graph_data
    
    gc.collect()
    tracemalloc.start()
    start_time = time.time()
    
    all_predictions, ranked_predictions = [], []
    
    for node in sample_nodes:
        preds = recommend_func(train_G, node, top_k)
        all_predictions.extend([(node, pred_node) for pred_node, _ in preds])
        ranked_predictions.append(preds)
    
    runtime = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    memory_used = peak / 1024 / 1024
    
    perf_metrics = calculate_performance_metrics(
        all_predictions, test_edges_pos, test_edges_neg, 
        train_G, score_func, ranked_predictions, sample_nodes
    )
    
    d_avg = 2 * nx_G.number_of_edges() / nx_G.number_of_nodes() if nx_G.number_of_nodes() > 0 else 0
    
    return {
        'graph_id': graph_id, 'algorithm': algo_name,
        'nodes': nx_G.number_of_nodes(), 'edges': nx_G.number_of_edges(),
        'avg_degree': d_avg, 'runtime': runtime, 'memory_mb': memory_used,
        'sample_size': len(sample_nodes),
        'theoretical_complexity': get_theoretical_complexity(algo_name, nx_G.number_of_nodes(), d_avg),
        'runtime_per_node': runtime / len(sample_nodes) if sample_nodes else 0,
        **perf_metrics
    }


def load_and_prepare_graph(graph_id, sample_size=50):
    """
    Load a graph and prepare it for evaluation with train/test split.
    
    Args:
        graph_id: Graph identifier
        sample_size: Number of nodes to sample for evaluation
        
    Returns:
        Tuple of prepared graph data or None if loading fails
    """
    try:
        nx_G, _, _, _ = create_complete_graph(graph_id)
        train_G, test_edges_pos, test_edges_neg = train_test_split(nx_G, test_ratio=0.2)
        sample_nodes = random.sample(list(train_G.nodes()), min(sample_size, train_G.number_of_nodes()))
        return (graph_id, nx_G, train_G, test_edges_pos, test_edges_neg, sample_nodes)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error loading graph {graph_id}: {e}")
        return None


def plot_scalability_results(df, results_dir, colors):
    """Generate plots comparing scalability metrics across algorithms."""
    os.makedirs(results_dir, exist_ok=True)
    algorithms = df['algorithm'].unique()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Scalability Analysis: Runtime and Memory Performance', fontsize=16, fontweight='bold')
    
    plot_configs = [
        (axes[0, 0], 'runtime', 'o', 'Runtime (seconds)', 'Runtime vs Graph Size'),
        (axes[0, 1], 'memory_mb', 's', 'Memory (MB)', 'Memory Usage vs Graph Size'),
        (axes[0, 2], 'runtime_per_node', '^', 'Runtime per Node (seconds)', 'Runtime per Node vs Graph Size')
    ]
    
    for ax, metric, marker, ylabel, title in plot_configs:
        for algo in algorithms:
            data = df[df['algorithm'] == algo].sort_values('nodes')
            ax.plot(data['nodes'], data[metric], marker=marker, label=algo, 
                   color=colors.get(algo, 'black'), linewidth=2)
        ax.set_xlabel('Number of Nodes', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for algo in algorithms:
        data = df[df['algorithm'] == algo].sort_values('nodes')
        if len(data) > 0:
            theoretical_norm = data['theoretical_complexity'] / data['theoretical_complexity'].max()
            actual_norm = data['runtime'] / data['runtime'].max()
            ax.plot(data['nodes'], theoretical_norm, marker='o', linestyle='--', 
                   label=f'{algo} (Theory)', color=colors.get(algo, 'black'), alpha=0.5)
            ax.plot(data['nodes'], actual_norm, marker='s', linestyle='-', 
                   label=f'{algo} (Actual)', color=colors.get(algo, 'black'), linewidth=2)
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Normalized Complexity', fontsize=12)
    ax.set_title('Theoretical vs Actual Complexity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    graph_data = df.groupby('graph_id').first().sort_values('nodes')
    axes[1, 1].plot(graph_data['nodes'], graph_data['avg_degree'], marker='d', 
                    color='navy', linewidth=2, label='Average Degree')
    axes[1, 1].set_xlabel('Number of Nodes', fontsize=12)
    axes[1, 1].set_ylabel('Average Degree', fontsize=12)
    axes[1, 1].set_title('Graph Density (Average Degree)', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(graph_data['nodes'], graph_data['edges'], marker='o', 
                    color='navy', linewidth=2, label='Edges')
    axes[1, 2].set_xlabel('Number of Nodes', fontsize=12)
    axes[1, 2].set_ylabel('Number of Edges', fontsize=12)
    axes[1, 2].set_title('Graph Structure (Nodes vs Edges)', fontsize=14, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'scalability_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_theoretical_vs_actual(df, results_dir, algorithms, colors):
    """Create detailed plots comparing theoretical vs actual complexity for each algorithm."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Theoretical vs Actual Time Complexity Analysis', fontsize=16, fontweight='bold')
    algo_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    
    for idx, algo in enumerate(algorithms[:5]):
        row, col = algo_positions[idx]
        ax = axes[row, col]
        data = df[df['algorithm'] == algo].sort_values('nodes')
        
        if len(data) > 0:
            theoretical_norm = data['theoretical_complexity'] / data['theoretical_complexity'].max()
            actual_norm = data['runtime'] / data['runtime'].max()
            
            ax.plot(data['nodes'], theoretical_norm, marker='o', linestyle='--', 
                   label='Theoretical', color=colors.get(algo, 'black'), alpha=0.6, linewidth=2)
            ax.plot(data['nodes'], actual_norm, marker='s', linestyle='-', 
                   label='Actual', color=colors.get(algo, 'black'), linewidth=2.5)
            
            if len(data) > 1:
                correlation = np.corrcoef(theoretical_norm, actual_norm)[0, 1]
                ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Number of Nodes', fontsize=11)
        ax.set_ylabel('Normalized Complexity', fontsize=11)
        ax.set_title(f'{algo}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    for algo in algorithms:
        data = df[df['algorithm'] == algo].sort_values('nodes')
        if len(data) > 0:
            theoretical_norm = data['theoretical_complexity'] / data['theoretical_complexity'].max()
            actual_norm = data['runtime'] / data['runtime'].max()
            ratio = actual_norm / theoretical_norm
            ax.plot(data['nodes'], ratio, marker='o', label=algo, color=colors.get(algo, 'black'), linewidth=2)
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Perfect Match', alpha=0.7)
    ax.set_xlabel('Number of Nodes', fontsize=11)
    ax.set_ylabel('Actual / Theoretical Ratio', fontsize=11)
    ax.set_title('Algorithm Efficiency (All)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'theoretical_vs_actual_complexity.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_metrics(df, results_dir, algorithms, colors):
    """Create plots for performance metrics vs graph size."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Performance Metrics vs Graph Size', fontsize=16, fontweight='bold')
    
    metrics = [
        ('precision', 'Precision', (0, 0)), ('recall', 'Recall', (0, 1)),
        ('f1_score', 'F1 Score', (0, 2)), ('roc_auc', 'ROC-AUC', (1, 0)), ('map', 'MAP', (1, 1))
    ]
    
    for metric_key, metric_name, (row, col) in metrics:
        ax = axes[row, col]
        for algo in algorithms:
            data = df[df['algorithm'] == algo].sort_values('nodes')
            if len(data) > 0:
                ax.plot(data['nodes'], data[metric_key], marker='o', label=algo,
                       color=colors.get(algo, 'black'), linewidth=2)
        ax.set_xlabel('Number of Nodes', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} vs Graph Size', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    for algo in algorithms:
        data = df[df['algorithm'] == algo].sort_values('nodes')
        if len(data) > 0:
            norm_precision = data['precision'] / data['precision'].max() if data['precision'].max() > 0 else data['precision']
            norm_recall = data['recall'] / data['recall'].max() if data['recall'].max() > 0 else data['recall']
            norm_f1 = data['f1_score'] / data['f1_score'].max() if data['f1_score'].max() > 0 else data['f1_score']
            avg_performance = (norm_precision + norm_recall + norm_f1) / 3
            ax.plot(data['nodes'], avg_performance, marker='o', label=algo,
                   color=colors.get(algo, 'black'), linewidth=2)
    
    ax.set_xlabel('Number of Nodes', fontsize=11)
    ax.set_ylabel('Average Normalized Performance', fontsize=11)
    ax.set_title('Overall Performance vs Graph Size', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance_metrics_vs_size.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function for comprehensive analysis."""
    print("  Analysis of Friend Recommendation Algorithms")
    print("="*80 + "\n")
    
    algorithms = [
        ("Common Neighbors", cm_recommend, compute_common_neighbors_score),
        ("Adamic-Adar", aa_recommend, compute_adamic_adar_score),
        ("Jaccard Coefficient", jc_recommend, compute_jaccard_coefficient),
        ("Preferential Attachment", pa_recommend, compute_preferential_attachment_score),
        ("Resource Allocation", ra_recommend, compute_resource_allocation_score)
    ]
    
    print("Loading and Preparing Graphs...")
    print("-" * 80)
    graph_data_list = []
    for gid in tqdm(range(1, 11), desc="Loading graphs", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        g = load_and_prepare_graph(gid, 50)
        if g:
            graph_data_list.append(g)
    
    print("\nGraph Statistics:")
    for g in graph_data_list:
        print(f"  • Graph {g[0]:2d}: {g[1].number_of_nodes():4d} nodes, {g[1].number_of_edges():5d} edges")
    
    print(f"\nRunning Algorithms")
    print("-" * 80)
    print(f"  Algorithms: {len(algorithms)}")
    print(f"  Graphs: {len(graph_data_list)}")
    print(f"  Total evaluations: {len(algorithms) * len(graph_data_list)}\n")
    
    all_results = []
    pbar = tqdm(total=len(graph_data_list) * len(algorithms), 
                desc="Evaluating", ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for g in graph_data_list:
        for name, rec, score in algorithms:
            try:
                result = evaluate_algorithm_on_graph(g, name, rec, score, 10, 50)
                if result:
                    all_results.append(result)
                pbar.set_postfix_str(f"{name[:20]:<20} | Graph {g[0]}")
                pbar.update(1)
            except Exception as e:
                pbar.write(f"  ⚠ Error: {name} on Graph {g[0]} - {str(e)[:50]}")
                pbar.update(1)
    
    pbar.close()
    
    df = pd.DataFrame(all_results)
    
    print(f"\nResults Summary")
    print("=" * 80)
    summary_cols = ['algorithm', 'graph_id', 'nodes', 'precision', 'recall', 'f1_score', 'runtime', 'memory_mb']
    print(df[summary_cols].sort_values(['algorithm', 'graph_id']).to_string(index=False))
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_file = os.path.join(results_dir, 'comprehensive_analysis.csv')
    df.to_csv(csv_file, index=False)
    print(f"\nResults saved: {os.path.basename(csv_file)}")
    
    print(f"\nGenerating Visualizations")
    print("-" * 80)
    colors = {'Common Neighbors': 'blue', 'Adamic-Adar': 'green', 
              'Jaccard Coefficient': 'red', 'Preferential Attachment': 'purple',
              'Resource Allocation': 'orange'}
    algo_names = [algo[0] for algo in algorithms]
    
    plots = [
        ("Scalability Analysis", lambda: plot_scalability_results(df, results_dir, colors)),
        ("Theoretical vs Actual", lambda: plot_theoretical_vs_actual(df, results_dir, algo_names, colors)),
        ("Performance Metrics", lambda: plot_performance_metrics(df, results_dir, algo_names, colors))
    ]
    
    for plot_name, plot_func in tqdm(plots, desc="Creating plots", ncols=80, 
                                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        plot_func()
    
    print(f"\nStatistical Analysis")
    print("=" * 80)
    
    print("\nAggregate Statistics by Algorithm:")
    print("-" * 80)
    agg_stats = df.groupby('algorithm').agg({
        'precision': ['mean', 'std'], 'recall': ['mean', 'std'],
        'f1_score': ['mean', 'std'], 'roc_auc': ['mean', 'std'],
        'map': ['mean', 'std'], 'runtime': ['mean', 'std'], 'memory_mb': ['mean', 'std']
    }).round(4)
    print(agg_stats)
    
    print(f"\nTheoretical vs Actual Complexity Correlation:")
    print("-" * 80)
    for algo in sorted(df['algorithm'].unique()):
        algo_data = df[df['algorithm'] == algo].sort_values('nodes')
        if len(algo_data) > 1:
            correlation = np.corrcoef(algo_data['theoretical_complexity'], algo_data['runtime'])[0, 1]
            print(f"  {algo:30s}: {correlation:6.4f}")
    
    print(f"\nBest Performers")
    print("=" * 80)
    avg_by_algo = df.groupby('algorithm').mean()
    metrics = [
        ('Highest Precision', 'precision', 'max'), ('Highest Recall', 'recall', 'max'),
        ('Highest F1 Score', 'f1_score', 'max'), ('Highest ROC-AUC', 'roc_auc', 'max'),
        ('Highest MAP', 'map', 'max'), ('Fastest Runtime', 'runtime', 'min'),
        ('Lowest Memory', 'memory_mb', 'min')
    ]
    
    for label, metric, func in metrics:
        best_val = avg_by_algo[metric].max() if func == 'max' else avg_by_algo[metric].min()
        best_algo = avg_by_algo[metric].idxmax() if func == 'max' else avg_by_algo[metric].idxmin()
        unit = 's' if 'Runtime' in label else ('MB' if 'Memory' in label else '')
        print(f"  {label:20s}: {best_algo:30s} ({best_val:.4f}{unit})")
    
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print("="*80 + "\n")
    
    return df


if __name__ == "__main__":
    results_df = main()
