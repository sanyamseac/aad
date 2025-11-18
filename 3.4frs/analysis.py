"""
Comprehensive Analysis for Friend Recommendation Algorithms
- Performance Metrics: Precision, Recall, F1, ROC-AUC, MAP across graph sizes
- Scalability Metrics: Runtime and Memory across graph sizes
- Complexity Analysis: Theoretical vs Actual time complexity comparison
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
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

from cm import recommend_friends as cm_recommend, compute_common_neighbors_score
from aa import recommend_friends as aa_recommend, compute_adamic_adar_score
from jc import recommend_friends as jc_recommend, compute_jaccard_coefficient
from pa import recommend_friends as pa_recommend, compute_preferential_attachment_score
from ra import recommend_friends as ra_recommend, compute_resource_allocation_score


def train_test_split(G, test_ratio=0.2, seed=42):
    """Split graph edges into train and test sets"""
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


def calculate_performance_metrics(predictions, test_edges_pos, test_edges_neg, 
                                  train_G, score_func, ranked_predictions, sample_nodes):
    """Calculate all performance metrics"""
    # Precision, Recall, F1
    test_set = set(tuple(sorted(edge)) for edge in test_edges_pos)
    pred_set = set(tuple(sorted(edge)) for edge in predictions)
    true_positives = len(pred_set.intersection(test_set))
    
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = true_positives / len(test_set) if len(test_set) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # ROC-AUC
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
        except (ValueError, ZeroDivisionError, IndexError) as e:
            print(f"Warning: ROC-AUC calculation failed: {e}")
            roc_auc = 0.0
    
    # MAP
    test_dict = {}
    for u, v in test_edges_pos:
        if u not in test_dict:
            test_dict[u] = set()
        if v not in test_dict:
            test_dict[v] = set()
        test_dict[u].add(v)
        test_dict[v].add(u)
    
    average_precisions = []
    for i, node in enumerate(sample_nodes):
        if node not in test_dict or i >= len(ranked_predictions) or len(ranked_predictions[i]) == 0:
            continue
            
        relevant_items = test_dict[node]
        predictions_list = ranked_predictions[i]
        
        hits = 0
        precision_sum = 0.0
        
        for k, (pred_node, _) in enumerate(predictions_list, 1):
            if pred_node in relevant_items:
                hits += 1
                precision_sum += hits / k
        
        if hits > 0:
            average_precisions.append(precision_sum / len(relevant_items))
    
    map_score = np.mean(average_precisions) if average_precisions else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'map': map_score,
        'true_positives': true_positives
    }


def get_theoretical_complexity(algo_name, n, m, k, d_avg):
    """
    Calculate theoretical time complexity for each algorithm.
    
    Args:
        algo_name: Algorithm name
        n: Number of nodes
        m: Number of edges
        k: Top-k recommendations
        d_avg: Average degree
    
    Returns:
        Theoretical complexity value (normalized)
    """
    complexities = {
        'Common Neighbors': n * d_avg**2,  # O(n * d^2) for all node pairs
        'Adamic-Adar': n * d_avg**2,       # O(n * d^2) similar to CN but with log
        'Jaccard Coefficient': n * d_avg**2,  # O(n * d^2) 
        'Preferential Attachment': n,      # O(n) just degree multiplication
        'Resource Allocation': n * d_avg**2   # O(n * d^2) similar to AA
    }
    return complexities.get(algo_name, n * d_avg**2)


def evaluate_algorithm_on_graph(graph_data, algo_name, recommend_func, score_func, top_k=10, sample_size=50):
    """Evaluate a single algorithm on pre-loaded graph data - both performance and scalability"""
    graph_id, nx_G, train_G, test_edges_pos, test_edges_neg, sample_nodes = graph_data
    
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024
    start_time = time.time()
    peak_memory = start_memory
    
    # Run recommendations and collect predictions
    all_predictions = []
    ranked_predictions = []
    
    for node in sample_nodes:
        preds = recommend_func(train_G, node, top_k)
        all_predictions.extend([(node, pred_node) for pred_node, _ in preds])
        ranked_predictions.append(preds)
        # Track peak memory during execution
        current_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, current_memory)
    
    runtime = time.time() - start_time
    # Use peak memory increment as the memory usage metric
    memory_used = max(0.1, peak_memory - start_memory)
    
    # Calculate performance metrics
    perf_metrics = calculate_performance_metrics(
        all_predictions, test_edges_pos, test_edges_neg, 
        train_G, score_func, ranked_predictions, sample_nodes
    )
    
    # Calculate average degree for theoretical complexity
    d_avg = 2 * nx_G.number_of_edges() / nx_G.number_of_nodes() if nx_G.number_of_nodes() > 0 else 0
    
    theoretical_complexity = get_theoretical_complexity(
        algo_name, 
        nx_G.number_of_nodes(), 
        nx_G.number_of_edges(), 
        top_k, 
        d_avg
    )
    
    return {
        'graph_id': graph_id,
        'algorithm': algo_name,
        'nodes': nx_G.number_of_nodes(),
        'edges': nx_G.number_of_edges(),
        'avg_degree': d_avg,
        'runtime': runtime,
        'memory_mb': memory_used,
        'sample_size': len(sample_nodes),
        'theoretical_complexity': theoretical_complexity,
        'runtime_per_node': runtime / len(sample_nodes) if len(sample_nodes) > 0 else 0,
        **perf_metrics  # Add all performance metrics
    }


def load_and_prepare_graph(graph_id, sample_size=50):
    """Load a graph and prepare it for evaluation with train/test split"""
    try:
        nx_G, _, _, _ = create_complete_graph(graph_id)
        train_G, test_edges_pos, test_edges_neg = train_test_split(nx_G, test_ratio=0.2)
        all_nodes = list(train_G.nodes())
        sample_nodes = random.sample(all_nodes, min(sample_size, len(all_nodes)))
        return (graph_id, nx_G, train_G, test_edges_pos, test_edges_neg, sample_nodes)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error loading graph {graph_id}: {e}")
        return None


def plot_scalability_results(df, results_dir):
    """Generate plots comparing scalability metrics across algorithms"""
    os.makedirs(results_dir, exist_ok=True)
    
    algorithms = df['algorithm'].unique()
    colors = {'Common Neighbors': 'blue', 'Adamic-Adar': 'green', 
              'Jaccard Coefficient': 'red', 'Preferential Attachment': 'purple',
              'Resource Allocation': 'orange'}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Scalability Analysis: Runtime and Memory Performance', fontsize=16, fontweight='bold')
    
    # 1. Runtime vs Graph Size
    ax = axes[0, 0]
    for algo in algorithms:
        data = df[df['algorithm'] == algo].sort_values('nodes')
        ax.plot(data['nodes'], data['runtime'], marker='o', label=algo, 
                color=colors.get(algo, 'black'), linewidth=2)
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Runtime vs Graph Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Memory vs Graph Size
    ax = axes[0, 1]
    for algo in algorithms:
        data = df[df['algorithm'] == algo].sort_values('nodes')
        ax.plot(data['nodes'], data['memory_mb'], marker='s', label=algo,
                color=colors.get(algo, 'black'), linewidth=2)
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.set_title('Memory Usage vs Graph Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Runtime per Node vs Graph Size
    ax = axes[0, 2]
    for algo in algorithms:
        data = df[df['algorithm'] == algo].sort_values('nodes')
        ax.plot(data['nodes'], data['runtime_per_node'], marker='^', label=algo,
                color=colors.get(algo, 'black'), linewidth=2)
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Runtime per Node (seconds)', fontsize=12)
    ax.set_title('Runtime per Node vs Graph Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Theoretical vs Actual Complexity
    ax = axes[1, 0]
    for algo in algorithms:
        data = df[df['algorithm'] == algo].sort_values('nodes')
        # Normalize both to [0, 1] for comparison
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
    
    # 5. Average Degree vs Graph Size
    ax = axes[1, 1]
    graph_data = df.groupby('graph_id').first().sort_values('nodes')
    ax.plot(graph_data['nodes'], graph_data['avg_degree'], marker='d', 
            color='navy', linewidth=2, label='Average Degree')
    ax.set_xlabel('Number of Nodes', fontsize=12)
    ax.set_ylabel('Average Degree', fontsize=12)
    ax.set_title('Graph Density (Average Degree)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Graph Structure (Nodes vs Edges)
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
    plot_path = os.path.join(os.path.dirname(__file__), 'scalability_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()
    
    # Create separate detailed plots for theoretical vs actual comparison
    plot_theoretical_vs_actual(df, results_dir, algorithms, colors)


def plot_theoretical_vs_actual(df, results_dir, algorithms, colors):
    """Create separate detailed plots comparing theoretical vs actual complexity for each algorithm"""
    
    # Create a 2x3 grid (5 algorithms + 1 combined view)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Theoretical vs Actual Time Complexity Analysis', fontsize=16, fontweight='bold')
    
    # Plot individual algorithms in first 5 subplots
    algo_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    
    for idx, algo in enumerate(algorithms):
        if idx >= 5:  # Safety check
            break
        
        row, col = algo_positions[idx]
        ax = axes[row, col]
        
        data = df[df['algorithm'] == algo].sort_values('nodes')
        if len(data) > 0:
            # Normalize for comparison
            theoretical_norm = data['theoretical_complexity'] / data['theoretical_complexity'].max()
            actual_norm = data['runtime'] / data['runtime'].max()
            
            # Plot both on same axes
            ax.plot(data['nodes'], theoretical_norm, marker='o', linestyle='--', 
                   label='Theoretical', color=colors.get(algo, 'black'), alpha=0.6, linewidth=2)
            ax.plot(data['nodes'], actual_norm, marker='s', linestyle='-', 
                   label='Actual', color=colors.get(algo, 'black'), linewidth=2.5)
            
            # Calculate and display correlation
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
    
    # Combined view in the 6th subplot
    ax = axes[1, 2]
    for algo in algorithms:
        data = df[df['algorithm'] == algo].sort_values('nodes')
        if len(data) > 0:
            # Calculate ratio of actual to theoretical
            theoretical_norm = data['theoretical_complexity'] / data['theoretical_complexity'].max()
            actual_norm = data['runtime'] / data['runtime'].max()
            ratio = actual_norm / theoretical_norm
            
            ax.plot(data['nodes'], ratio, marker='o', label=algo,
                   color=colors.get(algo, 'black'), linewidth=2)
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Perfect Match', alpha=0.7)
    ax.set_xlabel('Number of Nodes', fontsize=11)
    ax.set_ylabel('Actual / Theoretical Ratio', fontsize=11)
    ax.set_title('Algorithm Efficiency (All)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    complexity_plot_path = os.path.join(results_dir, 'theoretical_vs_actual_complexity.png')
    plt.savefig(complexity_plot_path, dpi=300, bbox_inches='tight')
    print(f"Complexity comparison plot saved to: {complexity_plot_path}")
    plt.close()


def plot_performance_metrics(df, results_dir, algorithms, colors):
    """Create plots for performance metrics vs graph size"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Performance Metrics vs Graph Size', fontsize=16, fontweight='bold')
    
    metrics = [
        ('precision', 'Precision', (0, 0)),
        ('recall', 'Recall', (0, 1)),
        ('f1_score', 'F1 Score', (0, 2)),
        ('roc_auc', 'ROC-AUC', (1, 0)),
        ('map', 'MAP', (1, 1))
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
    
    # Last subplot: Summary comparison
    ax = axes[1, 2]
    for algo in algorithms:
        data = df[df['algorithm'] == algo].sort_values('nodes')
        if len(data) > 0:
            # Average of normalized metrics
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
    perf_plot_path = os.path.join(results_dir, 'performance_metrics_vs_size.png')
    plt.savefig(perf_plot_path, dpi=300, bbox_inches='tight')
    print(f"Performance metrics plot saved to: {perf_plot_path}")
    plt.close()


def main():
    print("="*80)
    print("COMPREHENSIVE ANALYSIS - FRIEND RECOMMENDATION ALGORITHMS")
    print("Performance Metrics + Scalability + Complexity Analysis")
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
    print("COMPREHENSIVE RESULTS")
    print(f"{'='*80}\n")
    
    summary_cols = ['algorithm', 'graph_id', 'nodes', 'edges', 'precision', 'recall', 
                    'f1_score', 'roc_auc', 'map', 'runtime', 'memory_mb']
    print(df[summary_cols].to_string(index=False))
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, 'comprehensive_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {csv_path}")
    
    # Generate all plots
    colors = {'Common Neighbors': 'blue', 'Adamic-Adar': 'green', 
              'Jaccard Coefficient': 'red', 'Preferential Attachment': 'purple',
              'Resource Allocation': 'orange'}
    
    algo_names = [algo[0] for algo in algorithms]
    
    plot_scalability_results(df, results_dir)
    plot_theoretical_vs_actual(df, results_dir, algo_names, colors)
    plot_performance_metrics(df, results_dir, algo_names, colors)
    
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS BY ALGORITHM")
    print(f"{'='*80}\n")
    
    agg_stats = df.groupby('algorithm').agg({
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'map': ['mean', 'std'],
        'runtime': ['mean', 'std'],
        'memory_mb': ['mean', 'std']
    }).round(4)
    
    print(agg_stats)
    
    # Complexity analysis
    print(f"\n{'='*80}")
    print("THEORETICAL VS ACTUAL COMPLEXITY ANALYSIS")
    print(f"{'='*80}\n")
    
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo].sort_values('nodes')
        if len(algo_data) > 1:
            # Calculate correlation between theoretical and actual
            correlation = np.corrcoef(algo_data['theoretical_complexity'], algo_data['runtime'])[0, 1]
            print(f"{algo:30s} - Correlation: {correlation:.4f}")
    
    # Best performers
    print(f"\n{'='*80}")
    print("BEST PERFORMERS")
    print(f"{'='*80}")
    
    avg_by_algo = df.groupby('algorithm').mean()
    print(f"\n  Highest Precision:    {avg_by_algo['precision'].idxmax()} ({avg_by_algo['precision'].max():.4f})")
    print(f"  Highest Recall:       {avg_by_algo['recall'].idxmax()} ({avg_by_algo['recall'].max():.4f})")
    print(f"  Highest F1 Score:     {avg_by_algo['f1_score'].idxmax()} ({avg_by_algo['f1_score'].max():.4f})")
    print(f"  Highest ROC-AUC:      {avg_by_algo['roc_auc'].idxmax()} ({avg_by_algo['roc_auc'].max():.4f})")
    print(f"  Highest MAP:          {avg_by_algo['map'].idxmax()} ({avg_by_algo['map'].max():.4f})")
    print(f"  Fastest Runtime:      {avg_by_algo['runtime'].idxmin()} ({avg_by_algo['runtime'].min():.4f}s)")
    print(f"  Lowest Memory:        {avg_by_algo['memory_mb'].idxmin()} ({avg_by_algo['memory_mb'].min():.2f}MB)")
    
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}")
    
    return df


if __name__ == "__main__":
    results_df = main()
