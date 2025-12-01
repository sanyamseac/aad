"""
Comprehensive Node2Vec Analysis
- Hyperparameter exploration: p, q, num_walks, walk_length
- Performance comparison with heuristic algorithms
- Scalability analysis across different graph sizes
"""

import os
import sys
import time
import gc
import tracemalloc
import multiprocessing
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import questionary
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import networkx as nx
from itertools import product
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

from nv import train_node2vec_model, recommend_friends as nv_recommend, get_edge_score

# Import functions from analysis.py for comparison
from analysis import (
    train_test_split, 
    calculate_performance_metrics,
    load_and_prepare_graph
)


def evaluate_node2vec_config(graph_data, p, q, num_walks, walk_length, top_k=10, sample_size=50):
    """Evaluate Node2Vec with specific hyperparameters.
    
    Trains a Node2Vec model and evaluates its performance on link prediction.
    
    Args:
        graph_data (tuple): (graph_id, nx_G, train_G, test_edges_pos, test_edges_neg, sample_nodes).
        p (float): Return parameter for random walks.
        q (float): In-out parameter for random walks.
        num_walks (int): Number of walks per node.
        walk_length (int): Length of each walk.
        top_k (int, optional): Number of recommendations per node. Defaults to 10.
        sample_size (int, optional): Number of nodes to evaluate. Defaults to 50.
        
    Returns:
        dict: Performance metrics including precision, recall, F1, runtime, memory usage.
        None: If evaluation fails.
    """
    graph_id, nx_G, train_G, test_edges_pos, test_edges_neg, sample_nodes = graph_data
    
    gc.collect()
    tracemalloc.start()
    
    train_start = time.time()
    try:
        model = train_node2vec_model(train_G, p=p, q=q, num_walks=num_walks, walk_length=walk_length)
        train_time = time.time() - train_start
    except Exception as e:
        tracemalloc.stop()
        return None
    
    inference_start = time.time()
    all_predictions, ranked_predictions = [], []
    
    for node in sample_nodes:
        preds = nv_recommend(model, node, train_G, top_k)
        all_predictions.extend([(node, pred_node) for pred_node, _ in preds])
        ranked_predictions.append(preds)
    
    inference_time = time.time() - inference_start
    total_runtime = train_time + inference_time
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_used = peak / 1024 / 1024
    
    def score_func_wrapper(G, u, v):
        return get_edge_score(model, (u, v))
    
    perf_metrics = calculate_performance_metrics(
        all_predictions, test_edges_pos, test_edges_neg,
        train_G, score_func_wrapper, 
        ranked_predictions, sample_nodes
    )
    
    return {
        'graph_id': graph_id,
        'nodes': nx_G.number_of_nodes(),
        'edges': nx_G.number_of_edges(),
        'p': p,
        'q': q,
        'num_walks': num_walks,
        'walk_length': walk_length,
        'train_time': train_time,
        'inference_time': inference_time,
        'total_runtime': total_runtime,
        'memory_mb': memory_used,
        'sample_size': len(sample_nodes),
        **perf_metrics
    }


def hyperparameter_exploration(graph_id=1, top_k=10, sample_size=50,
                              p_values=None, q_values=None, 
                              num_walks_values=None, walk_length_values=None,
                              execution_mode='parallel', max_workers=None):
    """Systematic hyperparameter exploration for Node2Vec.
    
    Tests all combinations of p, q, num_walks, and walk_length parameters
    to find optimal configurations.
    
    Args:
        graph_id (int, optional): Graph to evaluate on. Defaults to 1.
        top_k (int, optional): Recommendations per node. Defaults to 10.
        sample_size (int, optional): Nodes to evaluate. Defaults to 50.
        p_values (list, optional): Return parameter values. Defaults to [0.5, 0.7, 1.0, 1.5, 2.0].
        q_values (list, optional): In-out parameter values. Defaults to [0.5, 0.7, 1.0, 1.5, 2.0].
        num_walks_values (list, optional): Walk count values. Defaults to [5, 10, 20].
        walk_length_values (list, optional): Walk length values. Defaults to [40, 80, 120].
        execution_mode (str, optional): 'parallel' or 'sequential'. Defaults to 'parallel'.
        max_workers (int, optional): Parallel workers. Defaults to CPU count.
        
    Returns:
        pd.DataFrame: Results for all parameter combinations.
        None: If graph loading fails.
    """
    if p_values is None:
        p_values = [0.5, 0.7, 1.0, 1.5, 2.0]
    if q_values is None:
        q_values = [0.5, 0.7, 1.0, 1.5, 2.0]
    if num_walks_values is None:
        num_walks_values = [5, 10, 20]
    if walk_length_values is None:
        walk_length_values = [40, 80, 120]
    
    graph_data = load_and_prepare_graph(graph_id, sample_size)
    if not graph_data:
        return None
    
    print(f"\nGraph {graph_id}: {graph_data[1].number_of_nodes()} nodes, {graph_data[1].number_of_edges()} edges")
    print(f"Parameter grid: p={len(p_values)}, q={len(q_values)}, num_walks={len(num_walks_values)}, walk_length={len(walk_length_values)}")
    
    configs = list(product(p_values, q_values, num_walks_values, walk_length_values))
    print(f"Total configurations: {len(configs)}")
    print(f"Execution mode: {execution_mode.upper()}")
    
    results = []
    
    if execution_mode == 'sequential':
        print("\nRunning sequential evaluation...\n")
        pbar = tqdm(configs, desc="Hyperparameter search", ncols=80,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for p, q, nw, wl in pbar:
            try:
                result = evaluate_node2vec_config(graph_data, p, q, nw, wl, top_k, sample_size)
                if result:
                    results.append(result)
                pbar.set_postfix_str(f"p={p:.1f}, q={q:.1f}")
            except Exception as e:
                pbar.write(f"  Warning: Error with config p={p}, q={q}: {str(e)[:50]}")
    else:
        # Parallel execution with ProcessPoolExecutor for true parallelism
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        else:
            max_workers = min(max_workers, len(configs))
        
        print(f"Running parallel evaluation with {max_workers} workers (ProcessPoolExecutor)...\n")
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(evaluate_node2vec_config, graph_data, p, q, nw, wl, top_k, sample_size): (p, q, nw, wl)
                    for p, q, nw, wl in configs
                }
                
                pbar = tqdm(total=len(futures), desc="Hyperparameter search", ncols=80,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
                
                for future in as_completed(futures):
                    config = futures[future]
                    try:
                        result = future.result(timeout=300)
                        if result:
                            results.append(result)
                        pbar.set_postfix_str(f"p={config[0]:.1f}, q={config[1]:.1f}")
                    except TimeoutError:
                        pbar.write(f"  Warning: Timeout for config p={config[0]}, q={config[1]}")
                    except Exception as e:
                        pbar.write(f"  Warning: Error with config p={config[0]}, q={config[1]}: {str(e)[:50]}")
                    pbar.update(1)
                
                pbar.close()
        except Exception as e:
            print(f"\nProcessPoolExecutor failed: {e}")
            print("Falling back to ThreadPoolExecutor...\n")
            
            # Fallback to ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(evaluate_node2vec_config, graph_data, p, q, nw, wl, top_k, sample_size): (p, q, nw, wl)
                    for p, q, nw, wl in configs
                }
                
                pbar = tqdm(total=len(futures), desc="Hyperparameter search", ncols=80,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
                
                for future in as_completed(futures):
                    config = futures[future]
                    try:
                        result = future.result(timeout=300)
                        if result:
                            results.append(result)
                        pbar.set_postfix_str(f"p={config[0]:.1f}, q={config[1]:.1f}")
                    except TimeoutError:
                        pbar.write(f"  Warning: Timeout for config p={config[0]}, q={config[1]}")
                    except Exception as e:
                        pbar.write(f"  Warning: Error with config p={config[0]}, q={config[1]}: {str(e)[:50]}")
                    pbar.update(1)
                
                pbar.close()
    
    return pd.DataFrame(results)


def scalability_analysis(graph_ids=None, p=0.7, q=0.7, num_walks=10, walk_length=80):
    """Analyze Node2Vec scalability across graph sizes.
    
    Evaluates how runtime and memory usage scale with graph size using
    fixed hyperparameters.
    
    Args:
        graph_ids (list, optional): Graph IDs to test. Defaults to range(1, 11).
        p (float, optional): Return parameter. Defaults to 0.7.
        q (float, optional): In-out parameter. Defaults to 0.7.
        num_walks (int, optional): Walks per node. Defaults to 10.
        walk_length (int, optional): Steps per walk. Defaults to 80.
        
    Returns:
        pd.DataFrame: Scalability metrics for each graph.
        None: If no graphs could be evaluated.
    """
    if graph_ids is None:
        graph_ids = list(range(1, 11))
    
    print(f"\nFixed hyperparameters: p={p}, q={q}, num_walks={num_walks}, walk_length={walk_length}")
    
    results = []
    tasks = []
    
    print("Loading graphs...")
    for graph_id in tqdm(graph_ids, desc="Loading", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        try:
            graph_data = load_and_prepare_graph(graph_id, sample_size=50)
            if graph_data:
                tasks.append((graph_data, p, q, num_walks, walk_length, 10, 50))
        except Exception as e:
            print(f"  Warning: Error loading graph {graph_id}: {str(e)[:50]}")
    
    max_workers = min(multiprocessing.cpu_count(), len(tasks))
    print(f"\nRunning evaluations with {max_workers} parallel workers (ProcessPoolExecutor)...\n")
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate_node2vec_config, *task): task[0][0] for task in tasks}
            
            pbar = tqdm(total=len(futures), desc="Scalability analysis", ncols=80,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            for future in as_completed(futures):
                graph_id = futures[future]
                try:
                    result = future.result(timeout=600)
                    if result:
                        results.append(result)
                        pbar.set_postfix_str(f"Graph {graph_id}: {result['nodes']} nodes")
                except TimeoutError:
                    pbar.write(f"  Warning: Timeout for graph {graph_id}")
                except Exception as e:
                    pbar.write(f"  Warning: Error for graph {graph_id}: {str(e)[:50]}")
                pbar.update(1)
            
            pbar.close()
    except Exception as e:
        print(f"\nProcessPoolExecutor failed: {e}")
        print("Falling back to ThreadPoolExecutor...\n")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate_node2vec_config, *task): task[0][0] for task in tasks}
            
            pbar = tqdm(total=len(futures), desc="Scalability analysis", ncols=80,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            for future in as_completed(futures):
                graph_id = futures[future]
                try:
                    result = future.result(timeout=600)
                    if result:
                        results.append(result)
                        pbar.set_postfix_str(f"Graph {graph_id}: {result['nodes']} nodes")
                except TimeoutError:
                    pbar.write(f"  Warning: Timeout for graph {graph_id}")
                except Exception as e:
                    pbar.write(f"  Warning: Error for graph {graph_id}: {str(e)[:50]}")
                pbar.update(1)
            
            pbar.close()
    
    if len(results) == 0:
        return None
    
    return pd.DataFrame(results)


def compare_with_heuristics(graph_id=1, nv_configs=None, top_k=10, sample_size=50):
    """Compare Node2Vec variants against heuristic baselines.
    
    Evaluates multiple Node2Vec configurations alongside Common Neighbors,
    Adamic-Adar, Jaccard, Preferential Attachment, and Resource Allocation.
    
    Args:
        graph_id (int, optional): Graph to evaluate on. Defaults to 1.
        nv_configs (list, optional): List of (p, q, num_walks, walk_length, name) tuples.
            Defaults to Balanced, DeepWalk, BFS-like, and DFS-like configurations.
        top_k (int, optional): Recommendations per node. Defaults to 10.
        sample_size (int, optional): Nodes to evaluate. Defaults to 50.
        
    Returns:
        pd.DataFrame: Comparative performance metrics.
        None: If graph loading fails.
    """
    from analysis import (
        cm_recommend, compute_common_neighbors_score,
        aa_recommend, compute_adamic_adar_score,
        jc_recommend, compute_jaccard_coefficient,
        pa_recommend, compute_preferential_attachment_score,
        ra_recommend, compute_resource_allocation_score,
        evaluate_algorithm_on_graph
    )
    
    graph_data = load_and_prepare_graph(graph_id, sample_size)
    if not graph_data:
        return None
    
    print(f"\nGraph {graph_id}: {graph_data[1].number_of_nodes()} nodes, {graph_data[1].number_of_edges()} edges")
    
    if nv_configs is None:
        nv_configs = [
            (0.7, 0.7, 10, 80, "Balanced (p=0.7, q=0.7)"),
            (1.0, 1.0, 10, 80, "DeepWalk (p=1.0, q=1.0)"),
            (0.5, 2.0, 10, 80, "BFS-like (p=0.5, q=2.0)"),
            (2.0, 0.5, 10, 80, "DFS-like (p=2.0, q=0.5)")
        ]
    
    print("\nEvaluating Node2Vec configurations...")
    nv_results = []
    for p, q, nw, wl, desc in tqdm(nv_configs, desc="Node2Vec", ncols=80, 
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        result = evaluate_node2vec_config(graph_data, p, q, nw, wl, top_k=top_k, sample_size=sample_size)
        if result:
            result['algorithm'] = f"Node2Vec: {desc}"
            nv_results.append(result)
    
    print("\nEvaluating heuristic algorithms...")
    heuristic_algos = [
        ("Common Neighbors", cm_recommend, compute_common_neighbors_score),
        ("Adamic-Adar", aa_recommend, compute_adamic_adar_score),
        ("Jaccard Coefficient", jc_recommend, compute_jaccard_coefficient),
        ("Preferential Attachment", pa_recommend, compute_preferential_attachment_score),
        ("Resource Allocation", ra_recommend, compute_resource_allocation_score)
    ]
    
    heuristic_results = []
    for algo_name, recommend_func, score_func in tqdm(heuristic_algos, desc="Heuristics", ncols=80,
                                                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        result = evaluate_algorithm_on_graph(graph_data, algo_name, recommend_func, score_func, top_k=top_k, sample_size=sample_size)
        if result:
            result['algorithm'] = algo_name
            result['train_time'] = 0.0
            result['inference_time'] = result['runtime']
            result['total_runtime'] = result['runtime']
            heuristic_results.append(result)
    
    all_results = nv_results + heuristic_results
    return pd.DataFrame(all_results)


def plot_hyperparameter_analysis(df, results_dir):
    """Plot hyperparameter exploration results"""
    
    if df is None or len(df) == 0:
        print("  Warning: No data to plot for hyperparameter analysis")
        return
    
    # 1. Effect of p and q
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Node2Vec Hyperparameter Analysis (p, q)', fontsize=16, fontweight='bold')
    
    # Fix num_walks and walk_length to specific values for p,q analysis
    target_nw = 10  # Most balanced
    target_wl = 80  # Most balanced
    pq_data = df[(df['num_walks'] == target_nw) & (df['walk_length'] == target_wl)]
    
    if len(pq_data) > 0:
        # Create pivot tables for heatmaps
        metrics = [
            ('f1_score', 'F1 Score', axes[0, 0]),
            ('roc_auc', 'ROC-AUC', axes[0, 1]),
            ('map', 'MAP', axes[0, 2]),
            ('total_runtime', 'Runtime (s)', axes[1, 0]),
            ('train_time', 'Training Time (s)', axes[1, 1]),
            ('inference_time', 'Inference Time (s)', axes[1, 2])
        ]
        
        for metric_key, metric_name, ax in metrics:
            pivot = pq_data.pivot_table(values=metric_key, index='q', columns='p', aggfunc='mean')
            im = ax.imshow(pivot.values, cmap='viridis', aspect='auto', interpolation='nearest')
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_xticklabels([f'{x:.1f}' for x in pivot.columns], fontsize=10)
            ax.set_yticklabels([f'{x:.1f}' for x in pivot.index], fontsize=10)
            ax.set_xlabel('p (return parameter)', fontsize=11)
            ax.set_ylabel('q (in-out parameter)', fontsize=11)
            ax.set_title(f'{metric_name} vs p,q', fontsize=12, fontweight='bold')
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=9)
            
            # Add values in cells
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    value = pivot.values[i, j]
                    if not np.isnan(value):
                        text_color = 'white' if value < pivot.values.mean() else 'black'
                        ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                               color=text_color, fontsize=8, fontweight='bold')
        
        print(f"  Generated p,q heatmaps with {len(pq_data)} data points")
    else:
        print(f"  Warning: No data for num_walks={target_nw}, walk_length={target_wl}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'nv_hyperparameter_pq.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Effect of num_walks and walk_length
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Node2Vec Walk Parameters Analysis (p=1.0, q=1.0)', fontsize=16, fontweight='bold')
    
    # Fix p and q to 1.0 (DeepWalk-like) for walk analysis
    walk_data = df[(df['p'] == 1.0) & (df['q'] == 1.0)]
    
    if len(walk_data) > 0:
        metrics = [
            ('f1_score', 'F1 Score', axes[0, 0]),
            ('roc_auc', 'ROC-AUC', axes[0, 1]),
            ('map', 'MAP', axes[0, 2]),
            ('total_runtime', 'Runtime (s)', axes[1, 0]),
            ('train_time', 'Training Time (s)', axes[1, 1]),
            ('memory_mb', 'Memory (MB)', axes[1, 2])
        ]
        
        for metric_key, metric_name, ax in metrics:
            for wl in sorted(walk_data['walk_length'].unique()):
                data = walk_data[walk_data['walk_length'] == wl]
                grouped = data.groupby('num_walks')[metric_key].mean()
                ax.plot(grouped.index, grouped.values, marker='o', label=f'walk_len={wl}', 
                       linewidth=2, markersize=8)
            
            ax.set_xlabel('Number of Walks', fontsize=11)
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(f'{metric_name} vs Walk Parameters', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        print(f"  Generated walk parameter plots with {len(walk_data)} data points")
    else:
        print("  Warning: No data for p=1.0, q=1.0")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'nv_hyperparameter_walks.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_scalability_analysis(df, results_dir):
    """Plot Node2Vec scalability results"""
    
    if df is None or len(df) == 0:
        print("  Warning: No data to plot for scalability analysis")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Node2Vec Scalability Analysis', fontsize=16, fontweight='bold')
    
    df_sorted = df.sort_values('nodes')
    
    # 1. Total Runtime
    ax = axes[0, 0]
    ax.plot(df_sorted['nodes'], df_sorted['total_runtime'], marker='o', color='purple', 
            linewidth=2, markersize=8)
    ax.set_xlabel('Number of Nodes', fontsize=11)
    ax.set_ylabel('Total Runtime (s)', fontsize=11)
    ax.set_title('Total Runtime vs Graph Size', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. Training vs Inference Time
    ax = axes[0, 1]
    ax.plot(df_sorted['nodes'], df_sorted['train_time'], marker='s', label='Training', 
            linewidth=2, markersize=8)
    ax.plot(df_sorted['nodes'], df_sorted['inference_time'], marker='^', label='Inference', 
            linewidth=2, markersize=8)
    ax.set_xlabel('Number of Nodes', fontsize=11)
    ax.set_ylabel('Time (s)', fontsize=11)
    ax.set_title('Training vs Inference Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Memory Usage
    ax = axes[0, 2]
    ax.plot(df_sorted['nodes'], df_sorted['memory_mb'], marker='d', color='red', 
            linewidth=2, markersize=8)
    ax.set_xlabel('Number of Nodes', fontsize=11)
    ax.set_ylabel('Memory (MB)', fontsize=11)
    ax.set_title('Memory Usage vs Graph Size', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. F1 Score
    ax = axes[1, 0]
    ax.plot(df_sorted['nodes'], df_sorted['f1_score'], marker='o', color='green', 
            linewidth=2, markersize=8)
    ax.set_xlabel('Number of Nodes', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('F1 Score vs Graph Size', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 5. ROC-AUC
    ax = axes[1, 1]
    ax.plot(df_sorted['nodes'], df_sorted['roc_auc'], marker='s', color='blue', 
            linewidth=2, markersize=8)
    ax.set_xlabel('Number of Nodes', fontsize=11)
    ax.set_ylabel('ROC-AUC', fontsize=11)
    ax.set_title('ROC-AUC vs Graph Size', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 6. MAP
    ax = axes[1, 2]
    ax.plot(df_sorted['nodes'], df_sorted['map'], marker='^', color='orange', 
            linewidth=2, markersize=8)
    ax.set_xlabel('Number of Nodes', fontsize=11)
    ax.set_ylabel('MAP', fontsize=11)
    ax.set_title('MAP vs Graph Size', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    print(f"  Generated scalability plots with {len(df_sorted)} data points")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'nv_scalability.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_with_heuristics(df, results_dir):
    """Plot comparison between Node2Vec and heuristic algorithms"""
    
    if df is None or len(df) == 0:
        print("  Warning: No data to plot for comparison analysis")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Node2Vec vs Heuristic Algorithms Comparison', fontsize=16, fontweight='bold')
    
    algorithms = df['algorithm'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    
    metrics = [
        ('precision', 'Precision', axes[0, 0]),
        ('recall', 'Recall', axes[0, 1]),
        ('f1_score', 'F1 Score', axes[0, 2]),
        ('roc_auc', 'ROC-AUC', axes[1, 0]),
        ('map', 'MAP', axes[1, 1]),
        ('total_runtime', 'Runtime (s)', axes[1, 2])
    ]
    
    for metric_key, metric_name, ax in metrics:
        values = [df[df['algorithm'] == algo][metric_key].iloc[0] for algo in algorithms]
        bars = ax.bar(range(len(algorithms)), values, color=colors)
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight Node2Vec algorithms
        for i, algo in enumerate(algorithms):
            if 'Node2Vec' in algo:
                bars[i].set_edgecolor('red')
                bars[i].set_linewidth(2)
    
    print(f"  Generated comparison plots with {len(algorithms)} algorithms")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'nv_vs_heuristics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("\n" + "="*80)
    print("  COMPREHENSIVE NODE2VEC ANALYSIS")
    print("  Performance • Scalability • Hyperparameter Exploration")
    print("="*80)
    
    # Ask user for execution mode
    execution_mode = questionary.select(
        "Select execution mode for hyperparameter exploration:",
        choices=[
            questionary.Choice("Parallel (faster, uses all CPU cores)", value="parallel"),
            questionary.Choice("Sequential (slower, more memory-accurate)", value="sequential")
        ]
    ).ask()
    
    max_workers = None
    if execution_mode == 'parallel':
        max_workers = multiprocessing.cpu_count()
        print(f"\nUsing parallel execution with {max_workers} workers")
    else:
        print("\nUsing sequential execution")
    
    # ===================== CONFIGURATION CONSTANTS =====================
    # Hyperparameter ranges for exploration
    P_VALUES = [0.2, 0.5, 0.7, 1.0, 1.5, 2.0]           # Return parameter
    Q_VALUES = [0.2, 0.5, 0.7, 1.0, 1.5, 2.0]           # In-out parameter
    NUM_WALKS_VALUES = [5, 10, 20, 40]                  # Number of walks per node
    WALK_LENGTH_VALUES = [40, 80, 120]                  # Length of each walk
    
    # Scalability test configuration
    SCALABILITY_GRAPH_IDS = list(range(1, 11))       # Test on graphs 1-10
    SCALABILITY_P = 0.7                              # Fixed p for scalability
    SCALABILITY_Q = 0.7                              # Fixed q for scalability
    SCALABILITY_NUM_WALKS = 10                       # Fixed num_walks
    SCALABILITY_WALK_LENGTH = 80                     # Fixed walk_length
    
    # Comparison configuration
    COMPARISON_GRAPH_ID = 10                          # Which graph to compare on
    COMPARISON_CONFIGS = [
        (0.7, 0.7, 10, 80, "Balanced (p=0.7, q=0.7)"),
        (1.0, 1.0, 10, 80, "DeepWalk (p=1.0, q=1.0)"),
        (0.5, 2.0, 10, 80, "BFS-like (p=0.5, q=2.0)"),
        (2.0, 0.5, 10, 80, "DFS-like (p=2.0, q=0.5)")
    ]
    
    # General settings
    HYPERPARAM_GRAPH_ID = 1                          # Graph for hyperparameter exploration
    TOP_K = 10                                       # Number of recommendations
    SAMPLE_SIZE = 50                                 # Number of nodes to sample
    # ==================================================================
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Hyperparameter Exploration
    print("\n" + "="*80)
    print("PHASE 1: Hyperparameter Exploration")
    print("-" * 80)
    print(f"Configuration:")
    print(f"  p values: {P_VALUES}")
    print(f"  q values: {Q_VALUES}")
    print(f"  num_walks values: {NUM_WALKS_VALUES}")
    print(f"  walk_length values: {WALK_LENGTH_VALUES}")
    print(f"  Total combinations: {len(P_VALUES) * len(Q_VALUES) * len(NUM_WALKS_VALUES) * len(WALK_LENGTH_VALUES)}")
    
    # Pass constants to hyperparameter_exploration
    hyperparam_df = hyperparameter_exploration(
        graph_id=HYPERPARAM_GRAPH_ID, 
        top_k=TOP_K, 
        sample_size=SAMPLE_SIZE,
        p_values=P_VALUES,
        q_values=Q_VALUES,
        num_walks_values=NUM_WALKS_VALUES,
        walk_length_values=WALK_LENGTH_VALUES,
        execution_mode=execution_mode,
        max_workers=max_workers
    )
    
    if hyperparam_df is not None and len(hyperparam_df) > 0:
        hyperparam_csv = os.path.join(results_dir, 'nv_hyperparameter_exploration.csv')
        hyperparam_df.to_csv(hyperparam_csv, index=False)
        print(f"\nResults saved: {os.path.basename(hyperparam_csv)}")
        
        print("\nTop 5 configurations by F1 Score:")
        top_configs = hyperparam_df.nlargest(5, 'f1_score')[['p', 'q', 'num_walks', 'walk_length', 'f1_score', 'roc_auc', 'map', 'total_runtime']]
        print(top_configs.to_string(index=False))
        
        print("\nGenerating hyperparameter plots...")
        plot_hyperparameter_analysis(hyperparam_df, results_dir)
        print("  ✓ Hyperparameter plots saved")
    else:
        print("\n  Warning: No hyperparameter results to save")
    
    # 2. Scalability Analysis
    print("\n" + "="*80)
    print("PHASE 2: Scalability Analysis")
    print("-" * 80)
    print(f"Configuration:")
    print(f"  Graph IDs: {SCALABILITY_GRAPH_IDS}")
    print(f"  Fixed p={SCALABILITY_P}, q={SCALABILITY_Q}")
    print(f"  Fixed num_walks={SCALABILITY_NUM_WALKS}, walk_length={SCALABILITY_WALK_LENGTH}")
    
    scalability_df = scalability_analysis(
        graph_ids=SCALABILITY_GRAPH_IDS,
        p=SCALABILITY_P,
        q=SCALABILITY_Q,
        num_walks=SCALABILITY_NUM_WALKS,
        walk_length=SCALABILITY_WALK_LENGTH
    )
    
    if scalability_df is not None and len(scalability_df) > 0:
        scalability_csv = os.path.join(results_dir, 'nv_scalability.csv')
        scalability_df.to_csv(scalability_csv, index=False)
        print(f"\nResults saved: {os.path.basename(scalability_csv)}")
        
        print("\nGenerating scalability plots...")
        plot_scalability_analysis(scalability_df, results_dir)
        print("  ✓ Scalability plots saved")
    else:
        print("\n  Warning: No scalability results to save")
    
    # 3. Comparison with Heuristics
    print("\n" + "="*80)
    print("PHASE 3: Comparison with Heuristic Algorithms")
    print("-" * 80)
    print(f"Configuration:")
    print(f"  Comparison graph ID: {COMPARISON_GRAPH_ID}")
    print(f"  Node2Vec configurations: {len(COMPARISON_CONFIGS)}")
    for p, q, nw, wl, desc in COMPARISON_CONFIGS:
        print(f"    {desc}: p={p}, q={q}, num_walks={nw}, walk_length={wl}")
    
    comparison_df = compare_with_heuristics(
        graph_id=COMPARISON_GRAPH_ID,
        nv_configs=COMPARISON_CONFIGS,
        top_k=TOP_K,
        sample_size=SAMPLE_SIZE
    )
    
    if comparison_df is not None and len(comparison_df) > 0:
        comparison_csv = os.path.join(results_dir, 'nv_vs_heuristics.csv')
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"\nResults saved: {os.path.basename(comparison_csv)}")
        
        print("\nPerformance Comparison:")
        print("=" * 80)
        display_cols = ['algorithm', 'precision', 'recall', 'f1_score', 'roc_auc', 'map', 'total_runtime', 'memory_mb']
        print(comparison_df[display_cols].to_string(index=False))
        
        print("\nGenerating comparison plots...")
        plot_comparison_with_heuristics(comparison_df, results_dir)
        print("  ✓ Comparison plots saved")
    else:
        print("\n  Warning: No comparison results to save")
    
    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)
    print(f"Results directory: {results_dir}\n")
    
    return {
        # 'hyperparameters': hyperparam_df,
        'scalability': scalability_df,
        'comparison': comparison_df
    }


if __name__ == "__main__":
    import sys
    try:
        results = main()
        # Force cleanup
        import matplotlib.pyplot as plt
        plt.close('all')
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
