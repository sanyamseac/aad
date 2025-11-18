"""
Comprehensive Performance Analysis for Friend Recommendation Algorithms

Evaluates all algorithms (CM, AA, JC, PA, RA, NV) with:
- Performance Metrics: Precision, Recall, F1, ROC-AUC, MAP
- Operational Metrics: Runtime, Scalability, Memory Usage
"""

import random
import networkx as nx
import sys
import os
import numpy as np
import time
import tracemalloc
import psutil
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, List, Tuple, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

from cm import recommend_friends as cm_recommend, compute_common_neighbors_score
from aa import recommend_friends as aa_recommend, compute_adamic_adar_score
from jc import recommend_friends as jc_recommend, compute_jaccard_coefficient
from pa import recommend_friends as pa_recommend, compute_preferential_attachment_score
from ra import recommend_friends as ra_recommend, compute_resource_allocation_score
from nv import train_node2vec_model, recommend_friends as nv_recommend, get_edge_score


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


def calculate_precision_recall_f1(predictions, test_edges_pos):
    test_set = set(tuple(sorted(edge)) for edge in test_edges_pos)
    pred_set = set(tuple(sorted(edge)) for edge in predictions)
    
    true_positives = len(pred_set.intersection(test_set))
    
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = true_positives / len(test_set) if len(test_set) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'total_predictions': len(pred_set),
        'total_actual': len(test_set)
    }


def calculate_roc_auc(scores_pos, scores_neg):
    y_true = [1] * len(scores_pos) + [0] * len(scores_neg)
    y_scores = scores_pos + scores_neg
    
    if len(set(y_true)) < 2:
        return 0.0
    
    try:
        return roc_auc_score(y_true, y_scores)
    except:
        return 0.0


def calculate_map(ranked_predictions, test_edges_pos, nodes):
    test_dict = {}
    for u, v in test_edges_pos:
        if u not in test_dict:
            test_dict[u] = set()
        if v not in test_dict:
            test_dict[v] = set()
        test_dict[u].add(v)
        test_dict[v].add(u)
    
    average_precisions = []
    
    for i, node in enumerate(nodes):
        if node not in test_dict or len(ranked_predictions[i]) == 0:
            continue
            
        relevant_items = test_dict[node]
        predictions = ranked_predictions[i]
        
        hits = 0
        precision_sum = 0.0
        
        for k, (pred_node, _) in enumerate(predictions, 1):
            if pred_node in relevant_items:
                hits += 1
                precision_sum += hits / k
        
        if hits > 0:
            average_precisions.append(precision_sum / len(relevant_items))
    
    return np.mean(average_precisions) if average_precisions else 0.0


def evaluate_algorithm(algo_name, train_G, test_edges_pos, test_edges_neg, 
                       recommend_func, score_func, top_k=10, sample_size=100, model=None):
    all_nodes = list(train_G.nodes())
    sample_nodes = random.sample(all_nodes, min(sample_size, len(all_nodes)))
    
    process = psutil.Process()
    process.memory_info()
    
    start_memory = process.memory_info().rss / 1024 / 1024
    start_time = time.time()
    
    all_predictions = []
    ranked_predictions = []
    
    for node in sample_nodes:
        if model is not None:
            preds = recommend_func(model, node, train_G, top_k)
        else:
            preds = recommend_func(train_G, node, top_k)
        all_predictions.extend([(node, pred_node) for pred_node, _ in preds])
        ranked_predictions.append(preds)
    
    runtime = time.time() - start_time
    end_memory = process.memory_info().rss / 1024 / 1024
    memory_used = max(0.01, end_memory - start_memory)
    
    prf_metrics = calculate_precision_recall_f1(all_predictions, test_edges_pos)
    
    roc_auc = 0.0
    if score_func:
        try:
            test_sample = min(100, len(test_edges_pos))
            if model is not None:
                scores_pos = [score_func(model, (u, v)) for u, v in test_edges_pos[:test_sample]]
                scores_neg = [score_func(model, (u, v)) for u, v in test_edges_neg[:test_sample]]
            else:
                scores_pos = [score_func(train_G, u, v) for u, v in test_edges_pos[:test_sample]]
                scores_neg = [score_func(train_G, u, v) for u, v in test_edges_neg[:test_sample]]
            roc_auc = calculate_roc_auc(scores_pos, scores_neg)
        except:
            roc_auc = 0.0
    
    map_score = calculate_map(ranked_predictions, test_edges_pos, sample_nodes)
    
    sklearn_map = 0.0
    if score_func:
        try:
            test_sample = min(100, len(test_edges_pos))
            y_true = [1] * test_sample + [0] * test_sample
            y_scores = scores_pos + scores_neg
            sklearn_map = average_precision_score(y_true, y_scores)
        except:
            sklearn_map = map_score
    
    results = {
        'algorithm': algo_name,
        'precision': prf_metrics['precision'],
        'recall': prf_metrics['recall'],
        'f1_score': prf_metrics['f1_score'],
        'roc_auc': roc_auc,
        'map': map_score,
        'sklearn_map': sklearn_map,
        'runtime': runtime,
        'memory_mb': memory_used,
        'sample_size': len(sample_nodes),
        'top_k': top_k,
        'true_positives': prf_metrics['true_positives'],
        'total_predictions': prf_metrics['total_predictions'],
        'total_actual': prf_metrics['total_actual']
    }
    
    return results


def evaluate_scalability(algo_name, G, recommend_func, sample_sizes=[50, 100, 200, 500]):
    print(f"\n{'='*80}")
    print(f"Scalability Analysis: {algo_name}")
    print(f"{'='*80}")
    
    scalability_results = []
    process = psutil.Process()
    
    for size in sample_sizes:
        if size > G.number_of_nodes():
            continue
            
        sample_nodes = random.sample(list(G.nodes()), size)
        
        process.memory_info()
        start_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        for node in sample_nodes:
            _ = recommend_func(G, node, 10)
        
        runtime = time.time() - start_time
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_used = max(0.01, end_memory - start_memory)
        
        scalability_results.append({
            'algorithm': algo_name,
            'sample_size': size,
            'runtime': runtime,
            'memory_mb': memory_used,
            'time_per_node': runtime / size
        })
        
        print(f"  Size {size:4d}: Runtime={runtime:6.3f}s, Memory={memory_used:6.2f}MB, Time/Node={runtime/size:6.4f}s")
    
    return scalability_results


def main(nx_G, test_ratio=0.2, top_k=10, sample_size=100):
    print("="*80)
    print("COMPREHENSIVE ALGORITHM ANALYSIS (Heuristic + Embedding-based)")
    print("="*80)
    
    print(f"\nGraph Statistics:")
    print(f"  Total Nodes: {nx_G.number_of_nodes()}")
    print(f"  Total Edges: {nx_G.number_of_edges()}")
    print(f"  Average Degree: {2 * nx_G.number_of_edges() / nx_G.number_of_nodes():.2f}")
    
    train_G, test_edges_pos, test_edges_neg = train_test_split(nx_G, test_ratio=test_ratio)
    
    print(f"\nDataset Split:")
    print(f"  Train Graph Edges:         {train_G.number_of_edges()}")
    print(f"  Test Edges (Positive):     {len(test_edges_pos)}")
    print(f"  Test Edges (Negative):     {len(test_edges_neg)}")
    print(f"  Test Ratio:                {test_ratio:.2f}")
    
    # Train Node2Vec model first
    print(f"\n{'='*80}")
    print("Training Node2Vec Model")
    print(f"{'='*80}")
    nv_model = train_node2vec_model(train_G, p=0.7, q=0.7, num_walks=10, walk_length=80)
    print("Node2Vec model trained successfully")
    
    # Define evaluation tasks
    eval_tasks = [
        ("Common Neighbors", train_G, test_edges_pos, test_edges_neg, 
         cm_recommend, compute_common_neighbors_score, top_k, sample_size, None),
        ("Adamic-Adar", train_G, test_edges_pos, test_edges_neg, 
         aa_recommend, compute_adamic_adar_score, top_k, sample_size, None),
        ("Jaccard Coefficient", train_G, test_edges_pos, test_edges_neg, 
         jc_recommend, compute_jaccard_coefficient, top_k, sample_size, None),
        ("Preferential Attachment", train_G, test_edges_pos, test_edges_neg, 
         pa_recommend, compute_preferential_attachment_score, top_k, sample_size, None),
        ("Resource Allocation", train_G, test_edges_pos, test_edges_neg, 
         ra_recommend, compute_resource_allocation_score, top_k, sample_size, None),
        ("Node2Vec", train_G, test_edges_pos, test_edges_neg, 
         nv_recommend, get_edge_score, top_k, sample_size, nv_model),
    ]
    
    # Run evaluations in parallel
    print(f"\n{'='*80}")
    print("RUNNING EVALUATIONS IN PARALLEL")
    print(f"{'='*80}")
    print("Evaluating algorithms...")
    
    all_results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(evaluate_algorithm, *task): task[0] 
            for task in eval_tasks
        }
        
        completed = 0
        total = len(futures)
        for future in as_completed(futures):
            algo_name = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                print(f"  [{completed}/{total}] {algo_name} completed")
            except Exception as e:
                print(f"  Error evaluating {algo_name}: {e}")
    
    # Sort results by algorithm name for consistent display
    all_results.sort(key=lambda x: x['algorithm'])
    
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS - ALL ALGORITHMS")
    print(f"{'='*80}")
    
    df = pd.DataFrame(all_results)
    df_display = df[['algorithm', 'precision', 'recall', 'f1_score', 'roc_auc', 'map', 'runtime', 'memory_mb']]
    
    print("\n" + df_display.to_string(index=False))
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save heuristic algorithms separately
    heuristic_df = df[df['algorithm'] != 'Node2Vec']
    csv_path = os.path.join(results_dir, 'heuristic_analysis.csv')
    heuristic_df.to_csv(csv_path, index=False)
    print(f"\n{'='*80}")
    print(f"Heuristic results saved to: {csv_path}")
    
    # Save all algorithms
    all_csv_path = os.path.join(results_dir, 'all_algorithms_analysis.csv')
    df.to_csv(all_csv_path, index=False)
    print(f"All algorithms results saved to: {all_csv_path}")
    
    print(f"\n{'='*80}")
    print("BEST PERFORMERS - HEURISTIC ALGORITHMS")
    print(f"{'='*80}")
    heuristic_display = heuristic_df[['algorithm', 'precision', 'recall', 'f1_score', 'roc_auc', 'map', 'runtime', 'memory_mb']]
    print(f"  Highest Precision:    {heuristic_display.loc[heuristic_display['precision'].idxmax(), 'algorithm']} ({heuristic_display['precision'].max():.4f})")
    print(f"  Highest Recall:       {heuristic_display.loc[heuristic_display['recall'].idxmax(), 'algorithm']} ({heuristic_display['recall'].max():.4f})")
    print(f"  Highest F1 Score:     {heuristic_display.loc[heuristic_display['f1_score'].idxmax(), 'algorithm']} ({heuristic_display['f1_score'].max():.4f})")
    print(f"  Highest ROC-AUC:      {heuristic_display.loc[heuristic_display['roc_auc'].idxmax(), 'algorithm']} ({heuristic_display['roc_auc'].max():.4f})")
    print(f"  Highest MAP:          {heuristic_display.loc[heuristic_display['map'].idxmax(), 'algorithm']} ({heuristic_display['map'].max():.4f})")
    print(f"  Fastest Runtime:      {heuristic_display.loc[heuristic_display['runtime'].idxmin(), 'algorithm']} ({heuristic_display['runtime'].min():.4f}s)")
    print(f"  Lowest Memory:        {heuristic_display.loc[heuristic_display['memory_mb'].idxmin(), 'algorithm']} ({heuristic_display['memory_mb'].min():.2f}MB)")
    
    print(f"\n{'='*80}")
    print("NODE2VEC PERFORMANCE (Embedding-based, not directly comparable)")
    print(f"{'='*80}")
    nv_results = df[df['algorithm'] == 'Node2Vec'].iloc[0]
    print(f"  Precision:    {nv_results['precision']:.4f}")
    print(f"  Recall:       {nv_results['recall']:.4f}")
    print(f"  F1 Score:     {nv_results['f1_score']:.4f}")
    print(f"  ROC-AUC:      {nv_results['roc_auc']:.4f}")
    print(f"  MAP:          {nv_results['map']:.4f}")
    print(f"  Runtime:      {nv_results['runtime']:.4f}s")
    print(f"  Memory:       {nv_results['memory_mb']:.2f}MB")
    
    return all_results


if __name__ == "__main__":
    G, _, _, _ = create_complete_graph(10)
    
    results = main(G, test_ratio=0.2, top_k=10, sample_size=100)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
