"""
Friend Recommendation Systems - Comprehensive Analysis
Compares performance of Adamic-Adar, Common Neighbors, and Jaccard Coefficient algorithms
on Facebook Ego Network dataset.

Author: Team aad.js
Course: Algorithm Analysis & Design
"""

import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import the friend recommendation functions from files in this folder
from aa import predict_links_for_node as aa_predict
from cm import predict_links_for_node as cm_predict
from jc import predict_links_for_node as jc_predict
from pa import predict_links_for_node as pa_predict
from ra import predict_links_for_node as ra_predict
from dw import predict_links_for_node as dw_predict, train_deepwalk_embeddings
from nv import predict_links_for_node as nv_predict, train_node2vec_embeddings

# Import 'graph.py' from the parent 'aad' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import create_complete_graph

def evaluate_algorithm_detailed(train_graph, test_edges, predict_func, k_values=[5, 10], sample_size=200):
    """
    Evaluate a friend recommendation algorithm at multiple k values.
    Uses sampling for efficiency on large graphs.
    
    Args:
        train_graph: Training graph
        test_edges: Test edges to predict
        predict_func: Function to predict links
        k_values: List of k values to evaluate
        sample_size: Number of nodes to sample for evaluation
        
    Returns:
        Dictionary with results at each k value
    """
    results = {}
    
    # Sample nodes for evaluation (for efficiency on large graphs)
    all_nodes = list(train_graph.nodes())
    np.random.seed(42)
    sample_nodes = np.random.choice(all_nodes, min(sample_size, len(all_nodes)), replace=False)
    
    print(f"  Evaluating on {len(sample_nodes)} sampled nodes...")
    
    for k in k_values:
        start_time = time.time()
        
        # Get all predictions for sampled nodes
        all_predictions = []
        test_edge_set = set(test_edges)
        
        for node in sample_nodes:
            predictions = predict_func(train_graph, node, k=k)
            for candidate, score in predictions:
                all_predictions.append((node, candidate, score))
        
        runtime = time.time() - start_time
        
        # Calculate metrics
        predicted_edges = set((u, v) for u, v, _ in all_predictions)
        
        # Also consider reverse edges since graph is undirected
        predicted_edges_with_reverse = predicted_edges | set((v, u) for u, v in predicted_edges)
        test_edge_set_with_reverse = test_edge_set | set((v, u) for u, v in test_edges)
        
        true_positives = len(predicted_edges_with_reverse.intersection(test_edge_set_with_reverse))
        
        precision = true_positives / len(predicted_edges) if predicted_edges else 0
        recall = true_positives / len(test_edges) if test_edges else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[k] = {
            'k': k,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'total_predictions': len(predicted_edges),
            'runtime': runtime
        }
        
        print(f"    k={k}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}, Runtime={runtime:.2f}s")
    
    return results


def print_top_recommendations(G, node, algo_name, predict_func, top_n=5):
    """Helper function to print top recommendations for a given node."""
    print(f"\n--- Top {top_n} Recommendations by {algo_name} for Node {node} ---")
    recommendations = predict_func(G, node, k=top_n)
    
    for i, (candidate, score) in enumerate(recommendations, 1):
        print(f"{i}. User {candidate}: Score = {score:.4f}")


def compare_algorithms_on_node(G, node, k=5):
    """Compare heuristic algorithms' recommendations for a specific node.

    For readability in the table we keep only AA/CM/JC here.
    Returns a DataFrame with combined results.
    """
    aa_recs = aa_predict(G, node, k=k)
    cm_recs = cm_predict(G, node, k=k)
    jc_recs = jc_predict(G, node, k=k)
    
    # Create comparison dataframe
    data = []
    
    # Adamic-Adar
    for i, (candidate, score) in enumerate(aa_recs, 1):
        data.append({
            'Rank': i,
            'Algorithm': 'Adamic-Adar',
            'Candidate': candidate,
            'Score': score
        })
    
    # Common Neighbors
    for i, (candidate, score) in enumerate(cm_recs, 1):
        data.append({
            'Rank': i,
            'Algorithm': 'Common Neighbors',
            'Candidate': candidate,
            'Score': score
        })
    
    # Jaccard Coefficient
    for i, (candidate, score) in enumerate(jc_recs, 1):
        data.append({
            'Rank': i,
            'Algorithm': 'Jaccard Coefficient',
            'Candidate': candidate,
            'Score': score
        })
    
    return pd.DataFrame(data)


def main():
    print("="*70)
    print("FRIEND RECOMMENDATION SYSTEMS - COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    # 1) LOAD GRAPH
    print("\nLoading complete graph from dataset...")
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.abspath(os.path.join(base_dir, "..", "dataset"))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    G, all_ego_nodes, all_circles, all_features = create_complete_graph(dataset_path=dataset_path)
    print(f"Graph loaded successfully.")
    print(f"Total Nodes: {G.number_of_nodes():,}")
    print(f"Total Edges: {G.number_of_edges():,}")
    print(f"Ego Nodes: {len(all_ego_nodes)}")
    
    # 2) TRAIN-TEST SPLIT
    print("\n" + "="*70)
    print("PREPARING TRAIN-TEST SPLIT (80-20)")
    print("="*70)
    
    edges = list(G.edges())
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(edges)
    split = int(0.8 * len(edges))
    train_edges = edges[:split]
    test_edges = edges[split:]
    
    print(f"Training edges: {len(train_edges):,}")
    print(f"Test edges: {len(test_edges):,}")
    
    # Create train graph
    train_graph = G.__class__()
    train_graph.add_nodes_from(G.nodes())
    train_graph.add_edges_from(train_edges)
    
    # 3) DELIVERABLE: Top Recommendations Comparison
    print("\n" + "="*70)
    print("TOP RECOMMENDATIONS FOR SAMPLE NODES (HEURISTIC METHODS)")
    print("="*70)
    
    sample_nodes = all_ego_nodes[:3]  # Use first 3 ego nodes
    
    for node in sample_nodes:
        print(f"\n{'='*70}")
        print(f"Recommendations for Node {node}:")
        print(f"{'='*70}")
        
    print_top_recommendations(G, node, "Adamic-Adar", aa_predict, top_n=5)
    print_top_recommendations(G, node, "Common Neighbors", cm_predict, top_n=5)
    print_top_recommendations(G, node, "Jaccard Coefficient", jc_predict, top_n=5)
    print_top_recommendations(G, node, "Preferential Attachment", pa_predict, top_n=5)
    print_top_recommendations(G, node, "Resource Allocation", ra_predict, top_n=5)
    
    # 4) DELIVERABLE: Side-by-Side Comparison Table
    print("\n" + "="*70)
    print("SIDE-BY-SIDE COMPARISON FOR NODE 0")
    print("="*70)
    
    comparison_df = compare_algorithms_on_node(G, 0, k=10)
    
    # Pivot table for better visualization
    pivot_table = comparison_df.pivot(index='Rank', columns='Algorithm', values='Candidate')
    print("\nCandidate Recommendations by Rank:")
    print(pivot_table)
    
    # Score comparison
    score_pivot = comparison_df.pivot(index='Rank', columns='Algorithm', values='Score')
    print("\nScores by Rank:")
    print(score_pivot)
    
    # 5) DELIVERABLE: Performance Evaluation at Multiple k Values
    print("\n" + "="*70)
    print("PERFORMANCE EVALUATION (TRAIN-TEST SPLIT)")
    print("="*70)
    
    # Smaller k values for faster experimentation
    k_values = [5, 10]
    
    print("\nEvaluating Adamic-Adar...")
    aa_results = evaluate_algorithm_detailed(train_graph, test_edges, aa_predict, k_values)
    
    print("Evaluating Common Neighbors...")
    cm_results = evaluate_algorithm_detailed(train_graph, test_edges, cm_predict, k_values)
    
    print("Evaluating Jaccard Coefficient...")
    jc_results = evaluate_algorithm_detailed(train_graph, test_edges, jc_predict, k_values)

    print("Evaluating Preferential Attachment...")
    pa_results = evaluate_algorithm_detailed(train_graph, test_edges, pa_predict, k_values)

    print("Evaluating Resource Allocation...")
    ra_results = evaluate_algorithm_detailed(train_graph, test_edges, ra_predict, k_values)

    # Embedding-based methods: train embeddings once on the train graph
    print("Training DeepWalk embeddings on train graph...")
    dw_embeddings = train_deepwalk_embeddings(train_graph, embedding_dim=32, num_walks=5, walk_length=20)

    def dw_predict_wrapper(graph, node, k=10):
        return dw_predict(graph, dw_embeddings, node, k=k)

    print("Evaluating DeepWalk (with fixed embeddings)...")
    dw_results = evaluate_algorithm_detailed(train_graph, test_edges, dw_predict_wrapper, k_values)

    print("Training Node2Vec embeddings on train graph...")
    nv_embeddings = train_node2vec_embeddings(train_graph, embedding_dim=32, num_walks=5, walk_length=20, p=1.0, q=1.0)

    def nv_predict_wrapper(graph, node, k=10):
        return nv_predict(graph, nv_embeddings, node, k=k)

    print("Evaluating Node2Vec (with fixed embeddings)...")
    nv_results = evaluate_algorithm_detailed(train_graph, test_edges, nv_predict_wrapper, k_values)
    
    # 6) DELIVERABLE: Performance Comparison Table
    print("\n" + "="*70)
    print("PERFORMANCE METRICS AT MULTIPLE K VALUES")
    print("="*70)
    
    # Create comparison dataframe
    performance_data = []
    
    for k in k_values:
        performance_data.append({
            'k': k,
            'Algorithm': 'Adamic-Adar',
            'Precision': aa_results[k]['precision'],
            'Recall': aa_results[k]['recall'],
            'F1-Score': aa_results[k]['f1_score'],
            'True Positives': aa_results[k]['true_positives'],
            'Runtime (s)': aa_results[k]['runtime']
        })
        
        performance_data.append({
            'k': k,
            'Algorithm': 'Common Neighbors',
            'Precision': cm_results[k]['precision'],
            'Recall': cm_results[k]['recall'],
            'F1-Score': cm_results[k]['f1_score'],
            'True Positives': cm_results[k]['true_positives'],
            'Runtime (s)': cm_results[k]['runtime']
        })
        
        performance_data.append({
            'k': k,
            'Algorithm': 'Jaccard Coefficient',
            'Precision': jc_results[k]['precision'],
            'Recall': jc_results[k]['recall'],
            'F1-Score': jc_results[k]['f1_score'],
            'True Positives': jc_results[k]['true_positives'],
            'Runtime (s)': jc_results[k]['runtime']
        })

        performance_data.append({
            'k': k,
            'Algorithm': 'Preferential Attachment',
            'Precision': pa_results[k]['precision'],
            'Recall': pa_results[k]['recall'],
            'F1-Score': pa_results[k]['f1_score'],
            'True Positives': pa_results[k]['true_positives'],
            'Runtime (s)': pa_results[k]['runtime']
        })

        performance_data.append({
            'k': k,
            'Algorithm': 'Resource Allocation',
            'Precision': ra_results[k]['precision'],
            'Recall': ra_results[k]['recall'],
            'F1-Score': ra_results[k]['f1_score'],
            'True Positives': ra_results[k]['true_positives'],
            'Runtime (s)': ra_results[k]['runtime']
        })

        performance_data.append({
            'k': k,
            'Algorithm': 'DeepWalk',
            'Precision': dw_results[k]['precision'],
            'Recall': dw_results[k]['recall'],
            'F1-Score': dw_results[k]['f1_score'],
            'True Positives': dw_results[k]['true_positives'],
            'Runtime (s)': dw_results[k]['runtime']
        })

        performance_data.append({
            'k': k,
            'Algorithm': 'Node2Vec',
            'Precision': nv_results[k]['precision'],
            'Recall': nv_results[k]['recall'],
            'F1-Score': nv_results[k]['f1_score'],
            'True Positives': nv_results[k]['true_positives'],
            'Runtime (s)': nv_results[k]['runtime']
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    # Print performance for each k
    for k in k_values:
        print(f"\n--- Performance at k={k} ---")
        k_data = performance_df[performance_df['k'] == k]
        print(k_data[['Algorithm', 'Precision', 'Recall', 'F1-Score', 'True Positives', 'Runtime (s)']].to_string(index=False))
    
    # 7) DELIVERABLE: Runtime Analysis
    print("\n" + "="*70)
    print("RUNTIME ANALYSIS SUMMARY")
    print("="*70)
    
    runtime_summary = performance_df.groupby('Algorithm')['Runtime (s)'].mean().reset_index()
    runtime_summary.columns = ['Algorithm', 'Average Runtime (s)']
    print("\nAverage Runtime across all k values:")
    print(runtime_summary.to_string(index=False))
    
    # 8) DELIVERABLE: Visualization - Performance vs k
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Precision vs k for all algorithms
    for algo in performance_df['Algorithm'].unique():
        algo_data = performance_df[performance_df['Algorithm'] == algo]
        axes[0, 0].plot(algo_data['k'], algo_data['Precision'], marker='o', label=algo)
    axes[0, 0].set_xlabel('k (Number of Recommendations)')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision vs k')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Recall vs k
    for algo in performance_df['Algorithm'].unique():
        algo_data = performance_df[performance_df['Algorithm'] == algo]
        axes[0, 1].plot(algo_data['k'], algo_data['Recall'], marker='o', label=algo)
    axes[0, 1].set_xlabel('k (Number of Recommendations)')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Recall vs k')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: F1-Score vs k
    for algo in performance_df['Algorithm'].unique():
        algo_data = performance_df[performance_df['Algorithm'] == algo]
        axes[1, 0].plot(algo_data['k'], algo_data['F1-Score'], marker='o', label=algo)
    axes[1, 0].set_xlabel('k (Number of Recommendations)')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title('F1-Score vs k')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Runtime vs k
    for algo in performance_df['Algorithm'].unique():
        algo_data = performance_df[performance_df['Algorithm'] == algo]
        axes[1, 1].plot(algo_data['k'], algo_data['Runtime (s)'], marker='o', label=algo)
    axes[1, 1].set_xlabel('k (Number of Recommendations)')
    axes[1, 1].set_ylabel('Runtime (seconds)')
    axes[1, 1].set_title('Runtime vs k')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Friend Recommendation Algorithms - Performance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    output_fig = os.path.join(results_dir, "recommendation_performance_analysis.png")
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"\nSaved performance plots to '{output_fig}'")
    
    # 9) DELIVERABLE: Algorithm Comparison Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Bar chart for each metric at a fixed k (use k=10 for speed)
    k_fixed = 10
    k_data = performance_df[performance_df['k'] == k_fixed]
    algorithms = k_data['Algorithm'].values
    
    x_pos = np.arange(len(algorithms))
    
    # Precision
    axes[0].bar(x_pos, k_data['Precision'])
    axes[0].set_xlabel('Algorithm')
    axes[0].set_ylabel('Precision')
    axes[0].set_title(f'Precision Comparison (k={k_fixed})')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(algorithms, rotation=15, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Recall
    axes[1].bar(x_pos, k_data['Recall'])
    axes[1].set_xlabel('Algorithm')
    axes[1].set_ylabel('Recall')
    axes[1].set_title(f'Recall Comparison (k={k_fixed})')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(algorithms, rotation=15, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # F1-Score
    axes[2].bar(x_pos, k_data['F1-Score'])
    axes[2].set_xlabel('Algorithm')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_title(f'F1-Score Comparison (k={k_fixed})')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(algorithms, rotation=15, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_fig2 = os.path.join(results_dir, "algorithm_comparison.png")
    plt.savefig(output_fig2, dpi=300, bbox_inches='tight')
    print(f"Saved comparison bar charts to '{output_fig2}'")
    
    # 10) DELIVERABLE: Summary Statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nOverall Best Performance (at k={k_fixed}):")
    k20_data_sorted = k_data.sort_values('F1-Score', ascending=False)
    print("\nBy F1-Score:")
    print(k20_data_sorted[['Algorithm', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
    
    print("\nBy True Positives:")
    k20_data_sorted_tp = k_data.sort_values('True Positives', ascending=False)
    print(k20_data_sorted_tp[['Algorithm', 'True Positives', 'Runtime (s)']].to_string(index=False))
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print(f"  - {output_fig}")
    print(f"  - {output_fig2}")


if __name__ == "__main__":
    main()
