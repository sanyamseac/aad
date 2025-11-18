"""
Scalability Analysis for Friend Recommendation Algorithms
Tests runtime and memory performance across different graph sizes.
Compares theoretical vs actual time complexity.
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

from cm import recommend_friends as cm_recommend
from aa import recommend_friends as aa_recommend
from jc import recommend_friends as jc_recommend
from pa import recommend_friends as pa_recommend
from ra import recommend_friends as ra_recommend


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


def evaluate_algorithm_on_graph(graph_data, algo_name, recommend_func, top_k=10, sample_size=50):
    """Evaluate a single algorithm's scalability on pre-loaded graph data"""
    graph_id, nx_G, sample_nodes = graph_data
    
    process = psutil.Process()
    process.memory_info()
    start_memory = process.memory_info().rss / 1024 / 1024
    start_time = time.time()
    
    # Just run recommendations to measure time and memory
    for node in sample_nodes:
        _ = recommend_func(nx_G, node, top_k)
    
    runtime = time.time() - start_time
    end_memory = process.memory_info().rss / 1024 / 1024
    memory_used = max(0.1, end_memory - start_memory)
    
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
        'runtime_per_node': runtime / len(sample_nodes) if len(sample_nodes) > 0 else 0
    }


def load_and_prepare_graph(graph_id, sample_size=50):
    """Load a graph and prepare it for evaluation"""
    try:
        nx_G, _, _, _ = create_complete_graph(graph_id)
        all_nodes = list(nx_G.nodes())
        sample_nodes = random.sample(all_nodes, min(sample_size, len(all_nodes)))
        return (graph_id, nx_G, sample_nodes)
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
    plot_path = os.path.join(results_dir, 'scalability_analysis.png')
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


def main():
    print("="*80)
    print("SCALABILITY ANALYSIS - FRIEND RECOMMENDATION ALGORITHMS")
    print("="*80)
    
    graph_ids = list(range(1, 11))
    
    algorithms = [
        ("Common Neighbors", cm_recommend),
        ("Adamic-Adar", aa_recommend),
        ("Jaccard Coefficient", jc_recommend),
        ("Preferential Attachment", pa_recommend),
        ("Resource Allocation", ra_recommend)
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
        for algo_name, recommend_func in algorithms:
            tasks.append((graph_data, algo_name, recommend_func, 10, 50))
    
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
    print("SCALABILITY RESULTS")
    print(f"{'='*80}\n")
    
    summary_cols = ['algorithm', 'graph_id', 'nodes', 'edges', 'avg_degree', 
                    'runtime', 'memory_mb', 'runtime_per_node']
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
        'runtime': ['mean', 'std', 'min', 'max'],
        'memory_mb': ['mean', 'std', 'min', 'max'],
        'runtime_per_node': ['mean', 'std']
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
    
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}")
    
    return df


if __name__ == "__main__":
    results_df = main()
