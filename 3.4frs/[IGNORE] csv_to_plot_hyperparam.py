"""
Generate hyperparameter exploration plots from CSV
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV
results_dir = os.path.join(os.path.dirname(__file__), 'results')
csv_path = os.path.join(results_dir, 'nv_hyperparameter_exploration.csv')

print(f"Reading data from: {csv_path}")
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"\nData shape: {df.shape}")
print(f"Unique p values: {sorted(df['p'].unique())}")
print(f"Unique q values: {sorted(df['q'].unique())}")
print(f"Unique num_walks: {sorted(df['num_walks'].unique())}")
print(f"Unique walk_length: {sorted(df['walk_length'].unique())}")

# 1. Effect of p and q (fix num_walks=10, walk_length=80)
print("\n" + "="*60)
print("Generating p,q heatmaps (num_walks=10, walk_length=80)...")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Node2Vec Hyperparameter Analysis (p, q)', fontsize=16, fontweight='bold')

# Filter for specific walk parameters
pq_data = df[(df['num_walks'] == 10) & (df['walk_length'] == 80)]
print(f"Filtered data for p,q analysis: {len(pq_data)} rows")

if len(pq_data) > 0:
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
        print(f"  {metric_name}: pivot shape {pivot.shape}")
        
        im = ax.imshow(pivot.values, cmap='viridis', aspect='auto', interpolation='nearest')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([f'{x:.1f}' for x in pivot.columns], fontsize=10)
        ax.set_yticklabels([f'{x:.1f}' for x in pivot.index], fontsize=10)
        ax.set_xlabel('p (return parameter)', fontsize=11)
        ax.set_ylabel('q (in-out parameter)', fontsize=11)
        ax.set_title(f'{metric_name} vs p,q', fontsize=12, fontweight='bold')
        
        # Add colorbar
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

plt.tight_layout()
output_path = os.path.join(results_dir, 'nv_hyperparameter_pq.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_path}")
plt.close()

# 2. Effect of num_walks and walk_length (fix p=1.0, q=1.0)
print("\n" + "="*60)
print("Generating walk parameter plots (p=1.0, q=1.0)...")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Node2Vec Walk Parameters Analysis (p=1.0, q=1.0)', fontsize=16, fontweight='bold')

walk_data = df[(df['p'] == 1.0) & (df['q'] == 1.0)]
print(f"Filtered data for walk analysis: {len(walk_data)} rows")

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
            print(f"  {metric_name} (walk_len={wl}): {len(grouped)} points")
            ax.plot(grouped.index, grouped.values, marker='o', label=f'walk_len={wl}', 
                   linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Walks', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} vs Walk Parameters', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(results_dir, 'nv_hyperparameter_walks.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_path}")
plt.close()

# 3. Best configurations summary
print("\n" + "="*60)
print("Top 10 Configurations by F1 Score:")
print("="*60)
top_configs = df.nlargest(10, 'f1_score')[['p', 'q', 'num_walks', 'walk_length', 
                                             'f1_score', 'roc_auc', 'map', 'total_runtime']]
print(top_configs.to_string(index=False))

print("\n" + "="*60)
print("Top 10 Configurations by ROC-AUC:")
print("="*60)
top_auc = df.nlargest(10, 'roc_auc')[['p', 'q', 'num_walks', 'walk_length', 
                                        'f1_score', 'roc_auc', 'map', 'total_runtime']]
print(top_auc.to_string(index=False))

print("\n" + "="*60)
print("Plots generated successfully!")
print("="*60)
