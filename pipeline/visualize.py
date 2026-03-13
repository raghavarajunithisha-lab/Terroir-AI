"""
Terroir AI — Phase 6: Visualization
=============================================================
Generates:
  1. Network graph visualization (3 cultivars side by side)
  2. Node importance bar chart
  3. Predicted vs Actual comparison
  4. Training loss curve
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import torch
import json
import os

# ─── Configuration ──────────────────────────────────────────────────
GRAPH_DIR = "d:/ChitoseProject/terroir_ai/pipeline/graphs"
OUT_DIR = "d:/ChitoseProject/terroir_ai/pipeline/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Load results
with open(f"{OUT_DIR}/analysis_results.json") as f:
    results = json.load(f)

losses_df = pd.read_csv(f"{OUT_DIR}/training_loss.csv")

# Load taxonomy for nicer labels
supp = "d:/ChitoseProject/terroir_ai/supplementary/extracted"
core = pd.read_csv(f"{supp}/TableS8_core_microbiome.csv")
core = core[core['OTU'] != 'OTU']
pathogens = pd.read_csv(f"{supp}/TableS4_fungal_pathogens.csv")
pathogens = pathogens[pathogens['OTU'] != 'OTU']

# OTU → short name lookup
otu_names = {}
for _, row in core.iterrows():
    genus = row.get('Genus', '-')
    if genus and genus != '-':
        otu_names[row['OTU']] = genus
    else:
        otu_names[row['OTU']] = row['OTU']
for _, row in pathogens.iterrows():
    name = str(row.get('UNITE species identification', ''))
    if name and name != 'nan':
        otu_names[row['OTU']] = name.split()[0]  # first word
    else:
        otu_names[row['OTU']] = row['OTU']

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except (IOError, OSError):
    try:
        plt.style.use('seaborn-darkgrid')
    except (IOError, OSError):
        plt.style.use('ggplot')

# ═══════════════════════════════════════════════════════════════════
# PLOT 1: Training Loss Curve
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(losses_df['epoch'], losses_df['loss'], color='#2196F3', linewidth=1.5)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss (MSE)', fontsize=12)
ax.set_title('Terroir AI GNN — Training Loss', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.axhline(y=results['training']['best_loss'], color='#f44336', linestyle='--', 
           label=f"Best: {results['training']['best_loss']:.6f}")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_loss.png", dpi=150, bbox_inches='tight')
print("Saved: training_loss.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# PLOT 2: Predicted vs Actual Flavor Comparison
# ═══════════════════════════════════════════════════════════════════
traits = ['Brix (Sweetness)', 'pH', 'Texture (N)', 'Acidity']
cultivars_list = ['Darselect', 'Elsanta', 'Monterey']
colors = ['#4CAF50', '#FF9800', '#2196F3']

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Terroir AI — Predicted vs Actual Flavor Profiles', fontsize=14, fontweight='bold')

for i, trait in enumerate(traits):
    ax = axes[i]
    actual_vals = [r['actual'][trait] for r in results['predictions']]
    pred_vals = [r['predicted'][trait] for r in results['predictions']]
    
    x = np.arange(len(cultivars_list))
    width = 0.35
    
    ax.bar(x - width/2, actual_vals, width, label='Actual', color=colors, alpha=0.8)
    ax.bar(x + width/2, pred_vals, width, label='Predicted', color=colors, alpha=0.4,
           edgecolor=colors, linewidth=2)
    
    ax.set_title(trait, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['D', 'E', 'M'], fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/predicted_vs_actual.png", dpi=150, bbox_inches='tight')
print("Saved: predicted_vs_actual.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# PLOT 3: Node Importance — Top 10 Microbes per Cultivar
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Terroir AI — Most Important Microbes per Cultivar', fontsize=14, fontweight='bold')

for idx, cv in enumerate(cultivars_list):
    ax = axes[idx]
    top = results['node_importance'][cv]
    
    names = [otu_names.get(t['otu'], t['otu']) for t in top]
    scores = [t['importance'] for t in top]
    
    bars = ax.barh(range(len(names)), scores, color=colors[idx], alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.set_title(f'{cv}', fontsize=12, fontweight='bold')
    
    # Add OTU ID annotations
    for j, (bar, t) in enumerate(zip(bars, top)):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                t['otu'], va='center', fontsize=7, color='gray')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/node_importance.png", dpi=150, bbox_inches='tight')
print("Saved: node_importance.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# PLOT 4: Network Graph Visualization (one per cultivar)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('Terroir AI — Microbial Interaction Networks', fontsize=16, fontweight='bold')

for idx, cv in enumerate(cultivars_list):
    ax = axes[idx]
    
    # Load graph
    g = torch.load(f"{GRAPH_DIR}/{cv}_graph.pt", weights_only=False)
    
    # Build networkx graph (show only top edges for clarity)
    G = nx.Graph()
    
    # Add all nodes
    for i, name in enumerate(g.node_names):
        G.add_node(i, name=name)
    
    # Add edges (subsample for readability — top 200 strongest)
    edge_index = g.edge_index.numpy()
    edge_weights = g.edge_attr.numpy().flatten()
    
    # Get unique edges with weights
    edges = []
    seen = set()
    for k in range(edge_index.shape[1]):
        src, dst = edge_index[0, k], edge_index[1, k]
        if (min(src, dst), max(src, dst)) not in seen:
            seen.add((min(src, dst), max(src, dst)))
            edges.append((src, dst, edge_weights[k]))
    
    # Sort by absolute weight, keep top 150
    edges.sort(key=lambda x: abs(x[2]), reverse=True)
    top_edges = edges[:150]
    
    for src, dst, w in top_edges:
        G.add_edge(src, dst, weight=w)
    
    # Node sizes based on importance
    top_otus = {t['otu']: t['importance'] for t in results['node_importance'][cv]}
    node_sizes = []
    node_colors_list = []
    
    for i, name in enumerate(g.node_names):
        if name in top_otus:
            node_sizes.append(300 + top_otus[name] * 20)
            node_colors_list.append('#f44336')  # red for important
        else:
            node_sizes.append(30)
            node_colors_list.append('#90CAF9')  # light blue for others
    
    # Only draw nodes that have edges in our subgraph
    nodelist = list(G.nodes())
    sizes = [node_sizes[n] if n < len(node_sizes) else 30 for n in nodelist]
    ncolors = [node_colors_list[n] if n < len(node_colors_list) else '#90CAF9' for n in nodelist]
    
    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Draw edges (color by positive/negative)
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 0) > 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 0) < 0]
    
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, ax=ax, 
                           edge_color='#4CAF50', alpha=0.15, width=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, ax=ax,
                           edge_color='#f44336', alpha=0.15, width=0.5, style='dashed')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=sizes,
                           node_color=ncolors, ax=ax, alpha=0.8, edgecolors='white', linewidths=0.5)
    
    # Label top 5 nodes
    top5 = results['node_importance'][cv][:5]
    labels = {}
    for t in top5:
        otu_id = t['otu']
        if otu_id in g.node_names:
            node_idx = g.node_names.index(otu_id)
            if node_idx in pos:
                labels[node_idx] = otu_names.get(otu_id, otu_id)
    
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight='bold')
    
    # Flavor info
    flavor = results['predictions'][idx]['actual']
    ax.set_title(f"{cv}\nBrix={flavor['Brix (Sweetness)']:.1f}  Texture={flavor['Texture (N)']:.1f}",
                 fontsize=12, fontweight='bold')
    ax.axis('off')

# Legend
legend_elements = [
    mpatches.Patch(color='#f44336', label='Key Microbe (high importance)'),
    mpatches.Patch(color='#90CAF9', label='Other Microbe'),
    plt.Line2D([0], [0], color='#4CAF50', label='Positive correlation (cooperate)'),
    plt.Line2D([0], [0], color='#f44336', linestyle='--', label='Negative correlation (compete)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(f"{OUT_DIR}/network_graphs.png", dpi=150, bbox_inches='tight')
print("Saved: network_graphs.png")
plt.close()

print(f"\nAll visualizations saved to: {OUT_DIR}/")
print("Files: training_loss.png, predicted_vs_actual.png, node_importance.png, network_graphs.png")
