"""
Terroir AI — Phase 3: Build Graphs from OTU Abundance Table
=============================================================
For each cultivar:
  1. Extract its 32 samples from the abundance table
  2. Calculate Spearman correlations between all OTU pairs
  3. Filter edges (keep |corr| > threshold)
  4. Add node features (taxonomy, functions, traits)
  5. Attach flavor labels from TableS12
  6. Save as PyTorch Geometric Data objects
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from scipy.stats import spearmanr
import os
import json

# ─── Configuration ──────────────────────────────────────────────────
EDGE_THRESHOLD = 0.4        # Keep edges with |correlation| > this
MIN_ABUNDANCE = 10          # Drop OTUs with mean abundance < this per cultivar
DATA_DIR = "d:/ChitoseProject/terroir_ai/pipeline/data"
SUPP_DIR = "d:/ChitoseProject/terroir_ai/supplementary/extracted"
OUT_DIR = "d:/ChitoseProject/terroir_ai/pipeline/graphs"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Load data ──────────────────────────────────────────────────────
print("Loading data...")
otu_df = pd.read_csv(f"{DATA_DIR}/otu_abundance_table.csv")
core = pd.read_csv(f"{SUPP_DIR}/TableS8_core_microbiome.csv")
core = core[core['OTU'] != 'OTU'].reset_index(drop=True)
pathogens = pd.read_csv(f"{SUPP_DIR}/TableS4_fungal_pathogens.csv")
pathogens = pathogens[pathogens['OTU'] != 'OTU'].reset_index(drop=True)
functions = pd.read_csv(f"{SUPP_DIR}/TableS9_microbial_functions.csv")

# Fruit quality targets (from TableS12)
flavor_data = {
    'Darselect': {'Brix': 6.37, 'pH': 3.75, 'Texture': 2.29, 'Acidity': 3.71},
    'Elsanta':   {'Brix': 11.55, 'pH': 3.68, 'Texture': 2.20, 'Acidity': 3.94},
    'Monterey':  {'Brix': 5.67, 'pH': 3.61, 'Texture': 3.05, 'Acidity': 2.70},
}

# ─── Build taxonomy lookup ──────────────────────────────────────────
# Encode phylum as a number
all_phyla = list(set(
    core['Phylum'].dropna().tolist() + pathogens['Phylum'].dropna().tolist()
))
phylum_to_idx = {p: i for i, p in enumerate(all_phyla)}

# Build OTU → info lookup
otu_info = {}
for _, row in core.iterrows():
    otu_info[row['OTU']] = {
        'phylum': row.get('Phylum', ''),
        'genus': row.get('Genus', ''),
        'is_core': 1,
        'is_pathogen': 0,
    }
for _, row in pathogens.iterrows():
    otu_id = row['OTU']
    if otu_id not in otu_info:
        otu_info[otu_id] = {
            'phylum': row.get('Phylum', ''),
            'genus': '',
            'is_core': 0,
            'is_pathogen': 1,
        }
    else:
        otu_info[otu_id]['is_pathogen'] = 1

# ─── Build one graph per cultivar ────────────────────────────────────
cultivars = ['Darselect', 'Elsanta', 'Monterey']
otu_columns = [c for c in otu_df.columns if c.startswith('Otu')]

graph_info = {}

for cv in cultivars:
    print(f"\n{'='*60}")
    print(f"Building graph for: {cv}")
    print(f"{'='*60}")
    
    # Get this cultivar's samples
    cv_data = otu_df[otu_df['cultivar'] == cv][otu_columns]
    
    # Filter out OTUs with very low abundance
    mean_abundance = cv_data.mean()
    keep_otus = mean_abundance[mean_abundance >= MIN_ABUNDANCE].index.tolist()
    cv_data = cv_data[keep_otus]
    print(f"  OTUs kept (mean abundance >= {MIN_ABUNDANCE}): {len(keep_otus)}")
    
    # ─── Calculate Spearman correlations ─────────────────────────────
    corr_matrix, p_matrix = spearmanr(cv_data.values)
    
    # Handle case where only one OTU survives filtering
    if isinstance(corr_matrix, float):
        print(f"  WARNING: Not enough OTUs for correlation. Skipping {cv}.")
        continue
    
    # ─── Create edges from correlations ──────────────────────────────
    n_otus = len(keep_otus)
    edge_src = []
    edge_dst = []
    edge_weights = []
    
    for i in range(n_otus):
        for j in range(i + 1, n_otus):
            corr_val = corr_matrix[i, j]
            p_val = p_matrix[i, j]
            
            if abs(corr_val) >= EDGE_THRESHOLD and p_val < 0.05:
                # Undirected: add both directions
                edge_src.extend([i, j])
                edge_dst.extend([j, i])
                edge_weights.extend([corr_val, corr_val])
    
    print(f"  Edges created (|corr| >= {EDGE_THRESHOLD}, p < 0.05): {len(edge_src) // 2}")
    
    # ─── Build node features ─────────────────────────────────────────
    # Feature vector per node: [mean_abundance, std_abundance, phylum_encoded, is_core, is_pathogen]
    node_features = []
    node_names = []
    
    for otu_name in keep_otus:
        info = otu_info.get(otu_name, {})
        mean_ab = float(cv_data[otu_name].mean())
        std_ab = float(cv_data[otu_name].std())
        phylum = info.get('phylum', '')
        phylum_enc = phylum_to_idx.get(phylum, len(all_phyla))
        is_core = info.get('is_core', 0)
        is_pathogen = info.get('is_pathogen', 0)
        
        node_features.append([mean_ab, std_ab, phylum_enc, is_core, is_pathogen])
        node_names.append(otu_name)
    
    # ─── Create PyTorch Geometric Data object ────────────────────────
    x = torch.tensor(node_features, dtype=torch.float)
    
    if len(edge_src) > 0:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    
    # Graph-level label (flavor profile)
    flavor = flavor_data[cv]
    y = torch.tensor([[flavor['Brix'], flavor['pH'], flavor['Texture'], flavor['Acidity']]],
                      dtype=torch.float)
    
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    graph.cultivar = cv
    graph.node_names = node_names
    
    # Save
    graph_path = f"{OUT_DIR}/{cv}_graph.pt"
    torch.save(graph, graph_path)
    
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.num_edges}")
    print(f"  Node features: {graph.x.shape}")
    print(f"  Label (Brix, pH, Texture, Acidity): {y.tolist()[0]}")
    print(f"  Saved: {graph_path}")
    
    graph_info[cv] = {
        'nodes': int(graph.num_nodes),
        'edges': int(graph.num_edges),
        'node_names': node_names[:10],  # first 10 for preview
        'flavor': flavor,
    }

# Save graph summary
summary_path = f"{OUT_DIR}/graph_summary.json"
with open(summary_path, 'w') as f:
    json.dump(graph_info, f, indent=2)
print(f"\nGraph summary saved: {summary_path}")
print("\nDONE! All 3 cultivar graphs built.")
