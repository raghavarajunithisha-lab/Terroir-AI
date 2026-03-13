"""
Terroir AI — Phase 4 & 5: GNN Model + Training + Analysis
=============================================================
- Graph Convolutional Network with attention for node importance
- Trains on 3 cultivar graphs to predict flavor profiles
- Extracts which microbes matter most for each flavor trait
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
import json
import os

# ─── Configuration ──────────────────────────────────────────────────
GRAPH_DIR = "d:/ChitoseProject/terroir_ai/pipeline/graphs"
OUT_DIR = "d:/ChitoseProject/terroir_ai/pipeline/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS = 500
LR = 0.005
HIDDEN_DIM = 64
NUM_TARGETS = 4  # Brix, pH, Texture, Acidity

# ─── Load graphs ─────────────────────────────────────────────────────
cultivars = ['Darselect', 'Elsanta', 'Monterey']
graphs = []
for cv in cultivars:
    g = torch.load(f"{GRAPH_DIR}/{cv}_graph.pt", weights_only=False)
    g.batch = torch.zeros(g.num_nodes, dtype=torch.long)  # single graph batch
    graphs.append(g)
    print(f"Loaded {cv}: {g.num_nodes} nodes, {g.num_edges} edges, target={g.y.tolist()}")


# ─── GNN Model ──────────────────────────────────────────────────────
class TerroirGNN(nn.Module):
    """
    Graph Neural Network for Terroir AI.
    
    Architecture:
    - 2 Graph Attention layers (learns WHICH neighbors matter)
    - Global mean pooling (summarizes entire graph into one vector)
    - 2 Linear layers (predicts flavor from graph summary)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        # Normalize input features
        self.input_norm = nn.BatchNorm1d(in_channels)
        
        # Graph Attention layers — learn which connections matter
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.gat2 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # Prediction head
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)
        
        # Store attention weights for analysis
        self.node_importance = None
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Normalize input
        x = self.input_norm(x)
        
        # GAT layer 1 — each node "asks" its neighbors for info
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # GAT layer 2 — deeper understanding of neighborhood
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Store node embeddings for importance analysis
        self.node_importance = x.detach()
        
        # Global mean pool — summarize entire graph into one vector
        graph_embedding = global_mean_pool(x, batch)
        
        # Predict flavor profile
        out = F.relu(self.fc1(graph_embedding))
        out = self.fc2(out)
        
        return out


# ─── Training ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("Training Terroir AI GNN")
print("="*60)

model = TerroirGNN(
    in_channels=graphs[0].x.shape[1],  # 5 node features
    hidden_channels=HIDDEN_DIM,
    out_channels=NUM_TARGETS,
)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
criterion = nn.MSELoss()

# Normalize targets for better training
all_targets = torch.cat([g.y for g in graphs], dim=0)
target_mean = all_targets.mean(dim=0)
target_std = all_targets.std(dim=0)
target_std[target_std == 0] = 1  # avoid div by zero

# Training loop
losses = []
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for g in graphs:
        optimizer.zero_grad()
        pred = model(g)
        
        # Normalize target
        target_norm = (g.y - target_mean) / target_std
        loss = criterion(pred, target_norm)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(graphs)
    losses.append(avg_loss)
    scheduler.step(avg_loss)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), f"{OUT_DIR}/best_model.pt")
    
    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")

print(f"\nTraining complete. Best loss: {best_loss:.6f}")

# ─── Evaluation ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("Evaluation: Predicted vs Actual Flavor")
print("="*60)

model.load_state_dict(torch.load(f"{OUT_DIR}/best_model.pt", weights_only=True))
model.eval()

trait_names = ['Brix (Sweetness)', 'pH', 'Texture (N)', 'Acidity']
results = []

for g in graphs:
    with torch.no_grad():
        pred_norm = model(g)
        # Denormalize
        pred = pred_norm * target_std + target_mean
    
    actual = g.y[0].tolist()
    predicted = pred[0].tolist()
    
    print(f"\n  {g.cultivar}:")
    for i, trait in enumerate(trait_names):
        print(f"    {trait:20s}  Actual: {actual[i]:6.2f}  |  Predicted: {predicted[i]:6.2f}")
    
    results.append({
        'cultivar': g.cultivar,
        'actual': {t: actual[i] for i, t in enumerate(trait_names)},
        'predicted': {t: round(predicted[i], 2) for i, t in enumerate(trait_names)},
    })

# ─── Node Importance Analysis ───────────────────────────────────────
print("\n" + "="*60)
print("Node Importance Analysis: Which Microbes Matter Most?")
print("="*60)

importance_results = {}

for g in graphs:
    with torch.no_grad():
        _ = model(g)
        node_emb = model.node_importance  # [num_nodes, hidden_dim]
    
    # Node importance = L2 norm of learned embedding
    importance = torch.norm(node_emb, dim=1).numpy()
    
    # Rank nodes
    ranked_indices = np.argsort(importance)[::-1]
    
    print(f"\n  Top 10 most important microbes for {g.cultivar}:")
    top_microbes = []
    for rank, idx in enumerate(ranked_indices[:10]):
        name = g.node_names[idx]
        score = importance[idx]
        print(f"    {rank+1}. {name:15s}  importance: {score:.4f}")
        top_microbes.append({'rank': rank+1, 'otu': name, 'importance': round(float(score), 4)})
    
    importance_results[g.cultivar] = top_microbes

# ─── Save all results ───────────────────────────────────────────────
output = {
    'predictions': results,
    'node_importance': importance_results,
    'training': {
        'epochs': EPOCHS,
        'best_loss': best_loss,
        'model_params': sum(p.numel() for p in model.parameters()),
    }
}

with open(f"{OUT_DIR}/analysis_results.json", 'w') as f:
    json.dump(output, f, indent=2)

# Save losses for plotting
pd.DataFrame({'epoch': range(EPOCHS), 'loss': losses}).to_csv(
    f"{OUT_DIR}/training_loss.csv", index=False
)

print(f"\nResults saved to: {OUT_DIR}/analysis_results.json")
print(f"Training loss saved to: {OUT_DIR}/training_loss.csv")
print(f"Best model saved to: {OUT_DIR}/best_model.pt")
