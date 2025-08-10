# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, global_mean_pool
from torch_geometric.nn.pool import TopKPooling
import math

class ImprovedGCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GCN layers with batch normalization
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Attention pooling
        self.attention = nn.MultiheadAttention(hidden_channels, num_heads=4, batch_first=True)
        
        # Classification head with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, out_channels)
        )
        
        # Residual connection
        self.residual_proj = nn.Linear(in_channels, hidden_channels)
        
    def forward(self, x, edge_index, batch=None):
        # Input projection
        x_proj = self.input_proj(x)
        residual = self.residual_proj(x)
        
        # GCN layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_new = conv(x_proj, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection every 2 layers
            if i % 2 == 1:
                x_proj = x_new + residual
                residual = x_proj
            else:
                x_proj = x_new
        
        # Skip self-attention for large graphs to prevent OOM
        # Self-attention on 20k+ nodes requires too much memory (~13GB)
        if batch is None and x_proj.size(0) > 5000:
            # For large graphs, use simple global pooling instead of attention
            x_att = x_proj  # Skip attention to save memory
        elif batch is None:
            # For smaller graphs, use self-attention
            x_att, _ = self.attention(x_proj.unsqueeze(0), x_proj.unsqueeze(0), x_proj.unsqueeze(0))
            x_att = x_att.squeeze(0)
        else:
            # Handle batched graphs
            x_att = x_proj
        
        return self.classifier(x_att)

class ImprovedGATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, num_layers=3, dropout=0.3, heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GAT layers with different head configurations
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == num_layers - 1:  # Last layer
                self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout))
            else:
                self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
            self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Feature importance weighting
        self.feature_importance = nn.Parameter(torch.ones(hidden_channels))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, out_channels)
        )
        
    def forward(self, x, edge_index, batch=None):
        x = self.input_proj(x)
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)  # ELU activation for GAT
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply feature importance weighting
        x = x * self.feature_importance
        
        return self.classifier(x)

class AdvancedHybridGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, num_layers=3, 
                 dropout=0.3, fusion_method='advanced_attention'):
        super().__init__()
        self.fusion_method = fusion_method
        
        # Improved component networks
        self.gcn = ImprovedGCNNet(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
        self.gat = ImprovedGATNet(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
        
        # Advanced fusion mechanisms
        if fusion_method == 'advanced_attention':
            self.fusion_attention = nn.MultiheadAttention(hidden_channels, num_heads=8, batch_first=True)
            self.fusion_norm = nn.LayerNorm(hidden_channels)
        elif fusion_method == 'cross_attention':
            self.cross_attention = nn.MultiheadAttention(hidden_channels, num_heads=4, batch_first=True)
        elif fusion_method == 'gated_fusion':
            self.gate = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.Sigmoid()
            )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(hidden_channels, 1)
        
        # Final classification with uncertainty
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
    def forward(self, x, edge_index, batch=None, return_uncertainty=False):
        # Get representations from both networks
        gcn_out = self.gcn(x, edge_index, batch)
        gat_out = self.gat(x, edge_index, batch)
        
        # Advanced fusion with memory safety
        if self.fusion_method == 'advanced_attention':
            # For very large graphs, skip attention to prevent OOM
            if gcn_out.size(0) > 10000:
                # Fallback to simple weighted fusion for very large graphs
                fused = 0.6 * gcn_out + 0.4 * gat_out
            else:
                # Stack for attention
                stacked = torch.stack([gcn_out, gat_out], dim=1)  # [N, 2, hidden]
                fused, attention_weights = self.fusion_attention(stacked, stacked, stacked)
                fused = self.fusion_norm(fused.mean(dim=1))  # Average over sequence dimension
            
        elif self.fusion_method == 'cross_attention':
            # For very large graphs, skip cross-attention to prevent OOM
            if gcn_out.size(0) > 10000:
                # Fallback to simple weighted fusion for very large graphs
                fused = 0.6 * gcn_out + 0.4 * gat_out
            else:
                fused, _ = self.cross_attention(gcn_out.unsqueeze(1), gat_out.unsqueeze(1), gat_out.unsqueeze(1))
                fused = fused.squeeze(1)
            
        elif self.fusion_method == 'gated_fusion':
            combined = torch.cat([gcn_out, gat_out], dim=-1)
            gate_weights = self.gate(combined)
            fused = gate_weights * gcn_out + (1 - gate_weights) * gat_out
            
        else:  # Default weighted sum
            fused = 0.6 * gcn_out + 0.4 * gat_out
        
        # Uncertainty estimation
        uncertainty = torch.sigmoid(self.uncertainty_head(fused))
        
        # Final prediction
        logits = self.final_classifier(fused)
        
        if return_uncertainty:
            return logits, uncertainty
        return logits

# Legacy classes for backward compatibility
class GCNNet(ImprovedGCNNet):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, num_layers=3, dropout=0.3):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)

class GATNet(ImprovedGATNet):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, num_layers=3, dropout=0.3):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)

class HybridGNN(AdvancedHybridGNN):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, num_layers=3, 
                 dropout=0.3, fusion_method='attention'):
        # Map old fusion methods to new ones
        fusion_mapping = {
            'attention': 'advanced_attention',
            'concat': 'gated_fusion',
            'weighted': 'advanced_attention'
        }
        new_fusion_method = fusion_mapping.get(fusion_method, 'advanced_attention')
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, new_fusion_method)
