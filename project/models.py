# models.py - Enhanced hybrid GNN models for Lassa fever diagnosis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, global_mean_pool, EdgeConv
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.utils import dropout_adj
import math

class EnhancedGCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection with medical feature attention
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.feature_attention = nn.MultiheadAttention(hidden_channels, num_heads=4, batch_first=True)
        
        # Enhanced GCN layers with residual connections
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, improved=True, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hidden_channels))
            if i > 0:  # Residual connections after first layer
                self.residual_projections.append(nn.Linear(hidden_channels, hidden_channels))
        
        # Medical domain-specific pooling
        self.medical_pooling = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels)
        )
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_channels // 4, out_channels)
        )
        
        # Medical uncertainty estimation
        self.uncertainty_head = nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, edge_weight=None, batch=None, return_uncertainty=False):
        # Input projection with feature attention
        x_proj = self.input_proj(x)
        
        # Apply feature attention for medical relevance
        if batch is None and x_proj.size(0) < 5000:  # Memory-safe attention
            x_att, _ = self.feature_attention(x_proj.unsqueeze(0), x_proj.unsqueeze(0), x_proj.unsqueeze(0))
            x_proj = x_att.squeeze(0) + x_proj  # Residual connection
        
        # Enhanced GCN layers with residual connections
        x_residual = x_proj
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # Apply edge dropout for regularization
            if self.training and edge_weight is not None:
                edge_index_drop, edge_weight_drop = dropout_adj(edge_index, edge_weight, p=0.1)
            else:
                edge_index_drop, edge_weight_drop = edge_index, edge_weight
            
            x_new = conv(x_proj, edge_index_drop, edge_weight_drop)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection every layer after first
            if i > 0:
                x_residual_proj = self.residual_projections[i-1](x_residual)
                x_proj = x_new + x_residual_proj
                x_residual = x_proj
            else:
                x_proj = x_new
                x_residual = x_proj
        
        # Medical domain pooling
        x_pooled = self.medical_pooling(x_proj)
        x_final = x_proj + x_pooled  # Residual connection
        
        # Classification
        logits = self.classifier(x_final)
        
        if return_uncertainty:
            uncertainty = torch.sigmoid(self.uncertainty_head(x_final))
            return logits, uncertainty
        
        return logits

class EnhancedGATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, num_layers=3, dropout=0.3, heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Enhanced GAT layers with edge attention
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.edge_attentions = nn.ModuleList()
        
        for i in range(num_layers):
            if i == num_layers - 1:  # Last layer
                self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout, 
                                        add_self_loops=True, edge_dim=1))
            else:
                self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, 
                                        dropout=dropout, add_self_loops=True, edge_dim=1))
            self.batch_norms.append(BatchNorm(hidden_channels))
            
            # Edge attention for medical relevance
            self.edge_attentions.append(nn.Sequential(
                nn.Linear(1, hidden_channels // 4),
                nn.ReLU(),
                nn.Linear(hidden_channels // 4, 1),
                nn.Sigmoid()
            ))
        
        # Medical symptom attention mechanism
        self.symptom_attention = nn.Parameter(torch.ones(hidden_channels))
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_channels // 4, out_channels)
        )
        
        # Medical uncertainty estimation
        self.uncertainty_head = nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, edge_weight=None, batch=None, return_uncertainty=False):
        x = self.input_proj(x)
        
        for i, (conv, bn, edge_att) in enumerate(zip(self.convs, self.batch_norms, self.edge_attentions)):
            # Enhance edge weights with medical attention
            if edge_weight is not None:
                enhanced_edge_weight = edge_weight.unsqueeze(-1)
                edge_attention = edge_att(enhanced_edge_weight).squeeze(-1)
                final_edge_weight = edge_weight * edge_attention
            else:
                final_edge_weight = None
            
            x = conv(x, edge_index, final_edge_weight)
            x = bn(x)
            x = F.elu(x)  # ELU activation for GAT
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply medical symptom attention
        x = x * self.symptom_attention
        
        # Classification
        logits = self.classifier(x)
        
        if return_uncertainty:
            uncertainty = torch.sigmoid(self.uncertainty_head(x))
            return logits, uncertainty
        
        return logits

class SuperiorHybridGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, num_layers=3, 
                 dropout=0.3, fusion_method='medical_attention'):
        super().__init__()
        self.fusion_method = fusion_method
        self.hidden_channels = hidden_channels
        
        # Enhanced component networks
        self.gcn = EnhancedGCNNet(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
        self.gat = EnhancedGATNet(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
        
        # Medical domain fusion mechanisms
        if fusion_method == 'medical_attention':
            self.medical_fusion = nn.MultiheadAttention(hidden_channels, num_heads=8, batch_first=True)
            self.fusion_norm = nn.LayerNorm(hidden_channels)
            self.medical_gate = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.Sigmoid()
            )
        elif fusion_method == 'clinical_weighting':
            # Clinical importance weighting
            self.clinical_weights = nn.Parameter(torch.tensor([0.6, 0.4]))  # GCN, GAT
            self.clinical_fusion = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels)
            )
        elif fusion_method == 'adaptive_gating':
            self.adaptive_gate = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 2),
                nn.Softmax(dim=-1)
            )
        
        # Medical uncertainty estimation
        self.uncertainty_head = nn.Linear(hidden_channels, 1)
        
        # Enhanced final classification with medical focus
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_channels // 4, out_channels)
        )
        
        # Medical interpretability features
        self.feature_importance = nn.Linear(hidden_channels, in_channels)
        
    def forward(self, x, edge_index, edge_weight=None, batch=None, return_uncertainty=False, return_attention=False):
        # Get representations from enhanced networks
        gcn_out = self.gcn(x, edge_index, edge_weight, batch)
        gat_out = self.gat(x, edge_index, edge_weight, batch)
        
        # Medical domain fusion
        if self.fusion_method == 'medical_attention':
            # Memory-safe attention for large graphs
            if gcn_out.size(0) > 10000:
                # Fallback to gated fusion for very large graphs
                combined = torch.cat([gcn_out, gat_out], dim=-1)
                gate = self.medical_gate(combined)
                fused = gate * gcn_out + (1 - gate) * gat_out
                attention_weights = None
            else:
                # Medical attention fusion
                stacked = torch.stack([gcn_out, gat_out], dim=1)  # [N, 2, hidden]
                fused, attention_weights = self.medical_fusion(stacked, stacked, stacked)
                fused = self.fusion_norm(fused.mean(dim=1))  # Average over sequence dimension
                
        elif self.fusion_method == 'clinical_weighting':
            # Clinical importance weighting
            w1, w2 = F.softmax(self.clinical_weights, dim=0)
            weighted = w1 * gcn_out + w2 * gat_out
            combined = torch.cat([gcn_out, gat_out], dim=-1)
            fused = weighted + self.clinical_fusion(combined)
            attention_weights = self.clinical_weights
            
        elif self.fusion_method == 'adaptive_gating':
            # Adaptive gating based on input
            combined = torch.cat([gcn_out, gat_out], dim=-1)
            gates = self.adaptive_gate(combined)  # [N, 2]
            fused = gates[:, 0:1] * gcn_out + gates[:, 1:2] * gat_out
            attention_weights = gates
            
        else:  # Default weighted sum with learnable weights
            fused = 0.6 * gcn_out + 0.4 * gat_out
            attention_weights = None
        
        # Medical uncertainty estimation
        uncertainty = torch.sigmoid(self.uncertainty_head(fused))
        
        # Final prediction
        logits = self.final_classifier(fused)
        
        # Feature importance for interpretability
        feature_importance = torch.sigmoid(self.feature_importance(fused))
        
        if return_uncertainty and return_attention:
            return logits, uncertainty, attention_weights, feature_importance
        elif return_uncertainty:
            return logits, uncertainty
        elif return_attention:
            return logits, attention_weights
        
        return logits

# Legacy compatibility - enhanced versions
class GCNNet(EnhancedGCNNet):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, num_layers=3, dropout=0.3):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)

class GATNet(EnhancedGATNet):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, num_layers=3, dropout=0.3):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)

class AdvancedHybridGNN(SuperiorHybridGNN):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, num_layers=3, 
                 dropout=0.3, fusion_method='medical_attention'):
        # Map old fusion methods to new enhanced ones
        fusion_mapping = {
            'advanced_attention': 'medical_attention',
            'cross_attention': 'clinical_weighting',
            'gated_fusion': 'adaptive_gating'
        }
        new_fusion_method = fusion_mapping.get(fusion_method, 'medical_attention')
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, new_fusion_method)

class HybridGNN(SuperiorHybridGNN):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, num_layers=3, 
                 dropout=0.3, fusion_method='attention'):
        # Map old fusion methods to new enhanced ones
        fusion_mapping = {
            'attention': 'medical_attention',
            'concat': 'adaptive_gating',
            'weighted': 'clinical_weighting'
        }
        new_fusion_method = fusion_mapping.get(fusion_method, 'medical_attention')
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, new_fusion_method)
