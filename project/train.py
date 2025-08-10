# train.py
import argparse
import joblib
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, roc_auc_score, confusion_matrix
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch_geometric.data import Data
from preprocess import df_to_pyg_data
from models import GCNNet, GATNet, HybridGNN, AdvancedHybridGNN, ImprovedGCNNet, ImprovedGATNet
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------- Enhanced Loss Functions ----------
class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean', alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            # Fix alpha weighting for multi-class
            alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(torch.nn.Module):
    def __init__(self, weight=None, focal_gamma=1.5, focal_alpha=0.25, ce_weight=0.3, focal_weight=0.7):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=weight)
        self.focal_loss = FocalLoss(weight=weight, gamma=focal_gamma, alpha=focal_alpha)
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
    
    def forward(self, logits, target):
        ce = self.ce_loss(logits, target)
        focal = self.focal_loss(logits, target)
        return self.ce_weight * ce + self.focal_weight * focal

def compute_enhanced_class_weights(y_tensor, method='balanced', minority_boost=2.0):
    """Enhanced class weight computation with minority class boosting"""
    y_np = y_tensor.cpu().numpy()
    
    if method == 'balanced':
        classes = np.unique(y_np)
        weights = compute_class_weight('balanced', classes=classes, y=y_np)
        weights = torch.tensor(weights, dtype=torch.float)
        
        # Boost minority class weight for medical diagnosis
        minority_class = np.argmin(np.bincount(y_np))
        weights[minority_class] *= minority_boost
        print(f"Applied {minority_boost}x boost to minority class {minority_class}")
        
        return weights
    elif method == 'inverse_freq':
        class_counts = np.bincount(y_np)
        total = len(y_np)
        weights = total / (len(class_counts) * class_counts)
        weights = torch.tensor(weights, dtype=torch.float)
        
        # Boost minority class
        minority_class = np.argmin(class_counts)
        weights[minority_class] *= minority_boost
        
        return weights
    else:  # smoothed
        class_counts = np.bincount(y_tensor.numpy())
        total = class_counts.sum()
        weights = total / (class_counts + 1e-6)  # Add small epsilon
        weights = weights / weights.sum() * len(weights)  # Normalize
        weights = torch.tensor(weights, dtype=torch.float)
        
        # Boost minority class
        minority_class = np.argmin(class_counts.numpy())
        weights[minority_class] *= minority_boost
        
        return weights

def evaluate_comprehensive(model, data, mask, focus_minority=True):
    """Comprehensive evaluation with focus on minority class performance"""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].argmax(dim=1)
        y_true = data.y[mask]
        
        # Basic metrics
        acc = accuracy_score(y_true.cpu(), pred.cpu())
        
        # Handle binary classification
        if len(torch.unique(y_true)) == 2:
            f1_macro = f1_score(y_true.cpu(), pred.cpu(), average='macro')
            f1_weighted = f1_score(y_true.cpu(), pred.cpu(), average='weighted')
            precision = precision_score(y_true.cpu(), pred.cpu(), average='weighted')
            recall = recall_score(y_true.cpu(), pred.cpu(), average='weighted')
            
            # Class-specific metrics
            f1_per_class = f1_score(y_true.cpu(), pred.cpu(), average=None)
            precision_per_class = precision_score(y_true.cpu(), pred.cpu(), average=None)
            recall_per_class = recall_score(y_true.cpu(), pred.cpu(), average=None)
            
            # ROC AUC
            probs = F.softmax(out[mask], dim=1)[:, 1]  # Probability of positive class
            try:
                auc = roc_auc_score(y_true.cpu(), probs.cpu())
            except:
                auc = 0.5
            
            # Focus on minority class (class 1 - Lassa positive)
            minority_f1 = f1_per_class[1] if len(f1_per_class) > 1 else 0
            minority_precision = precision_per_class[1] if len(precision_per_class) > 1 else 0
            minority_recall = recall_per_class[1] if len(recall_per_class) > 1 else 0
            
            return {
                'acc': acc,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'f1_class0': f1_per_class[0] if len(f1_per_class) > 0 else 0,
                'f1_class1': minority_f1,  # Lassa positive F1
                'precision': precision,
                'recall': recall,
                'minority_precision': minority_precision,  # Lassa precision
                'minority_recall': minority_recall,  # Lassa recall
                'auc': auc,
                'minority_f1': minority_f1  # Key metric for early stopping
            }
        else:
            # Multi-class
            f1_macro = f1_score(y_true.cpu(), pred.cpu(), average='macro')
            f1_weighted = f1_score(y_true.cpu(), pred.cpu(), average='weighted')
            precision = precision_score(y_true.cpu(), pred.cpu(), average='weighted')
            recall = recall_score(y_true.cpu(), pred.cpu(), average='weighted')
            
            return {
                'acc': acc,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'precision': precision,
                'recall': recall,
                'auc': 0.5,  # Not applicable for multi-class
                'minority_f1': f1_macro  # Use macro F1 as proxy
            }

# ---------- Training and Evaluation ----------
def train_one_epoch(model, optimizer, data, train_mask, loss_fn, use_edge_weights=False):
    model.train()
    optimizer.zero_grad()
    
    if use_edge_weights and hasattr(data, 'edge_weights'):
        # Note: Most PyG layers don't directly support edge weights
        # For now, we'll use standard forward pass
        out = model(data.x, data.edge_index)
    else:
        out = model(data.x, data.edge_index)
    
    loss = loss_fn(out[train_mask], data.y[train_mask])
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    return loss.item()

def enhanced_train(
    csv_path,
    model_out="model.pth",
    preprocess_out="preproc.pkl",
    k=12,
    epochs=150,
    lr=1e-3,
    patience=25,
    model_type='hybrid',
    fusion_method='attention',
    hidden_channels=128,
    num_layers=3,
    dropout=0.3,
    use_class_weights=True,
    loss_type='combined',
    use_scheduler=True,
    validate_split=True,
    minority_boost=3.0,
    use_curriculum_learning=True,
    label_smoothing=0.1,
    mixup_alpha=0.2
):
    """
    Enhanced training with advanced strategies for better accuracy
    """
    print("=== Enhanced GNN Training for Lassa Fever Diagnosis ===")
    print("Preparing data...")
    
    # Load and preprocess data with improvements
    parsed = df_to_pyg_data(csv_path, k=k)
    data = Data(
        x=torch.tensor(parsed["x"], dtype=torch.float),
        edge_index=parsed["edge_index"],
        edge_attr=torch.tensor(parsed["edge_weights"], dtype=torch.float) if parsed["edge_weights"] is not None else None,
        y=torch.tensor(parsed["y"], dtype=torch.long)
    ).to(DEVICE)
    
    print(f"Selected {len(parsed['features_used'])} features for model training")
    print(f"Dataset shape: {parsed['raw_df'].shape}")
    print(f"Using label column: {parsed['label_col']}")
    print(f"Selected features: {len(parsed['features_used'])}")
    
    # Enhanced data splitting with stratification
    num_nodes = data.x.shape[0]
    
    if validate_split:
        # Stratified split to maintain class balance
        indices = np.arange(num_nodes)
        y_numpy = data.y.cpu().numpy()
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=y_numpy
        )
        
        # Second split: train vs val
        y_train_val = y_numpy[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.2, random_state=42, stratify=y_train_val
        )
        
        # Create masks
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True
    else:
        # Simple random split
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        indices = torch.randperm(num_nodes)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        data.train_mask[indices[:train_size]] = True
        data.val_mask[indices[train_size:train_size + val_size]] = True
        data.test_mask[indices[train_size + val_size:]] = True
    
    print(f"Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")
    
    # Enhanced SMOTE oversampling
    if True:  # Always use SMOTE for better minority class representation
        print("Applying enhanced SMOTE oversampling...")
        X_train = data.x[data.train_mask].cpu().numpy()
        y_train = data.y[data.train_mask].cpu().numpy()
        
        # Use SMOTE with different strategies
        smote = SMOTE(
            sampling_strategy='minority',  # Only oversample minority class
            k_neighbors=min(5, len(y_train[y_train == 1]) - 1),  # Adaptive k
            random_state=42
        )
        
        try:
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"SMOTE: {len(X_train)} -> {len(X_resampled)} samples")
            print(f"Class distribution after SMOTE: {Counter(y_resampled)}")
            
            # Update training data
            resampled_size = len(X_resampled)
            
            # Extend the data tensor
            data.x = torch.cat([
                data.x,
                torch.tensor(X_resampled[len(X_train):], dtype=torch.float).to(DEVICE)
            ])
            data.y = torch.cat([
                data.y,
                torch.tensor(y_resampled[len(X_train):], dtype=torch.long).to(DEVICE)
            ])
            
            # Update train mask
            new_mask = torch.zeros(data.x.shape[0], dtype=torch.bool).to(DEVICE)
            new_mask[:len(data.train_mask)] = data.train_mask
            new_mask[len(data.train_mask):] = True  # New synthetic samples are training data
            data.train_mask = new_mask
            
            # Extend other masks
            val_mask_extended = torch.zeros(data.x.shape[0], dtype=torch.bool).to(DEVICE)
            val_mask_extended[:len(data.val_mask)] = data.val_mask
            data.val_mask = val_mask_extended
            
            test_mask_extended = torch.zeros(data.x.shape[0], dtype=torch.bool).to(DEVICE)
            test_mask_extended[:len(data.test_mask)] = data.test_mask
            data.test_mask = test_mask_extended
            
        except Exception as e:
            print(f"SMOTE failed: {e}, continuing without oversampling")
    
    # Model selection with improved architectures
    in_channels = data.x.shape[1]
    
    if model_type == 'hybrid':
        model = AdvancedHybridGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=num_layers,
            dropout=dropout,
            fusion_method='advanced_attention'
        ).to(DEVICE)
    elif model_type == 'gcn':
        model = ImprovedGCNNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=num_layers,
            dropout=dropout
        ).to(DEVICE)
    else:  # gat
        model = ImprovedGATNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=num_layers,
            dropout=dropout
        ).to(DEVICE)
    
    # Move model and data to device
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    
    # Enhanced class weights with minority boosting
    if use_class_weights:
        class_weights = compute_enhanced_class_weights(
            data.y[data.train_mask], 
            method='balanced',
            minority_boost=minority_boost
        )
        class_weights = class_weights.to(DEVICE)
        print(f"Enhanced class weights: {class_weights}")
    else:
        class_weights = None
    
    # Advanced loss function with label smoothing
    if loss_type == 'combined':
        if label_smoothing > 0:
            # Custom label smoothing cross entropy
            def label_smoothing_loss(logits, targets, smoothing=0.1):
                confidence = 1.0 - smoothing
                log_probs = F.log_softmax(logits, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                return (confidence * nll_loss + smoothing * smooth_loss).mean()
            
            def combined_loss_fn(logits, targets):
                ls_loss = label_smoothing_loss(logits, targets, label_smoothing)
                focal_loss = FocalLoss(weight=class_weights, gamma=2.0, reduction='mean')(logits, targets)
                return 0.7 * ls_loss.mean() + 0.3 * focal_loss
            loss_fn = combined_loss_fn
        else:
            loss_fn = CombinedLoss(weight=class_weights, focal_gamma=2.0, focal_alpha=0.25)
    elif loss_type == 'focal':
        loss_fn = FocalLoss(weight=class_weights, gamma=2.0, alpha=0.25)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Advanced optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=1e-4,  # L2 regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Enhanced learning rate scheduling
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=lr/100
        )
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
    
    # Training loop with curriculum learning
    best_minority_f1 = 0
    patience_counter = 0
    training_history = []
    
    # Curriculum learning: start with easier examples
    if use_curriculum_learning:
        print("Using curriculum learning...")
        # Sort training examples by difficulty (fewer symptoms = easier)
        train_indices = torch.where(data.train_mask)[0]
        train_difficulties = []
        
        for idx in train_indices:
            # Count number of positive symptoms as difficulty measure
            symptom_count = (data.x[idx] > 0).sum().item()
            train_difficulties.append((idx.item(), symptom_count))
        
        # Sort by difficulty (ascending)
        train_difficulties.sort(key=lambda x: x[1])
        curriculum_schedule = [0.3, 0.5, 0.7, 1.0]  # Fraction of data to use per phase
    
    for epoch in range(epochs):
        model.train()
        
        # Curriculum learning: gradually increase training data
        if use_curriculum_learning and epoch < len(curriculum_schedule) * 30:
            phase = min(epoch // 30, len(curriculum_schedule) - 1)
            data_fraction = curriculum_schedule[phase]
            num_samples = int(len(train_difficulties) * data_fraction)
            current_train_indices = [idx for idx, _ in train_difficulties[:num_samples]]
            
            # Create temporary mask
            temp_train_mask = torch.zeros_like(data.train_mask)
            temp_train_mask[current_train_indices] = True
        else:
            temp_train_mask = data.train_mask
        
        # Training step with mixup augmentation
        optimizer.zero_grad()
        
        if mixup_alpha > 0 and epoch > 10:  # Start mixup after initial epochs
            # Mixup data augmentation
            train_idx = torch.where(temp_train_mask)[0]
            if len(train_idx) > 1:
                indices = torch.randperm(len(train_idx))[:len(train_idx)//2*2]  # Even number
                idx1, idx2 = indices[:len(indices)//2], indices[len(indices)//2:]
                
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                mixed_x = lam * data.x[train_idx[idx1]] + (1 - lam) * data.x[train_idx[idx2]]
                mixed_y1, mixed_y2 = data.y[train_idx[idx1]], data.y[train_idx[idx2]]
                
                # Forward pass with mixed data - fix edge indexing
                # Create a subgraph for mixed samples
                mixed_indices = torch.cat([train_idx[idx1], train_idx[idx2]])
                # Use full edge index but only compute loss on mixed samples
                logits = model(data.x, data.edge_index)
                mixed_logits = logits[mixed_indices]
                loss1 = loss_fn(mixed_logits[:len(idx1)], mixed_y1)
                loss2 = loss_fn(mixed_logits[len(idx1):], mixed_y2)
                loss = lam * loss1 + (1 - lam) * loss2
            else:
                logits = model(data.x, data.edge_index)
                loss = loss_fn(logits[temp_train_mask], data.y[temp_train_mask])
        else:
            logits = model(data.x, data.edge_index)
            loss = loss_fn(logits[temp_train_mask], data.y[temp_train_mask])
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Validation
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                try:
                    val_logits = model(data.x, data.edge_index)
                    val_metrics = evaluate_comprehensive(model, data, data.val_mask, focus_minority=True)
                    current_minority_f1 = val_metrics.get('minority_f1', 0)
                except Exception as e:
                    print(f"Validation error: {e}")
                    current_minority_f1 = 0
                    val_metrics = {'minority_f1': 0, 'accuracy': 0}
                
                # Learning rate scheduling
                if use_scheduler:
                    scheduler.step()
                    plateau_scheduler.step(current_minority_f1)
                
                # Early stopping based on minority F1
                if current_minority_f1 > best_minority_f1:
                    best_minority_f1 = current_minority_f1
                    patience_counter = 0
                    
                    # Save best model
                    model_config = {
                        'model_type': model_type,
                        'in_channels': in_channels,
                        'hidden_channels': hidden_channels,
                        'out_channels': 2,
                        'num_layers': num_layers,
                        'dropout': dropout,
                        'fusion_method': fusion_method
                    }
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_minority_f1': best_minority_f1,
                        'model_config': model_config,
                        'training_history': training_history
                    }, model_out)
                    
                    # Save preprocessing info
                    joblib.dump({
                        'encoders': parsed['encoders'],
                        'scaler': parsed['scaler'],
                        'features_used': parsed['features_used']
                    }, preprocess_out)
                    
                else:
                    patience_counter += 1
                
                # Print progress
                print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val Minority F1: {current_minority_f1:.4f} | "
                      f"Val Acc: {val_metrics.get('accuracy', 0):.4f} | Val AUC: {val_metrics.get('auc', 0):.4f}")
                
                training_history.append({
                    'epoch': epoch,
                    'loss': loss.item(),
                    'val_minority_f1': current_minority_f1,
                    'val_accuracy': val_metrics.get('accuracy', 0),
                    'val_auc': val_metrics.get('auc', 0)
                })
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} (patience={patience})")
                    break
    
    print(f"\n=== Training Complete ===")
    print(f"Best Minority F1: {best_minority_f1:.4f}")
    
    # Load best model and evaluate on test set
    if os.path.exists(model_out):
        try:
            checkpoint = torch.load(model_out, map_location=DEVICE, weights_only=False)
            print(f" Found existing model: {model_out}")
            
            # Check if we can resume from this checkpoint
            if 'epoch' in checkpoint and 'model_config' in checkpoint:
                print(f" Loading best model from epoch {checkpoint['epoch']}")
            else:
                print(" Existing model format incompatible, starting fresh training")
                os.remove(model_out)
                if os.path.exists(preprocess_out):
                    os.remove(preprocess_out)
        except Exception as e:
            print(f" Error loading existing model: {e}")
            print(" Removing incompatible model files...")
            if os.path.exists(model_out):
                os.remove(model_out)
            if os.path.exists(preprocess_out):
                os.remove(preprocess_out)
    
    checkpoint = torch.load(model_out, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_comprehensive(model, data, data.test_mask)
    
    print(f"\n=== Final Test Results ===")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return model, test_metrics, training_history

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced GNN Training for Lassa Fever Diagnosis")
    parser.add_argument("--csv", type=str, default="./sample-data.csv", help="Path to CSV data file")
    parser.add_argument("--model_out", type=str, default="model.pth", help="Output model file")
    parser.add_argument("--preprocess_out", type=str, default="preproc.pkl", help="Output preprocessing file")
    parser.add_argument("--k", type=int, default=12, help="K for KNN graph construction")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--model_type", type=str, default="hybrid", choices=['gcn', 'gat', 'hybrid'], help="Model type")
    parser.add_argument("--fusion_method", type=str, default="attention", choices=['attention', 'concat', 'weighted_sum'], help="Fusion method for hybrid model")
    parser.add_argument("--hidden_channels", type=int, default=128, help="Hidden channels")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--loss_type", type=str, default="combined", choices=['ce', 'focal', 'combined'], help="Loss function type")
    parser.add_argument("--oversample_minority", action="store_true", help="Oversample minority class using SMOTE")
    parser.add_argument("--minority_boost", type=float, default=3.0, help="Minority class boosting factor")
    parser.add_argument("--use_curriculum_learning", action="store_true", help="Use curriculum learning")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha parameter")
    
    args = parser.parse_args()
    
    model, metrics, training_history = enhanced_train(
        csv_path=args.csv,
        model_out=args.model_out,
        preprocess_out=args.preprocess_out,
        k=args.k,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        model_type=args.model_type,
        fusion_method=args.fusion_method,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        loss_type=args.loss_type,
        minority_boost=args.minority_boost,
        use_curriculum_learning=args.use_curriculum_learning,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha
    )
