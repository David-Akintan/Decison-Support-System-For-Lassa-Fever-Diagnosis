import argparse
import joblib
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch_geometric.data import Data
from preprocess import df_to_pyg_data
from models import GCNNet, GATNet, HybridGNN
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------- Training Visualizer Class ----------
class TrainingVisualizer:
    """Class to handle all training visualizations"""
    
    def __init__(self, save_dir="./training_plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        print(f"📊 Visualization plots will be saved to: {save_dir}")
        
    def plot_training_curves(self, training_history, save_name="training_curves.png"):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress Visualization', fontsize=16, fontweight='bold')
        
        epochs = [h['epoch'] for h in training_history]
        
        # Loss curve
        axes[0, 0].plot(epochs, [h['loss'] for h in training_history], 'b-', linewidth=2, label='Training Loss')
        axes[0, 0].set_title('Training Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Validation Accuracy
        axes[0, 1].plot(epochs, [h['val_acc'] for h in training_history], 'g-', linewidth=2, label='Validation Accuracy')
        axes[0, 1].set_title('Validation Accuracy', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # F1 Score
        axes[1, 0].plot(epochs, [h['val_f1'] for h in training_history], 'r-', linewidth=2, label='Validation F1')
        axes[1, 0].plot(epochs, [h['minority_f1'] for h in training_history], 'orange', linewidth=2, label='Minority F1')
        axes[1, 0].set_title('F1 Scores', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Training Summary
        best_epoch = np.argmax([h['minority_f1'] for h in training_history])
        axes[1, 1].text(0.5, 0.5, f'Best Performance at Epoch {best_epoch+1}\n'
                              f'Best Minority F1: {training_history[best_epoch]["minority_f1"]:.4f}\n'
                              f'Best Validation F1: {training_history[best_epoch]["val_f1"]:.4f}\n'
                              f'Best Validation Acc: {training_history[best_epoch]["val_acc"]:.4f}',
                              ha='center', va='center', transform=axes[1, 1].transAxes,
                              fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_title('Training Summary', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training curves saved to: {save_path}")
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred, save_name="confusion_matrix.png"):
        """Plot confusion matrix with heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Lassa Positive'],
                   yticklabels=['Negative', 'Lassa Positive'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add performance metrics as text
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        plt.text(0.02, 0.98, f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {save_path}")
        plt.show()
        
    def plot_roc_curve(self, y_true, y_probs, save_name="roc_curve.png"):
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 ROC curve saved to: {save_path}")
        plt.show()
        
    def plot_precision_recall_curve(self, y_true, y_probs, save_name="pr_curve.png"):
        """Plot Precision-Recall curve"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
        avg_precision = average_precision_score(y_true, y_probs[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Precision-Recall curve saved to: {save_path}")
        plt.show()
        
    def plot_feature_importance(self, feature_names, importance_scores, save_name="feature_importance.png"):
        """Plot feature importance scores"""
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_scores = importance_scores[sorted_indices]
        
        # Plot top 20 features
        top_n = min(20, len(sorted_features))
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(top_n), sorted_scores[:top_n], color='skyblue', edgecolor='navy')
        
        plt.yticks(range(top_n), sorted_features[:top_n])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Top Feature Importance Scores', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, sorted_scores[:top_n])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Feature importance plot saved to: {save_path}")
        plt.show()
        
    def plot_model_performance_summary(self, test_metrics, save_name="performance_summary.png"):
        """Plot comprehensive performance summary"""
        metrics = list(test_metrics.keys())
        values = list(test_metrics.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.title('Model Performance Summary', fontsize=16, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance summary saved to: {save_path}")
        plt.show()

# ---------- Enhanced Feature Engineering ----------
def advanced_feature_engineering(df, target_col):
    """Advanced feature engineering for medical data"""
    print("=== Advanced Feature Engineering ===")
    
    # Create feature interaction terms
    symptom_cols = [col for col in df.columns if any(term in col.lower() 
                   for term in ['fever', 'headache', 'vomit', 'cough', 'bleed', 'pain', 'weakness'])]
    
    if len(symptom_cols) >= 2:
        # Symptom severity score
        df['symptom_count'] = df[symptom_cols].sum(axis=1)
        df['symptom_severity'] = df[symptom_cols].mean(axis=1)
        
        # High-risk symptom combinations
        bleeding_cols = [col for col in symptom_cols if 'bleed' in col.lower()]
        fever_cols = [col for col in symptom_cols if 'fever' in col.lower()]
        
        if bleeding_cols:
            df['has_bleeding'] = df[bleeding_cols].max(axis=1)
        if fever_cols:
            df['has_fever'] = df[fever_cols].max(axis=1)
            
        # Critical combination: fever + bleeding
        if 'has_bleeding' in df.columns and 'has_fever' in df.columns:
            df['fever_bleeding_combo'] = df['has_fever'] * df['has_bleeding']
    
    # Age-related features
    age_cols = [col for col in df.columns if 'age' in col.lower()]
    if age_cols:
        age_col = age_cols[0]
        if df[age_col].dtype in ['int64', 'float64']:
            df['age_risk_group'] = pd.cut(df[age_col], bins=[0, 18, 45, 65, 100], 
                                        labels=['child', 'adult', 'middle', 'elderly'])
    
    # Geographic risk factors
    state_cols = [col for col in df.columns if 'state' in col.lower()]
    if state_cols:
        # Encode high-risk states based on Lassa fever epidemiology
        high_risk_states = ['EDO', 'ONDO', 'BAUCHI', 'TARABA', 'EBONYI']
        for state_col in state_cols:
            df[f'{state_col}_high_risk'] = df[state_col].astype(str).str.upper().isin(high_risk_states).astype(int)
    
    print(f"Feature engineering complete. New features added.")
    return df

def intelligent_feature_selection(X, y, feature_names, top_k=50):
    """Intelligent feature selection using multiple methods"""
    print("=== Intelligent Feature Selection ===")
    
    # Method 1: Statistical tests
    selector_stats = SelectKBest(score_func=f_classif, k=min(top_k, X.shape[1]))
    X_stats = selector_stats.fit_transform(X, y)
    stats_features = [feature_names[i] for i in selector_stats.get_support(indices=True)]
    
    # Method 2: Mutual information
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(top_k, X.shape[1]))
    X_mi = selector_mi.fit_transform(X, y)
    mi_features = [feature_names[i] for i in selector_mi.get_support(indices=True)]
    
    # Method 3: Random Forest feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    feature_importance = rf.feature_importances_
    rf_top_indices = np.argsort(feature_importance)[-top_k:]
    rf_features = [feature_names[i] for i in rf_top_indices]
    
    # Combine features from all methods
    combined_features = list(set(stats_features + mi_features + rf_features))
    feature_indices = [i for i, name in enumerate(feature_names) if name in combined_features]
    
    print(f"Selected {len(combined_features)} features from {len(feature_names)} total")
    print(f"Top 10 features by RF importance: {rf_features[-10:]}")
    
    return feature_indices, combined_features

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
        
        # Fix alpha weighting for multi-class
        if self.alpha is not None:
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

class ClinicalLoss(torch.nn.Module):
    """Medical-focused loss function"""
    def __init__(self, weight=None, false_negative_penalty=5.0, false_positive_penalty=1.0):
        super().__init__()
        self.weight = weight
        self.fn_penalty = false_negative_penalty
        self.fp_penalty = false_positive_penalty
        
    def forward(self, logits, target):
        # Base cross entropy
        ce_loss = F.cross_entropy(logits, target, weight=self.weight, reduction='none')
        
        # Get predictions
        pred = torch.argmax(logits, dim=1)
        
        # Penalty for false negatives (missing Lassa cases - dangerous!)
        fn_mask = (target == 1) & (pred == 0)
        fp_mask = (target == 0) & (pred == 1)
        
        # Apply penalties
        penalty = torch.ones_like(ce_loss)
        penalty[fn_mask] *= self.fn_penalty
        penalty[fp_mask] *= self.fp_penalty
        
        return (ce_loss * penalty).mean()

# ---------- Enhanced Training Function ----------
def enhanced_train_v2(
    csv_path,
    model_out="model.pth",
    preprocess_out="preproc.pkl",
    k=8,  # Reduced k for better generalization
    epochs=200,
    lr=5e-4,  # Lower learning rate
    patience=40,
    model_type='hybrid',
    hidden_channels=64,  # Reduced to prevent overfitting
    num_layers=2,  # Simpler architecture
    dropout=0.5,  # Higher dropout
    use_advanced_features=True,
    use_feature_selection=True,
    use_clinical_loss=True,
    use_ensemble_sampling=True,
    minority_boost=2.0,  # Reduced boost
    validation_split=0.25,  # Larger validation set
    test_split=0.2
):
    """Enhanced training with comprehensive improvements and visualization"""
    print("=== Enhanced GNN Training v2 with Visualization for Lassa Fever Diagnosis ===")
    print("Implementing advanced improvements and comprehensive visualization...")
    
    # Initialize visualizer
    visualizer = TrainingVisualizer()
    
    # Load raw data for feature engineering
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape}")
    
    # Advanced feature engineering
    if use_advanced_features:
        df = advanced_feature_engineering(df, 'InitialSampleFinalLaboratoryResultPathogentest')
    
    # Enhanced preprocessing
    parsed = df_to_pyg_data(csv_path, k=k)
    
    # Intelligent feature selection
    if use_feature_selection:
        feature_indices, selected_features = intelligent_feature_selection(
            parsed["x"], parsed["y"], parsed["features_used"], top_k=30
        )
        # Update features
        parsed["x"] = parsed["x"][:, feature_indices]
        parsed["features_used"] = selected_features
    
    # Create PyG data
    data = Data(
        x=torch.tensor(parsed["x"], dtype=torch.float),
        edge_index=parsed["edge_index"],
        y=torch.tensor(parsed["y"], dtype=torch.long)
    ).to(DEVICE)
    
    print(f"Final feature count: {data.x.shape[1]}")
    print(f"Dataset: {data.x.shape[0]} samples")
    
    # Enhanced data splitting
    num_nodes = data.x.shape[0]
    indices = np.arange(num_nodes)
    y_numpy = data.y.cpu().numpy()
    
    # Three-way stratified split
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_split, random_state=42, stratify=y_numpy
    )
    y_train_val = y_numpy[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=validation_split, random_state=42, stratify=y_train_val
    )
    
    # Create masks
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    
    print(f"Split - Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")
    
    # Enhanced sampling strategy
    if use_ensemble_sampling:
        print("Applying ensemble sampling strategy...")
        X_train = data.x[data.train_mask].cpu().numpy()
        y_train = data.y[data.train_mask].cpu().numpy()
        
        # Combine SMOTE with edited nearest neighbors
        smote_enn = SMOTEENN(
            smote=SMOTE(sampling_strategy='minority', k_neighbors=3, random_state=42),
            enn=EditedNearestNeighbours(sampling_strategy='majority', n_neighbors=3),
            random_state=42
        )
        
        try:
            X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
            print(f"Ensemble sampling: {len(X_train)} -> {len(X_resampled)} samples")
            print(f"Class distribution: {Counter(y_resampled)}")
            
            # Update data
            data.x = torch.cat([
                data.x,
                torch.tensor(X_resampled[len(X_train):], dtype=torch.float).to(DEVICE)
            ])
            data.y = torch.cat([
                data.y,
                torch.tensor(y_resampled[len(X_train):], dtype=torch.long).to(DEVICE)
            ])
            
            # Update masks
            new_mask = torch.zeros(data.x.shape[0], dtype=torch.bool).to(DEVICE)
            new_mask[:len(data.train_mask)] = data.train_mask
            new_mask[len(data.train_mask):] = True
            data.train_mask = new_mask
            
            # Extend validation and test masks
            for mask_name in ['val_mask', 'test_mask']:
                old_mask = getattr(data, mask_name)
                new_mask = torch.zeros(data.x.shape[0], dtype=torch.bool).to(DEVICE)
                new_mask[:len(old_mask)] = old_mask
                setattr(data, mask_name, new_mask)
                
        except Exception as e:
            print(f"Ensemble sampling failed: {e}, using original data")
    
    # Optimized model architecture
    in_channels = data.x.shape[1]
    
    if model_type == 'hybrid':
        model = HybridGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=num_layers,
            dropout=dropout,
            fusion_method='gated_fusion'  # More stable than attention
        ).to(DEVICE)
    elif model_type == 'gcn':
        model = GCNNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=num_layers,
            dropout=dropout
        ).to(DEVICE)
    else:
        model = GATNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=num_layers,
            dropout=dropout
        ).to(DEVICE)
    
    # Enhanced class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(data.y[data.train_mask].cpu().numpy()),
        y=data.y[data.train_mask].cpu().numpy()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    class_weights[1] *= minority_boost  # Boost minority class
    print(f"Enhanced class weights: {class_weights}")
    
    # Medical-focused loss function
    if use_clinical_loss:
        loss_fn = ClinicalLoss(
            weight=class_weights,
            false_negative_penalty=3.0,  # Missing Lassa is dangerous
            false_positive_penalty=1.0
        )
    else:
        loss_fn = FocalLoss(weight=class_weights, gamma=2.0, alpha=0.25)
    
    # Optimized optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-3,  # Stronger regularization
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduling
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=15, min_lr=1e-6
    )
    
    # Training loop
    best_f1 = 0
    patience_counter = 0
    training_history = []
    
    print("Starting enhanced training with visualization...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        logits = model(data.x, data.edge_index)
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(data.x, data.edge_index)
                val_pred = torch.argmax(val_logits[data.val_mask], dim=1)
                val_true = data.y[data.val_mask]
                
                # Calculate metrics
                val_acc = (val_pred == val_true).float().mean().item()
                val_f1 = f1_score(val_true.cpu(), val_pred.cpu(), average='macro')
                
                # Focus on minority class performance
                minority_f1 = f1_score(val_true.cpu(), val_pred.cpu(), pos_label=1, average='binary')
                
                scheduler.step(minority_f1)
                
                print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"Val F1: {val_f1:.4f} | Minority F1: {minority_f1:.4f}")
                
                # Early stopping on minority F1
                if minority_f1 > best_f1:
                    best_f1 = minority_f1
                    patience_counter = 0
                    
                    # Save best model
                    model_config = {
                        'model_type': model_type,
                        'in_channels': in_channels,
                        'hidden_channels': hidden_channels,
                        'out_channels': 2,
                        'num_layers': num_layers,
                        'dropout': dropout,
                        'fusion_method': 'gated_fusion'
                    }
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_f1': best_f1,
                        'model_config': model_config,
                        'training_history': training_history
                    }, model_out)
                    
                    # Save preprocessing
                    joblib.dump({
                        'encoders': parsed['encoders'],
                        'scaler': parsed['scaler'],
                        'features_used': parsed['features_used'],
                        'feature_indices': feature_indices if use_feature_selection else None
                    }, preprocess_out)
                    
                else:
                    patience_counter += 1
                
                training_history.append({
                    'epoch': epoch,
                    'loss': loss.item(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'minority_f1': minority_f1
                })
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    print(f"\n=== Enhanced Training Complete ===")
    print(f"Best Minority F1: {best_f1:.4f}")
    
    # Final evaluation with comprehensive visualization
    if os.path.exists(model_out):
        checkpoint = torch.load(model_out, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        with torch.no_grad():
            test_logits = model(data.x, data.edge_index)
            test_pred = torch.argmax(test_logits[data.test_mask], dim=1)
            test_true = data.y[data.test_mask]
            test_probs = torch.softmax(test_logits[data.test_mask], dim=1)
            
            test_acc = (test_pred == test_true).float().mean().item()
            test_f1 = f1_score(test_true.cpu(), test_pred.cpu(), average='macro')
            test_minority_f1 = f1_score(test_true.cpu(), test_pred.cpu(), pos_label=1, average='binary')
            test_precision = precision_score(test_true.cpu(), test_pred.cpu(), average='macro')
            test_recall = recall_score(test_true.cpu(), test_pred.cpu(), average='macro')
            
            print(f"\n=== Final Test Results ===")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test F1 (Macro): {test_f1:.4f}")
            print(f"Test Minority F1: {test_minority_f1:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test Recall: {test_recall:.4f}")
            
            # Detailed classification report
            print("\nDetailed Classification Report:")
            print(classification_report(test_true.cpu(), test_pred.cpu(), 
                                      target_names=['Negative', 'Lassa Positive']))
            
            # Generate comprehensive visualizations
            print("\n📊 Generating comprehensive visualizations...")
            
            # 1. Training curves
            visualizer.plot_training_curves(training_history)
            
            # 2. Confusion matrix
            visualizer.plot_confusion_matrix(test_true.cpu().numpy(), test_pred.cpu().numpy())
            
            # 3. ROC curve
            visualizer.plot_roc_curve(test_true.cpu().numpy(), test_probs.cpu().numpy())
            
            # 4. Precision-Recall curve
            visualizer.plot_precision_recall_curve(test_true.cpu().numpy(), test_probs.cpu().numpy())
            
            # 5. Performance summary
            performance_metrics = {
                'Accuracy': test_acc,
                'F1 Score': test_f1,
                'Precision': test_precision,
                'Recall': test_recall
            }
            visualizer.plot_model_performance_summary(performance_metrics)
            
            # 6. Feature importance (if available)
            if hasattr(model, 'feature_importance'):
                try:
                    with torch.no_grad():
                        feature_imp = model.feature_importance(data.x, data.edge_index, data.edge_attr)
                        feature_imp = torch.sigmoid(feature_imp).mean(dim=0).cpu().numpy()
                        feature_names = parsed.get('features_used', [f'Feature_{i}' for i in range(len(feature_imp))])
                        visualizer.plot_feature_importance(feature_names, feature_imp)
                except Exception as e:
                    print(f"Feature importance visualization skipped: {e}")
            
            print(f"\n📈 All visualizations completed and saved!")
            print(f"📂 Check the '{visualizer.save_dir}' directory for all plots")
    
    return model, {
        'test_acc': test_acc, 
        'test_f1': test_f1, 
        'minority_f1': test_minority_f1,
        'test_precision': test_precision,
        'test_recall': test_recall
    }, training_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced GNN Training v2")
    parser.add_argument("--csv", type=str, default="./sample-data.csv")
    parser.add_argument("--model_out", type=str, default="model.pth")
    parser.add_argument("--preprocess_out", type=str, default="preproc.pkl")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--model_type", type=str, default="hybrid", choices=["gcn", "gat", "hybrid"])
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=8)
    
    args = parser.parse_args()
    
    model, metrics, history = enhanced_train_v2(
        csv_path=args.csv,
        model_out=args.model_out,
        preprocess_out=args.preprocess_out,
        epochs=args.epochs,
        lr=args.lr,
        model_type=args.model_type,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        k=args.k
    )
    
    print(f"\n🎯 Enhanced training complete!")
    print(f"📊 Final Performance: {metrics}")
