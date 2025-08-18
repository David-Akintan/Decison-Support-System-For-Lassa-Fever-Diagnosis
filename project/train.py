# train.py - Enhanced training with comprehensive improvements
import argparse
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch_geometric.data import Data
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from preprocess import df_to_pyg_data
from models import GCNNet, GATNet, HybridGNN, AdvancedHybridGNN, SuperiorHybridGNN, EnhancedGCNNet, EnhancedGATNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

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

def medical_domain_feature_engineering(df, target_col):
    """Medical domain-specific feature engineering for Lassa fever"""
    print("=== Medical Domain Feature Engineering ===")
    
    # Clinical symptom severity scoring
    symptom_cols = [col for col in df.columns if any(term in col.lower() 
                   for term in ['fever', 'headache', 'vomit', 'cough', 'bleed', 'pain', 'weakness', 
                               'diarrhea', 'fatigue', 'nausea', 'chills', 'muscle', 'joint'])]
    
    if len(symptom_cols) >= 3:
        # Create comprehensive symptom profiles
        df['total_symptoms'] = df[symptom_cols].sum(axis=1)
        df['symptom_severity_score'] = df[symptom_cols].mean(axis=1)
        
        # Critical Lassa fever symptom combinations
        bleeding_symptoms = [col for col in symptom_cols if any(bleed in col.lower() 
                           for bleed in ['bleed', 'hemorrhage', 'blood'])]
        fever_symptoms = [col for col in symptom_cols if 'fever' in col.lower()]
        neurological_symptoms = [col for col in symptom_cols if any(neuro in col.lower() 
                               for neuro in ['headache', 'confusion', 'consciousness'])]
        
        if bleeding_symptoms:
            df['hemorrhagic_score'] = df[bleeding_symptoms].sum(axis=1)
            df['has_bleeding'] = (df[bleeding_symptoms].sum(axis=1) > 0).astype(int)
            
        if fever_symptoms:
            df['fever_intensity'] = df[fever_symptoms].max(axis=1)
            df['has_fever'] = (df[fever_symptoms].sum(axis=1) > 0).astype(int)
            
        if neurological_symptoms:
            df['neurological_score'] = df[neurological_symptoms].sum(axis=1)
            
        # Lassa fever signature: fever + bleeding + neurological
        if all(col in df.columns for col in ['has_fever', 'has_bleeding']):
            df['lassa_signature'] = df['has_fever'] * df['has_bleeding']
            if 'neurological_score' in df.columns:
                df['lassa_signature'] += (df['neurological_score'] > 0).astype(int)
    
    # Geographic risk stratification (Lassa fever endemic zones)
    state_cols = [col for col in df.columns if 'state' in col.lower()]
    if state_cols:
        # High-risk states based on Lassa fever epidemiology
        high_risk_states = ['EDO', 'ONDO', 'BAUCHI', 'TARABA', 'EBONYI', 'KOGI', 'PLATEAU', 'RIVERS']
        moderate_risk_states = ['OGUN', 'OSUN', 'KWARA', 'NIGER', 'KADUNA', 'NASARAWA']
        
        for state_col in state_cols:
            df[f'{state_col}_risk_level'] = df[state_col].astype(str).str.upper().map({
                **{state: 3 for state in high_risk_states},
                **{state: 2 for state in moderate_risk_states}
            }).fillna(1)  # Low risk for other states
    
    # Contact tracing and exposure risk
    contact_cols = ['contact_with_source_case_new', 'direct_contact_probable_case', 'rodents_excreta']
    exposure_risk = 0
    for col in contact_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            exposure_risk += df[col]
    
    if exposure_risk.sum() > 0:
        df['exposure_risk_score'] = exposure_risk
        df['high_exposure_risk'] = (exposure_risk >= 2).astype(int)
    
    # Age-based risk stratification for Lassa fever
    age_cols = [col for col in df.columns if 'age' in col.lower()]
    if age_cols:
        age_col = age_cols[0]
        if df[age_col].dtype in ['int64', 'float64']:
            # Lassa fever age risk categories
            df['age_risk_category'] = pd.cut(df[age_col], 
                                           bins=[0, 5, 15, 30, 50, 65, 100], 
                                           labels=['infant', 'child', 'young_adult', 'adult', 'middle_aged', 'elderly'])
            
            # High-risk age groups (very young, elderly, pregnant women)
            df['age_high_risk'] = ((df[age_col] < 5) | (df[age_col] > 60)).astype(int)
    
    # Pregnancy risk factor
    pregnancy_cols = [col for col in df.columns if 'pregnan' in col.lower()]
    if pregnancy_cols:
        for preg_col in pregnancy_cols:
            df[f'{preg_col}_risk'] = df[preg_col].astype(str).str.upper().isin(['YES', 'POSITIVE', '1']).astype(int)
    
    # Temporal clustering (outbreak patterns)
    date_cols = [col for col in df.columns if any(date_term in col.lower() 
                for date_term in ['date', 'onset', 'report'])]
    
    for date_col in date_cols:
        if date_col in df.columns:
            try:
                df[date_col] = pd.to_numeric(df[date_col], errors='coerce')
                if not df[date_col].isnull().all():
                    # Create temporal features for outbreak detection
                    df[f'{date_col}_month'] = (df[date_col] / (30 * 24 * 3600)).astype(int) % 12
                    # Lassa fever seasonal pattern (dry season: Nov-Apr)
                    df[f'{date_col}_dry_season'] = df[f'{date_col}_month'].isin([11, 0, 1, 2, 3]).astype(int)
            except:
                pass
    
    # Clinical severity index
    severity_indicators = []
    for col in df.columns:
        if any(severe in col.lower() for severe in ['blood', 'bleed', 'unconscious', 'shock', 'death']):
            severity_indicators.append(col)
    
    if severity_indicators:
        df['clinical_severity_index'] = df[severity_indicators].sum(axis=1)
        df['critical_condition'] = (df['clinical_severity_index'] >= 2).astype(int)
    
    print(f"Medical feature engineering complete. Added domain-specific features.")
    return df

def medical_feature_selection(X, y, feature_names, top_k=40):
    """Medical domain-focused feature selection"""
    print("=== Medical Feature Selection ===")
    
    # Method 1: Medical relevance scoring
    medical_keywords = ['fever', 'bleed', 'hemorrhage', 'headache', 'contact', 'exposure', 
                       'state', 'age', 'symptom', 'clinical', 'risk', 'lassa']
    medical_scores = []
    
    for feature in feature_names:
        score = sum(1 for keyword in medical_keywords if keyword in feature.lower())
        medical_scores.append(score)
    
    # Method 2: Statistical significance
    selector_stats = SelectKBest(score_func=f_classif, k=min(top_k, X.shape[1]))
    X_stats = selector_stats.fit_transform(X, y)
    stats_features = [feature_names[i] for i in selector_stats.get_support(indices=True)]
    
    # Method 3: Mutual information for medical relevance
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(top_k, X.shape[1]))
    X_mi = selector_mi.fit_transform(X, y)
    mi_features = [feature_names[i] for i in selector_mi.get_support(indices=True)]
    
    # Method 4: Medical Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced',
                               max_depth=10, min_samples_split=5)
    rf.fit(X, y)
    feature_importance = rf.feature_importances_
    
    # Combine medical relevance with statistical importance
    combined_scores = []
    for i, (feature, med_score) in enumerate(zip(feature_names, medical_scores)):
        stat_score = selector_stats.scores_[i] if i < len(selector_stats.scores_) else 0
        mi_score = selector_mi.scores_[i] if i < len(selector_mi.scores_) else 0
        rf_score = feature_importance[i]
        
        # Weighted combination favoring medical relevance
        combined_score = (0.3 * med_score + 0.25 * stat_score + 0.25 * mi_score + 0.2 * rf_score)
        combined_scores.append((i, feature, combined_score))
    
    # Select top features
    combined_scores.sort(key=lambda x: x[2], reverse=True)
    selected_indices = [item[0] for item in combined_scores[:top_k]]
    selected_features = [item[1] for item in combined_scores[:top_k]]
    
    print(f"Selected {len(selected_features)} medically relevant features from {len(feature_names)} total")
    print(f"Top 10 medical features: {[item[1] for item in combined_scores[:10]]}")
    
    return selected_indices, selected_features

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
def enhanced_train_v3(
    csv_path,
    model_out="model.pth",
    preprocess_out="preproc.pkl",
    k=8,
    epochs=300,  # Increased for better convergence
    lr=5e-5,  # Lower learning rate for stability
    patience=50,  # Increased patience
    model_type='hybrid',
    hidden_channels=96,  # Slightly increased
    num_layers=3,  # Optimal depth
    dropout=0.4,  # Balanced dropout
    use_advanced_features=True,
    use_feature_selection=True,
    use_clinical_loss=True,
    use_ensemble_sampling=True,
    minority_boost=1.5,  # Reduced boost
    validation_split=0.2,
    test_split=0.2
):
    """Enhanced training v3 with medical domain optimizations"""
    print("=== Enhanced GNN Training v3 for Lassa Fever Diagnosis ===")
    print("Implementing medical domain optimizations...")
    
    # Load raw data for feature engineering
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape}")
    
    # Enhanced medical feature engineering
    if use_advanced_features:
        df = medical_domain_feature_engineering(df, 'InitialSampleFinalLaboratoryResultPathogentest')
    
    # Enhanced preprocessing with medical graph
    parsed = df_to_pyg_data(csv_path, k=k)
    
    # Medical-focused feature selection - keep more features
    if use_feature_selection:
        feature_indices, selected_features = medical_feature_selection(
            parsed["x"], parsed["y"], parsed["features_used"], top_k=80
        )
        parsed["x"] = parsed["x"][:, feature_indices]
        parsed["features_used"] = selected_features
    
    # Create enhanced PyG data with edge weights
    data = Data(
        x=torch.tensor(parsed["x"], dtype=torch.float),
        edge_index=parsed["edge_index"],
        edge_attr=torch.tensor(parsed["edge_weights"], dtype=torch.float) if parsed["edge_weights"] is not None else None,
        y=torch.tensor(parsed["y"], dtype=torch.long)
    ).to(DEVICE)
    
    print(f"Final feature count: {data.x.shape[1]}")
    print(f"Dataset: {data.x.shape[0]} samples")
    print(f"Graph edges: {data.edge_index.shape[1]}")
    
    # Enhanced data splitting with stratification
    num_nodes = data.x.shape[0]
    indices = np.arange(num_nodes)
    y_numpy = data.y.cpu().numpy()
    
    # Stratified three-way split
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_split, random_state=42, stratify=y_numpy
    )
    y_train_val = y_numpy[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=validation_split/(1-test_split), random_state=42, stratify=y_train_val
    )
    
    # Create masks
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    
    print(f"Split - Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")
    
    # Medical-focused sampling strategy
    if use_ensemble_sampling:
        print("Applying medical-focused sampling strategy...")
        X_train = data.x[data.train_mask].cpu().numpy()
        y_train = data.y[data.train_mask].cpu().numpy()
        
        # Use ADASYN for better minority class synthesis
        try:
            adasyn = ADASYN(sampling_strategy='minority', n_neighbors=5, random_state=42)
            X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
            print(f"ADASYN sampling: {len(X_train)} -> {len(X_resampled)} samples")
            print(f"Class distribution: {Counter(y_resampled)}")
            
            # Update data with resampled training set
            additional_samples = len(X_resampled) - len(X_train)
            if additional_samples > 0:
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
                
                # Extend other masks
                for mask_name in ['val_mask', 'test_mask']:
                    old_mask = getattr(data, mask_name)
                    new_mask = torch.zeros(data.x.shape[0], dtype=torch.bool).to(DEVICE)
                    new_mask[:len(old_mask)] = old_mask
                    setattr(data, mask_name, new_mask)
                    
        except Exception as e:
            print(f"ADASYN sampling failed: {e}, using original data")
    
    # Enhanced model architecture
    in_channels = data.x.shape[1]
    
    if model_type == 'hybrid':
        model = SuperiorHybridGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=num_layers,
            dropout=dropout,
            fusion_method='medical_attention'
        ).to(DEVICE)
    elif model_type == 'gcn':
        model = EnhancedGCNNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=num_layers,
            dropout=dropout
        ).to(DEVICE)
    else:
        model = EnhancedGATNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=num_layers,
            dropout=dropout
        ).to(DEVICE)
    
    # Enhanced medical class weights for better precision-recall balance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # Adjust weights to reduce false positives while maintaining sensitivity
    class_weights[0] = class_weights[0] * 2.0  # Increase negative class weight
    class_weights[1] = class_weights[1] * 1.2  # Slightly increase positive class weight
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"Enhanced medical class weights: {class_weights}")
    
    # Enhanced loss function
    if use_clinical_loss:
        loss_fn = ClinicalLoss(
            weight=class_weights,
            false_negative_penalty=4.0,  # Higher penalty for missing Lassa cases
            false_positive_penalty=1.0
        )
    else:
        loss_fn = FocalLoss(weight=class_weights, gamma=2.5, alpha=0.3)
    
    # Advanced optimizer with medical focus
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4,  # Moderate regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Enhanced learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=20, min_lr=1e-7
    )
    
    # Training loop with medical monitoring
    best_medical_score = 0
    patience_counter = 0
    training_history = []
    
    print("Starting enhanced medical training...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with edge weights
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            logits = model(data.x, data.edge_index, data.edge_attr)
        else:
            logits = model(data.x, data.edge_index)
            
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation every 3 epochs
        if epoch % 3 == 0:
            model.eval()
            with torch.no_grad():
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    val_logits = model(data.x, data.edge_index, data.edge_attr)
                else:
                    val_logits = model(data.x, data.edge_index)
                    
                val_pred = torch.argmax(val_logits[data.val_mask], dim=1)
                val_true = data.y[data.val_mask]
                
                # Medical-focused metrics
                val_acc = (val_pred == val_true).float().mean().item()
                val_f1_macro = f1_score(val_true.cpu(), val_pred.cpu(), average='macro')
                val_f1_positive = f1_score(val_true.cpu(), val_pred.cpu(), pos_label=1, average='binary')
                val_precision = precision_score(val_true.cpu(), val_pred.cpu(), pos_label=1, zero_division=0)
                val_recall = recall_score(val_true.cpu(), val_pred.cpu(), pos_label=1, zero_division=0)
                
                # Medical score combines recall (sensitivity) and precision
                medical_score = 0.6 * val_recall + 0.4 * val_precision
                
                scheduler.step(medical_score)
                
                print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Acc: {val_acc:.4f} | "
                      f"F1: {val_f1_macro:.4f} | Pos F1: {val_f1_positive:.4f} | "
                      f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | "
                      f"Medical Score: {medical_score:.4f}")
                
                # Early stopping based on medical score
                if medical_score > best_medical_score:
                    best_medical_score = medical_score
                    patience_counter = 0
                    
                    # Save best model
                    model_config = {
                        'model_type': model_type,
                        'in_channels': in_channels,
                        'hidden_channels': hidden_channels,
                        'out_channels': 2,
                        'num_layers': num_layers,
                        'dropout': dropout,
                        'fusion_method': 'medical_attention'
                    }
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_medical_score': best_medical_score,
                        'model_config': model_config,
                        'training_history': training_history
                    }, model_out)
                    
                    # Save preprocessing
                    joblib.dump({
                        'encoders': parsed['encoders'],
                        'scaler': parsed['scaler'],
                        'features_used': parsed['features_used'],
                        'feature_indices': feature_indices if use_feature_selection else None,
                        'variance_selector': parsed.get('variance_selector', None)
                    }, preprocess_out)
                    
                else:
                    patience_counter += 1
                
                training_history.append({
                    'epoch': epoch,
                    'loss': loss.item(),
                    'val_acc': val_acc,
                    'val_f1_macro': val_f1_macro,
                    'val_f1_positive': val_f1_positive,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'medical_score': medical_score
                })
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    print(f"\n=== Enhanced Medical Training Complete ===")
    print(f"Best Medical Score: {best_medical_score:.4f}")
    
    # Final comprehensive evaluation
    if os.path.exists(model_out):
        checkpoint = torch.load(model_out, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        with torch.no_grad():
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                test_logits = model(data.x, data.edge_index, data.edge_attr)
            else:
                test_logits = model(data.x, data.edge_index)
                
            test_pred = torch.argmax(test_logits[data.test_mask], dim=1)
            test_true = data.y[data.test_mask]
            
            # Comprehensive medical metrics
            test_acc = (test_pred == test_true).float().mean().item()
            test_f1_macro = f1_score(test_true.cpu(), test_pred.cpu(), average='macro')
            test_f1_positive = f1_score(test_true.cpu(), test_pred.cpu(), pos_label=1, average='binary')
            test_precision = precision_score(test_true.cpu(), test_pred.cpu(), pos_label=1, zero_division=0)
            test_recall = recall_score(test_true.cpu(), test_pred.cpu(), pos_label=1, zero_division=0)
            
            print(f"\n=== Final Medical Test Results ===")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test F1 (Macro): {test_f1_macro:.4f}")
            print(f"Test F1 (Lassa Positive): {test_f1_positive:.4f}")
            print(f"Test Precision (Sensitivity): {test_precision:.4f}")
            print(f"Test Recall (Specificity): {test_recall:.4f}")
            
            # Medical interpretation
            if test_recall > 0.85:
                print("✅ Excellent sensitivity - low risk of missing Lassa cases")
            elif test_recall > 0.75:
                print("⚠️  Good sensitivity - acceptable risk of missing cases")
            else:
                print("❌ Low sensitivity - high risk of missing Lassa cases")
            
            # Detailed classification report
            print("\nDetailed Medical Classification Report:")
            print(classification_report(test_true.cpu(), test_pred.cpu(), 
                                      target_names=['Negative', 'Lassa Positive']))
            
            # Confusion matrix analysis
            cm = confusion_matrix(test_true.cpu(), test_pred.cpu())
            print(f"\nConfusion Matrix:")
            print(f"True Negative: {cm[0,0]}, False Positive: {cm[0,1]}")
            print(f"False Negative: {cm[1,0]}, True Positive: {cm[1,1]}")
            
            if cm[1,0] > 0:
                print(f"⚠️  {cm[1,0]} Lassa cases were missed (False Negatives)")
    
    return model, {
        'test_acc': test_acc, 
        'test_f1_macro': test_f1_macro, 
        'test_f1_positive': test_f1_positive,
        'test_precision': test_precision,
        'test_recall': test_recall
    }, training_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced GNN Training v3")
    parser.add_argument("--csv", type=str, default="./sample-data.csv")
    parser.add_argument("--model_out", type=str, default="model.pth")
    parser.add_argument("--preprocess_out", type=str, default="preproc.pkl")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--model_type", type=str, default="hybrid", choices=["gcn", "gat", "hybrid"])
    parser.add_argument("--hidden_channels", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--k", type=int, default=8)
    
    args = parser.parse_args()
    
    model, metrics, history = enhanced_train_v3(
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
