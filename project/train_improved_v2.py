#!/usr/bin/env python3
"""
Improved training script addressing critical performance issues
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
from preprocess import df_to_pyg_data, medical_feature_selection
from models import SuperiorHybridGNN, EnhancedGCNNet, EnhancedGATNet
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

class ImprovedTrainer:
    def __init__(self, csv_path, model_type='xgboost', use_sampling=True):
        self.csv_path = csv_path
        self.model_type = model_type
        self.use_sampling = use_sampling
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_and_preprocess_data(self):
        """Load data with improved preprocessing"""
        print("📊 Loading and preprocessing data...")
        
        # Load raw data
        df = pd.read_csv(self.csv_path)
        print(f"Dataset shape: {df.shape}")
        
        # Check class distribution
        target_col = 'InitialSampleFinalLaboratoryResultPathogentest'
        if target_col in df.columns:
            class_dist = df[target_col].value_counts()
            print(f"Class distribution: {class_dist}")
            print(f"Class ratio: {class_dist.iloc[0]/class_dist.iloc[1]:.2f}:1")
        
        return df
    
    def train_xgboost_baseline(self, df):
        """Train XGBoost baseline model (recommended approach)"""
        print("\n🚀 Training XGBoost Baseline Model")
        print("=" * 50)
        
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.model_selection import cross_val_score, GridSearchCV
        
        # Prepare features
        target_col = 'InitialSampleFinalLaboratoryResultPathogentest'
        
        # Select relevant clinical features
        clinical_features = [
            'age', 'sex', 'fever_new', 'headache_new', 'vomiting_new', 
            'bleeding_new', 'weakness_new', 'abdominal_pain', 'diarrhea_new',
            'sore_throat', 'cough_new', 'backache_new', 'chest_pain',
            'difficulty_breathing', 'fatigue_weakness', 'joint_pain_arthritis',
            'muscle_pain', 'nausea_new', 'contact_with_source_case_new',
            'direct_contact_probable_case', 'travelled_outside_district',
            'state_residence_new'
        ]
        
        # Filter available features
        available_features = [f for f in clinical_features if f in df.columns]
        print(f"Using {len(available_features)} clinical features")
        
        # Prepare data
        X = df[available_features].copy()
        y = df[target_col].copy()
        
        # Handle categorical variables
        le_dict = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Encode target
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Handle class imbalance with SMOTE + undersampling
        if self.use_sampling:
            print("🔄 Applying SMOTE + Random Undersampling...")
            
            # Create sampling pipeline
            over = SMOTE(sampling_strategy=0.5, random_state=42)  # Increase minority to 50% of majority
            under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # Reduce majority
            
            sampling_pipeline = ImbPipeline([
                ('over', over),
                ('under', under)
            ])
            
            X_train_resampled, y_train_resampled = sampling_pipeline.fit_resample(X_train, y_train)
            print(f"Original training set: {np.bincount(y_train)}")
            print(f"Resampled training set: {np.bincount(y_train_resampled)}")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        # Calculate balanced class weights
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_train_resampled), y=y_train_resampled
        )
        weight_dict = {i: w for i, w in enumerate(class_weights)}
        print(f"Class weights: {weight_dict}")
        
        # Train XGBoost with hyperparameter tuning
        print("🎯 Training XGBoost with hyperparameter tuning...")
        
        # Base model
        xgb_base = XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=class_weights[1]/class_weights[0]  # Handle imbalance
        )
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            xgb_base, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_resampled, y_train_resampled)
        
        # Best model
        best_xgb = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate on test set
        y_pred = best_xgb.predict(X_test)
        y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        test_acc = (y_pred == y_test).mean()
        test_f1 = classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n📊 XGBoost Test Results:")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"F1-Score: {test_f1:.4f}")
        print(f"AUC-ROC: {test_auc:.4f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le_target.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': best_xgb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Save model
        model_data = {
            'model': best_xgb,
            'label_encoders': le_dict,
            'target_encoder': le_target,
            'features': available_features,
            'performance': {
                'accuracy': test_acc,
                'f1_score': test_f1,
                'auc_roc': test_auc
            }
        }
        
        joblib.dump(model_data, 'xgboost_lassa_model.pkl')
        print(f"\n✅ Model saved to 'xgboost_lassa_model.pkl'")
        
        return model_data
    
    def train_improved_gnn(self, df):
        """Train improved GNN with better class handling"""
        print("\n🧠 Training Improved GNN Model")
        print("=" * 50)
        
        # Use smaller k for more meaningful connections
        parsed = df_to_pyg_data(self.csv_path, k=4)
        
        # Feature selection with clinical focus
        feature_indices, selected_features = medical_feature_selection(
            parsed["x"], parsed["y"], parsed["features_used"], top_k=40  # Reduce features
        )
        
        X = parsed["x"][:, feature_indices]
        y = parsed["y"]
        
        print(f"Selected {len(selected_features)} features")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Create PyG data
        data = Data(
            x=torch.tensor(X, dtype=torch.float),
            edge_index=parsed["edge_index"],
            y=torch.tensor(y, dtype=torch.long)
        ).to(self.device)
        
        # Split data
        num_nodes = data.x.shape[0]
        indices = np.arange(num_nodes)
        
        train_val_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=y
        )
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.2, random_state=42, 
            stratify=y[train_val_idx]
        )
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        # Calculate balanced class weights
        train_y = y[train_idx]
        class_weights = compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
        
        # Use more conservative weights
        weight_tensor = torch.tensor([
            class_weights[0] * 0.7,  # Reduce negative class weight
            class_weights[1] * 0.8   # Reduce positive class weight
        ], dtype=torch.float).to(self.device)
        
        print(f"Conservative class weights: {weight_tensor}")
        
        # Initialize model
        model = SuperiorHybridGNN(
            input_dim=X.shape[1],
            hidden_dim=64,  # Reduce complexity
            output_dim=2,
            num_heads=4,
            dropout=0.5  # Increase dropout
        ).to(self.device)
        
        # Use standard CrossEntropyLoss with class weights
        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Training loop
        best_val_f1 = 0
        patience = 20
        patience_counter = 0
        
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            
            logits = model(data.x, data.edge_index)
            loss = criterion(logits[train_mask], data.y[train_mask])
            
            loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_logits = model(data.x, data.edge_index)
                    val_pred = val_logits[val_mask].argmax(dim=1)
                    val_true = data.y[val_mask]
                    
                    val_acc = (val_pred == val_true).float().mean()
                    
                    # Calculate F1 score
                    val_pred_np = val_pred.cpu().numpy()
                    val_true_np = val_true.cpu().numpy()
                    
                    if len(np.unique(val_pred_np)) > 1:
                        val_report = classification_report(val_true_np, val_pred_np, output_dict=True)
                        val_f1 = val_report['macro avg']['f1-score']
                    else:
                        val_f1 = 0.0
                    
                    print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
                    
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        patience_counter = 0
                        # Save best model
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'features_used': selected_features,
                            'model_config': {
                                'input_dim': X.shape[1],
                                'hidden_dim': 64,
                                'output_dim': 2,
                                'num_heads': 4,
                                'dropout': 0.5
                            }
                        }, 'improved_gnn_model.pth')
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        # Load best model and evaluate
        checkpoint = torch.load('improved_gnn_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        with torch.no_grad():
            test_logits = model(data.x, data.edge_index)
            test_pred = test_logits[test_mask].argmax(dim=1)
            test_true = data.y[test_mask]
            
            test_acc = (test_pred == test_true).float().mean()
            
            test_pred_np = test_pred.cpu().numpy()
            test_true_np = test_true.cpu().numpy()
            
            print(f"\n📊 Improved GNN Test Results:")
            print(f"Accuracy: {test_acc:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(test_true_np, test_pred_np))
            print(f"\nConfusion Matrix:")
            print(confusion_matrix(test_true_np, test_pred_np))
        
        return model

def main():
    """Main training function"""
    csv_path = "lassa_fever_data.csv"  # Update with your data path
    
    trainer = ImprovedTrainer(csv_path)
    
    # Load data
    df = trainer.load_and_preprocess_data()
    
    print("\n" + "="*60)
    print("🎯 RECOMMENDATION: Start with XGBoost baseline")
    print("GNN approach has fundamental issues with artificial graph structure")
    print("="*60)
    
    # Train XGBoost baseline (recommended)
    print("\n1️⃣ Training XGBoost Baseline (RECOMMENDED)")
    xgb_results = trainer.train_xgboost_baseline(df)
    
    # Optionally train improved GNN
    response = input("\nDo you want to also train the improved GNN? (y/n): ")
    if response.lower() == 'y':
        print("\n2️⃣ Training Improved GNN")
        gnn_results = trainer.train_improved_gnn(df)
    
    print("\n✅ Training complete!")
    print("\n📋 RECOMMENDATIONS:")
    print("1. Use XGBoost model for production deployment")
    print("2. Focus on feature engineering and clinical validation")
    print("3. Collect more balanced training data")
    print("4. Consider ensemble methods combining multiple algorithms")

if __name__ == "__main__":
    main()
