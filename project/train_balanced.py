#!/usr/bin/env python3
"""
Improved training script addressing class imbalance for >95% accuracy
Uses XGBoost with proper sampling techniques and cost-sensitive learning
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BalancedLassaFeverTrainer:
    def __init__(self, csv_path='sample-data.csv'):
        self.csv_path = csv_path
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.features_used = []
        self.class_weights = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess data with enhanced feature engineering"""
        print("📊 Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(self.csv_path)
        print(f"Original dataset shape: {df.shape}")
        
        # Target variable
        target_col = 'InitialSampleFinalLaboratoryResultPathogentest'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Map target to binary
        df['target'] = df[target_col].map({'POSITIVE': 1, 'NEGATIVE': 0})
        df = df.dropna(subset=['target'])
        
        # Check class distribution
        class_dist = df['target'].value_counts()
        print(f"Class distribution:")
        print(f"  Negative (0): {class_dist[0]} ({class_dist[0]/len(df)*100:.1f}%)")
        print(f"  Positive (1): {class_dist[1]} ({class_dist[1]/len(df)*100:.1f}%)")
        print(f"  Imbalance ratio: {class_dist[0]/class_dist[1]:.1f}:1")
        
        # Feature selection and engineering
        features = self.select_and_engineer_features(df)
        
        # Prepare final dataset
        X = df[features].copy()
        y = df['target'].copy()
        
        print(f"Final feature set: {len(features)} features")
        print(f"Final dataset shape: {X.shape}")
        
        return X, y, features
    
    def select_and_engineer_features(self, df):
        """Enhanced feature selection and engineering"""
        
        # Core demographic features
        demographic_features = ['age', 'sex']
        
        # Symptom features (binary)
        symptom_features = [
            'fever_new', 'headache_new', 'vomiting_new', 'bleeding_gums', 
            'bleeding_from_eyes', 'diarrhea_new', 'muscle_pain', 'joint_pain_arthritis',
            'sore_throat', 'cough_new', 'difficulty_breathing', 'weakness_new',
            'abdominal_pain', 'backache_new', 'chest_pain', 'fatigue_weakness', 'nausea_new'
        ]
        
        # Epidemiological features
        epi_features = [
            'contact_with_source_case_new', 'direct_contact_probable_case',
            'travelled_outside_district', 'burial_of_case'
        ]
        
        # Geographic features
        geographic_features = ['state_residence_new']
        
        # Select available features
        available_features = []
        
        # Add demographic features
        for feat in demographic_features:
            if feat in df.columns:
                available_features.append(feat)
                # Handle missing values
                if feat == 'age':
                    df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(df[feat].median())
                elif feat == 'sex':
                    df[feat] = df[feat].map({'MALE': 1, 'FEMALE': 0, 'M': 1, 'F': 0, 1: 1, 0: 0}).fillna(0)
        
        # Add symptom features
        for feat in symptom_features:
            if feat in df.columns:
                available_features.append(feat)
                # Convert to binary
                df[feat] = df[feat].map({'YES': 1, 'NO': 0, 'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0)
        
        # Add epidemiological features
        for feat in epi_features:
            if feat in df.columns:
                available_features.append(feat)
                # Handle unknown values as neutral (0.5) rather than missing
                df[feat] = df[feat].map({'YES': 1, 'NO': 0, 'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0.5)
        
        # Add geographic features with encoding
        for feat in geographic_features:
            if feat in df.columns:
                # Create state encoder
                le = LabelEncoder()
                df[feat] = df[feat].fillna('UNKNOWN')
                df[feat + '_encoded'] = le.fit_transform(df[feat])
                available_features.append(feat + '_encoded')
                self.encoders[feat] = le
        
        # Feature engineering - create composite features
        if 'fever_new' in df.columns and 'headache_new' in df.columns:
            df['fever_headache_combo'] = df['fever_new'] * df['headache_new']
            available_features.append('fever_headache_combo')
        
        if 'bleeding_gums' in df.columns and 'bleeding_from_eyes' in df.columns:
            df['bleeding_combo'] = np.maximum(df['bleeding_gums'], df['bleeding_from_eyes'])
            available_features.append('bleeding_combo')
        
        # Age-based risk groups
        if 'age' in df.columns:
            df['age_risk_group'] = pd.cut(df['age'], 
                                        bins=[0, 18, 35, 50, 65, 100], 
                                        labels=[0, 1, 2, 3, 4]).astype(int)
            available_features.append('age_risk_group')
        
        # Symptom count
        symptom_cols = [col for col in symptom_features if col in df.columns]
        if symptom_cols:
            df['symptom_count'] = df[symptom_cols].sum(axis=1)
            available_features.append('symptom_count')
        
        # Contact risk score
        contact_cols = [col for col in epi_features if col in df.columns]
        if contact_cols:
            df['contact_risk_score'] = df[contact_cols].sum(axis=1)
            available_features.append('contact_risk_score')
        
        print(f"Selected {len(available_features)} features: {available_features[:10]}...")
        return available_features
    
    def handle_class_imbalance(self, X, y, method='smote_tomek'):
        """Handle class imbalance with multiple techniques"""
        print(f"\n🔄 Handling class imbalance using {method}...")
        
        original_counts = np.bincount(y)
        print(f"Original distribution: {original_counts}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=42, k_neighbors=3)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=42)
        elif method == 'smote_enn':
            sampler = SMOTEENN(random_state=42)
        else:
            print("No sampling applied")
            return X, y
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            new_counts = np.bincount(y_resampled)
            print(f"Resampled distribution: {new_counts}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"Sampling failed: {e}. Using original data.")
            return X, y
    
    def train_xgboost_model(self, X, y, use_sampling=True):
        """Train XGBoost model with proper handling of class imbalance"""
        print("\n🚀 Training XGBoost Model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate class weights for cost-sensitive learning
        classes = np.unique(y_train)
        self.class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        scale_pos_weight = self.class_weights[1] / self.class_weights[0]
        
        print(f"Class weights: {dict(zip(classes, self.class_weights))}")
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Handle class imbalance with sampling if requested
        if use_sampling:
            X_train_balanced, y_train_balanced = self.handle_class_imbalance(
                X_train_scaled, y_train, method='smote_tomek'
            )
        else:
            X_train_balanced, y_train_balanced = X_train_scaled, y_train
        
        # XGBoost parameters optimized for medical diagnosis
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'scale_pos_weight': scale_pos_weight,  # Handle imbalance
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 20
        }
        
        # Train model
        self.model = xgb.XGBClassifier(**xgb_params)
        
        # Fit with validation set for early stopping
        X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(
            X_train_balanced, y_train_balanced, test_size=0.2, random_state=42, stratify=y_train_balanced
        )
        
        self.model.fit(
            X_val_train, y_val_train,
            eval_set=[(X_val_test, y_val_test)],
            verbose=False
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n📊 Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Calculate precision and recall for positive class
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nDetailed Metrics for Positive Class:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
        
        # Store model info
        model_info = {
            'model_type': 'xgboost',
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': dict(zip(feature_names, feature_importance)),
            'training_date': datetime.now().isoformat(),
            'class_weights': dict(zip(classes, self.class_weights)),
            'scale_pos_weight': scale_pos_weight
        }
        
        return model_info, (X_test_scaled, y_test, y_pred, y_pred_proba)
    
    def save_model(self, model_info, save_path='model_balanced.pkl'):
        """Save the trained model and preprocessing components"""
        print(f"\n💾 Saving model to {save_path}...")
        
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'encoders': self.encoders,
            'features_used': self.features_used,
            'class_weights': self.class_weights,
            'model_info': model_info
        }
        
        joblib.dump(model_package, save_path)
        print(f"✅ Model saved successfully!")
        
        return save_path
    
    def plot_results(self, test_data):
        """Plot model performance visualizations"""
        X_test, y_test, y_pred, y_pred_proba = test_data
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, label=f'ROC AUC = {roc_auc_score(y_test, y_pred_proba):.3f}')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1, 0].plot(recall, precision)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        
        # Feature Importance (top 10)
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            indices = np.argsort(importance)[-10:]
            axes[1, 1].barh(range(len(indices)), importance[indices])
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_training(self):
        """Complete training pipeline"""
        print("🎯 Starting Balanced Lassa Fever Model Training...")
        
        # Load and preprocess data
        X, y, features = self.load_and_preprocess_data()
        self.features_used = features
        
        # Train model
        model_info, test_data = self.train_xgboost_model(X, y, use_sampling=True)
        
        # Save model
        model_path = self.save_model(model_info)
        
        # Plot results
        self.plot_results(test_data)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"Final accuracy: {model_info['accuracy']:.4f}")
        print(f"ROC AUC: {model_info['roc_auc']:.4f}")
        
        return model_info

if __name__ == "__main__":
    # Initialize trainer
    trainer = BalancedLassaFeverTrainer('sample-data.csv')
    
    # Run training
    results = trainer.run_training()
