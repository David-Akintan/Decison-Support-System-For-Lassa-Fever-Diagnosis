import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score
)
from sklearn.calibration import calibration_curve
import joblib
from preprocess import df_to_pyg_data
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

def comprehensive_model_evaluation(model_path, preproc_path, csv_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Comprehensive evaluation of the trained Lassa fever diagnosis model
    """
    print("🔬 Comprehensive Model Evaluation for Lassa Fever Diagnosis")
    print("=" * 60)
    
    # Load model and preprocessor
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    preproc_data = joblib.load(preproc_path)
    
    # Load and preprocess data
    parsed = df_to_pyg_data(csv_path, k=12)
    data = Data(
        x=torch.tensor(parsed["x"], dtype=torch.float),
        edge_index=parsed["edge_index"],
        edge_attr=torch.tensor(parsed["edge_weights"], dtype=torch.float) if parsed["edge_weights"] is not None else None,
        y=torch.tensor(parsed["y"], dtype=torch.long)
    ).to(device)
    
    # Recreate model architecture
    model_config = checkpoint.get('model_config', {})
    model_type = model_config.get('model_type', 'hybrid')
    
    if model_type == 'hybrid':
        from models import HybridGNN
        model = HybridGNN(**model_config).to(device)
    elif model_type == 'gcn':
        from models import GCNNet
        model = GCNNet(**model_config).to(device)
    elif model_type == 'gat':
        from models import GATNet
        model = GATNet(**model_config).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test split (same as training)
    num_nodes = data.x.shape[0]
    indices = np.arange(num_nodes)
    y_numpy = data.y.cpu().numpy()
    
    from sklearn.model_selection import train_test_split
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_numpy
    )
    
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True
    
    # Model predictions
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
    
    # Extract test data
    test_true = data.y[test_mask].cpu().numpy()
    test_pred = predictions[test_mask].cpu().numpy()
    test_probs = probabilities[test_mask].cpu().numpy()
    
    print(f"📊 Test Set Size: {len(test_true)} samples")
    print(f"📊 Class Distribution: {np.bincount(test_true)}")
    
    # 1. Basic Classification Metrics
    print("\n🎯 Classification Performance:")
    print("-" * 40)
    
    accuracy = (test_pred == test_true).mean()
    balanced_acc = balanced_accuracy_score(test_true, test_pred)
    mcc = matthews_corrcoef(test_true, test_pred)
    kappa = cohen_kappa_score(test_true, test_pred)
    
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Balanced Accuracy:  {balanced_acc:.4f}")
    print(f"Matthews Corr Coef: {mcc:.4f}")
    print(f"Cohen's Kappa:      {kappa:.4f}")
    
    # 2. Class-specific Metrics
    print("\n📋 Detailed Classification Report:")
    print(classification_report(test_true, test_pred, 
                              target_names=['Negative', 'Lassa Positive']))
    
    # 3. Confusion Matrix
    cm = confusion_matrix(test_true, test_pred)
    print("\n🔢 Confusion Matrix:")
    print(f"                Predicted")
    print(f"Actual    Negative  Positive")
    print(f"Negative     {cm[0,0]:4d}     {cm[0,1]:4d}")
    print(f"Positive     {cm[1,0]:4d}     {cm[1,1]:4d}")
    
    # Calculate clinical metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\n🏥 Clinical Metrics:")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"PPV (Precision):      {ppv:.4f}")
    print(f"NPV:                  {npv:.4f}")
    
    # 4. ROC Analysis
    fpr, tpr, roc_thresholds = roc_curve(test_true, test_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    print(f"\n📈 ROC Analysis:")
    print(f"AUC-ROC: {roc_auc:.4f}")
    
    # 5. Precision-Recall Analysis
    precision, recall, pr_thresholds = precision_recall_curve(test_true, test_probs[:, 1])
    avg_precision = average_precision_score(test_true, test_probs[:, 1])
    
    print(f"AUC-PR:  {avg_precision:.4f}")
    
    # 6. Calibration Analysis
    fraction_of_positives, mean_predicted_value = calibration_curve(
        test_true, test_probs[:, 1], n_bins=10
    )
    
    # 7. Threshold Analysis for Clinical Decision Making
    print(f"\n⚖️  Threshold Analysis for Clinical Decision Making:")
    print("-" * 50)
    
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print(f"{'Threshold':<10} {'Sensitivity':<12} {'Specificity':<12} {'PPV':<8} {'NPV':<8} {'F1':<8}")
    print("-" * 60)
    
    for threshold in thresholds_to_test:
        pred_at_threshold = (test_probs[:, 1] >= threshold).astype(int)
        cm_thresh = confusion_matrix(test_true, pred_at_threshold)
        
        if cm_thresh.shape == (2, 2):
            tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()
            sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
            ppv_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            npv_t = tn_t / (tn_t + fn_t) if (tn_t + fn_t) > 0 else 0
            f1_t = 2 * tp_t / (2 * tp_t + fp_t + fn_t) if (2 * tp_t + fp_t + fn_t) > 0 else 0
            
            print(f"{threshold:<10.1f} {sens_t:<12.3f} {spec_t:<12.3f} {ppv_t:<8.3f} {npv_t:<8.3f} {f1_t:<8.3f}")
    
    # 8. Feature Importance Analysis
    print(f"\n🔍 Feature Importance Analysis:")
    analyze_feature_importance(model, data, parsed['features_used'], device)
    
    # 9. Error Analysis
    print(f"\n❌ Error Analysis:")
    analyze_prediction_errors(test_true, test_pred, test_probs, parsed['raw_df'].iloc[test_idx])
    
    # 10. Create Visualizations
    create_evaluation_plots(test_true, test_pred, test_probs, fpr, tpr, roc_auc, 
                          precision, recall, avg_precision, fraction_of_positives, 
                          mean_predicted_value, cm)
    
    # Return comprehensive results
    results = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'auc_roc': roc_auc,
        'auc_pr': avg_precision,
        'mcc': mcc,
        'kappa': kappa,
        'confusion_matrix': cm,
        'test_predictions': test_pred,
        'test_probabilities': test_probs,
        'test_true': test_true
    }
    
    return results

def analyze_feature_importance(model, data, feature_names, device):
    """
    Analyze feature importance using gradient-based methods
    """
    model.eval()
    
    # Simple gradient-based importance
    data.x.requires_grad_(True)
    
    with torch.enable_grad():
        logits = model(data.x, data.edge_index)
        # Focus on positive class (Lassa fever)
        positive_logits = logits[:, 1].sum()
        positive_logits.backward()
    
    # Get gradients as importance scores
    gradients = data.x.grad.abs().mean(dim=0).cpu().numpy()
    
    # Sort features by importance
    feature_importance = list(zip(feature_names, gradients))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance[:10]):
        print(f"  {i+1:2d}. {feature:<30} {importance:.6f}")

def analyze_prediction_errors(y_true, y_pred, y_probs, original_data):
    """
    Analyze prediction errors to understand model limitations
    """
    # False positives and false negatives
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    print(f"False Positives: {fp_mask.sum()}")
    print(f"False Negatives: {fn_mask.sum()}")
    
    if fp_mask.sum() > 0:
        print("\n🔴 False Positive Analysis:")
        fp_probs = y_probs[fp_mask, 1]
        print(f"  Average confidence: {fp_probs.mean():.3f}")
        print(f"  Confidence range: {fp_probs.min():.3f} - {fp_probs.max():.3f}")
    
    if fn_mask.sum() > 0:
        print("\n🟡 False Negative Analysis:")
        fn_probs = y_probs[fn_mask, 1]
        print(f"  Average confidence: {fn_probs.mean():.3f}")
        print(f"  Confidence range: {fn_probs.min():.3f} - {fn_probs.max():.3f}")

def create_evaluation_plots(y_true, y_pred, y_probs, fpr, tpr, roc_auc, 
                          precision, recall, avg_precision, fraction_of_positives, 
                          mean_predicted_value, cm):
    """
    Create comprehensive evaluation plots
    """
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # 2. ROC Curve
    axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0,1].set_xlim([0.0, 1.0])
    axes[0,1].set_ylim([0.0, 1.05])
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curve')
    axes[0,1].legend(loc="lower right")
    
    # 3. Precision-Recall Curve
    axes[0,2].plot(recall, precision, color='blue', lw=2,
                   label=f'PR curve (AUC = {avg_precision:.3f})')
    axes[0,2].set_xlabel('Recall')
    axes[0,2].set_ylabel('Precision')
    axes[0,2].set_title('Precision-Recall Curve')
    axes[0,2].legend()
    
    # 4. Calibration Plot
    axes[1,0].plot(mean_predicted_value, fraction_of_positives, "s-", 
                   label="Model")
    axes[1,0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    axes[1,0].set_xlabel('Mean Predicted Probability')
    axes[1,0].set_ylabel('Fraction of Positives')
    axes[1,0].set_title('Calibration Plot')
    axes[1,0].legend()
    
    # 5. Probability Distribution
    axes[1,1].hist(y_probs[y_true == 0, 1], bins=20, alpha=0.7, 
                   label='Negative', color='blue')
    axes[1,1].hist(y_probs[y_true == 1, 1], bins=20, alpha=0.7, 
                   label='Lassa Positive', color='red')
    axes[1,1].set_xlabel('Predicted Probability (Positive Class)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Probability Distribution')
    axes[1,1].legend()
    
    # 6. Class Balance
    class_counts = [np.sum(y_true == 0), np.sum(y_true == 1)]
    axes[1,2].bar(['Negative', 'Lassa Positive'], class_counts, 
                  color=['blue', 'red'], alpha=0.7)
    axes[1,2].set_ylabel('Count')
    axes[1,2].set_title('Test Set Class Distribution')
    
    plt.tight_layout()
    plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Evaluation plots saved to 'model_evaluation_plots.png'")

def clinical_validation_report(results):
    """
    Generate clinical validation report for healthcare deployment
    """
    print(f"\n🏥 Clinical Validation Report for Lassa Fever Diagnosis AI")
    print("=" * 60)
    
    sensitivity = results['sensitivity']
    specificity = results['specificity']
    ppv = results['ppv']
    npv = results['npv']
    
    print(f"📋 Clinical Performance Summary:")
    print(f"  • Sensitivity (True Positive Rate): {sensitivity:.1%}")
    print(f"    → Out of 100 Lassa fever cases, {sensitivity*100:.0f} would be correctly identified")
    print(f"  • Specificity (True Negative Rate): {specificity:.1%}")
    print(f"    → Out of 100 healthy patients, {specificity*100:.0f} would be correctly identified")
    print(f"  • Positive Predictive Value: {ppv:.1%}")
    print(f"    → When AI predicts Lassa fever, it's correct {ppv*100:.0f}% of the time")
    print(f"  • Negative Predictive Value: {npv:.1%}")
    print(f"    → When AI predicts no Lassa fever, it's correct {npv*100:.0f}% of the time")
    
    print(f"\n⚠️  Clinical Considerations:")
    if sensitivity < 0.85:
        print(f"  • CONCERN: Sensitivity ({sensitivity:.1%}) may miss critical cases")
        print(f"    → Consider lowering decision threshold for higher sensitivity")
    else:
        print(f"  • ✅ Good sensitivity for critical disease detection")
    
    if specificity < 0.80:
        print(f"  • CONCERN: Specificity ({specificity:.1%}) may cause unnecessary treatments")
    else:
        print(f"  • ✅ Good specificity to avoid false alarms")
    
    print(f"\n📊 Recommended Clinical Usage:")
    print(f"  • Use as screening tool in endemic areas")
    print(f"  • Combine with clinical judgment and laboratory tests")
    print(f"  • Consider patient history and epidemiological factors")
    print(f"  • Regular model retraining with new data recommended")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Model Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--preproc", type=str, required=True, help="Path to preprocessor file")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--clinical", action="store_true", help="Generate clinical validation report")
    
    args = parser.parse_args()
    
    # Run comprehensive evaluation
    results = comprehensive_model_evaluation(args.model, args.preproc, args.csv)
    
    if args.clinical:
        clinical_validation_report(results)
    
    print(f"\n✅ Evaluation Complete!")
    print(f"📊 Key Metrics: Sensitivity={results['sensitivity']:.3f}, "
          f"Specificity={results['specificity']:.3f}, AUC={results['auc_roc']:.3f}")
