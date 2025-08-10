import optuna
import torch
import numpy as np
from train import enhanced_train
from preprocess import df_to_pyg_data
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def objective(trial):
    """
    Optuna objective function for hyperparameter optimization
    """
    # Suggest hyperparameters
    params = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'hidden_channels': trial.suggest_categorical('hidden_channels', [64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 2, 4),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'k': trial.suggest_int('k', 8, 16),
        'fusion_method': trial.suggest_categorical('fusion_method', 
            ['advanced_attention', 'cross_attention', 'gated_fusion']),
        'minority_boost': trial.suggest_float('minority_boost', 2.0, 5.0),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
        'mixup_alpha': trial.suggest_float('mixup_alpha', 0.0, 0.4),
        'loss_type': trial.suggest_categorical('loss_type', ['combined', 'focal']),
        'model_type': trial.suggest_categorical('model_type', ['hybrid', 'ensemble'])
    }
    
    # Cross-validation for robust evaluation
    csv_path = "data/lassa_fever_data.csv"  # Update with your data path
    
    try:
        # Quick training with reduced epochs for optimization
        model, metrics, _ = enhanced_train(
            csv_path=csv_path,
            model_out=f"temp_model_{trial.number}.pth",
            preprocess_out=f"temp_preproc_{trial.number}.pkl",
            epochs=50,  # Reduced for faster optimization
            patience=15,
            **params
        )
        
        # Clean up temporary files
        if os.path.exists(f"temp_model_{trial.number}.pth"):
            os.remove(f"temp_model_{trial.number}.pth")
        if os.path.exists(f"temp_preproc_{trial.number}.pkl"):
            os.remove(f"temp_preproc_{trial.number}.pkl")
        
        # Return minority F1 score (primary metric for Lassa fever)
        return metrics.get('minority_f1', 0.0)
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0

def run_hyperparameter_optimization(n_trials=50, csv_path="data/lassa_fever_data.csv"):
    """
    Run comprehensive hyperparameter optimization
    """
    print("🔍 Starting Hyperparameter Optimization for Lassa Fever GNN")
    print(f"📊 Running {n_trials} trials...")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, timeout=7200)  # 2 hours max
    
    # Results
    print("\n🎯 Optimization Results:")
    print(f"Best Minority F1: {study.best_value:.4f}")
    print(f"Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    joblib.dump(study, 'hyperparameter_optimization_results.pkl')
    
    # Train final model with best parameters
    print("\n🚀 Training final model with best parameters...")
    final_model, final_metrics, training_history = enhanced_train(
        csv_path=csv_path,
        model_out="best_model.pth",
        preprocess_out="best_preproc.pkl",
        epochs=150,
        patience=25,
        **study.best_params
    )
    
    print("\n✅ Final Model Results:")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return study, final_model, final_metrics

def analyze_feature_importance(model_path, preproc_path, csv_path):
    """
    Analyze feature importance using permutation importance
    """
    print("🔬 Analyzing Feature Importance...")
    
    # Load model and preprocessor
    checkpoint = torch.load(model_path, weights_only=False)
    preproc_data = joblib.load(preproc_path)
    
    # Load and preprocess data
    parsed = df_to_pyg_data(csv_path, k=12)
    features_used = parsed['features_used']
    
    print(f"📋 Top Important Features:")
    
    # Simple feature importance based on model weights
    # This is a simplified version - for full analysis, use SHAP or permutation importance
    if hasattr(checkpoint, 'model_state_dict'):
        state_dict = checkpoint['model_state_dict']
        
        # Get first layer weights as proxy for feature importance
        first_layer_key = None
        for key in state_dict.keys():
            if 'weight' in key and len(state_dict[key].shape) == 2:
                first_layer_key = key
                break
        
        if first_layer_key:
            weights = state_dict[first_layer_key].abs().mean(dim=0).cpu().numpy()
            
            # Sort features by importance
            feature_importance = list(zip(features_used, weights))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("Top 15 Most Important Features:")
            for i, (feature, importance) in enumerate(feature_importance[:15]):
                print(f"  {i+1:2d}. {feature:<25} {importance:.4f}")
    
    return feature_importance if 'feature_importance' in locals() else None

def create_optimization_report(study_path='hyperparameter_optimization_results.pkl'):
    """
    Create detailed optimization report
    """
    if not os.path.exists(study_path):
        print("❌ No optimization results found. Run optimization first.")
        return
    
    study = joblib.load(study_path)
    
    print("\n📊 Hyperparameter Optimization Report")
    print("=" * 50)
    
    print(f"🎯 Best Trial: #{study.best_trial.number}")
    print(f"🏆 Best Minority F1: {study.best_value:.4f}")
    
    print(f"\n📈 Parameter Importance:")
    importance = optuna.importance.get_param_importances(study)
    for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param:<20} {imp:.3f}")
    
    print(f"\n🔍 Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key:<20} {value}")
    
    # Plot optimization history (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Optimization history
        trials = study.trials
        values = [t.value for t in trials if t.value is not None]
        ax1.plot(values)
        ax1.set_title('Optimization History')
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Minority F1 Score')
        
        # Parameter importance
        params = list(importance.keys())
        importances = list(importance.values())
        ax2.barh(params, importances)
        ax2.set_title('Parameter Importance')
        ax2.set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('optimization_report.png', dpi=300, bbox_inches='tight')
        print("\n📊 Plots saved to 'optimization_report.png'")
        
    except ImportError:
        print("\n📊 Install matplotlib for visualization: pip install matplotlib")

def benchmark_models(csv_path="data/lassa_fever_data.csv"):
    """
    Benchmark different model architectures
    """
    print("🏁 Benchmarking Model Architectures...")
    
    models_to_test = [
        {'model_type': 'hybrid', 'fusion_method': 'advanced_attention'},
        {'model_type': 'hybrid', 'fusion_method': 'cross_attention'},
        {'model_type': 'hybrid', 'fusion_method': 'gated_fusion'},
        {'model_type': 'gcn'},
        {'model_type': 'gat'},
        {'model_type': 'ensemble'}
    ]
    
    results = []
    
    for i, config in enumerate(models_to_test):
        print(f"\n🔄 Testing {config}...")
        
        try:
            model, metrics, _ = enhanced_train(
                csv_path=csv_path,
                model_out=f"benchmark_model_{i}.pth",
                preprocess_out=f"benchmark_preproc_{i}.pkl",
                epochs=75,  # Moderate training for comparison
                patience=20,
                **config
            )
            
            results.append({
                'config': config,
                'metrics': metrics
            })
            
            # Clean up
            if os.path.exists(f"benchmark_model_{i}.pth"):
                os.remove(f"benchmark_model_{i}.pth")
            if os.path.exists(f"benchmark_preproc_{i}.pkl"):
                os.remove(f"benchmark_preproc_{i}.pkl")
                
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                'config': config,
                'metrics': {'error': str(e)}
            })
    
    # Print benchmark results
    print("\n🏆 Benchmark Results:")
    print("-" * 80)
    print(f"{'Model':<25} {'Minority F1':<12} {'Accuracy':<10} {'AUC':<8} {'Precision':<10} {'Recall':<8}")
    print("-" * 80)
    
    for result in results:
        config = result['config']
        metrics = result['metrics']
        
        if 'error' not in metrics:
            model_name = f"{config.get('model_type', 'unknown')}"
            if 'fusion_method' in config:
                model_name += f"_{config['fusion_method']}"
            
            print(f"{model_name:<25} "
                  f"{metrics.get('minority_f1', 0):<12.4f} "
                  f"{metrics.get('accuracy', 0):<10.4f} "
                  f"{metrics.get('auc', 0):<8.4f} "
                  f"{metrics.get('precision', 0):<10.4f} "
                  f"{metrics.get('recall', 0):<8.4f}")
        else:
            print(f"{str(config):<25} ERROR: {metrics['error']}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for Lassa Fever GNN")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--benchmark", action="store_true", help="Run model benchmarking")
    parser.add_argument("--analyze", action="store_true", help="Analyze feature importance")
    parser.add_argument("--report", action="store_true", help="Generate optimization report")
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_models(args.csv)
    elif args.analyze:
        analyze_feature_importance("best_model.pth", "best_preproc.pkl", args.csv)
    elif args.report:
        create_optimization_report()
    else:
        run_hyperparameter_optimization(args.trials, args.csv)
