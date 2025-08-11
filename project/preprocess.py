# preprocess_.py - Enhanced preprocessing with data quality improvements
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

def enhanced_data_quality_check(df, target_col):
    """Comprehensive data quality analysis and improvement"""
    print("=== Data Quality Analysis ===")
    
    # Basic statistics
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Target distribution
    if target_col in df.columns:
        print(f"Target distribution:\n{df[target_col].value_counts()}")
        
        # Check for data leakage (features that perfectly predict target)
        for col in df.columns:
            if col != target_col and df[col].dtype in ['object', 'category']:
                try:
                    correlation = df.groupby(col)[target_col].apply(lambda x: x.value_counts(normalize=True))
                    if len(correlation) > 0:
                        max_correlation = correlation.max()
                        if max_correlation > 0.95:
                            print(f"⚠️  Potential data leakage in column '{col}' (correlation: {max_correlation:.3f})")
                except:
                    pass
    
    # Remove completely empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        print(f"Removing {len(empty_cols)} completely empty columns")
        df = df.drop(columns=empty_cols)
    
    # Remove duplicate rows
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
        print(f"Removed duplicate rows. New shape: {df.shape}")
    
    # Handle extreme outliers in numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col != target_col:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"Column '{col}': {outliers} extreme outliers detected")
                # Cap outliers instead of removing
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def intelligent_feature_engineering(df, target_col):
    """Enhanced feature engineering with medical domain knowledge"""
    print("=== Intelligent Feature Engineering ===")
    
    # Symptom-based features
    symptom_keywords = ['fever', 'headache', 'vomit', 'cough', 'bleed', 'pain', 'weakness', 
                       'diarrhea', 'fatigue', 'nausea', 'chills', 'muscle', 'joint']
    
    symptom_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in symptom_keywords):
            symptom_cols.append(col)
    
    if len(symptom_cols) >= 3:
        # Create symptom severity indicators
        df['total_symptoms'] = 0
        df['severe_symptoms'] = 0
        
        for col in symptom_cols:
            # Convert to binary if not already
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.upper().isin(['YES', 'POSITIVE', 'TRUE', '1']).astype(int)
            elif df[col].dtype in ['int64', 'float64']:
                df[col] = (df[col] > 0).astype(int)
            
            df['total_symptoms'] += df[col]
            
            # Severe symptoms (bleeding, high fever, etc.)
            if any(severe in col.lower() for severe in ['bleed', 'hemorrhage', 'blood']):
                df['severe_symptoms'] += df[col]
        
        # Risk categories
        df['symptom_risk_level'] = pd.cut(df['total_symptoms'], 
                                        bins=[-1, 0, 2, 5, 100], 
                                        labels=['none', 'mild', 'moderate', 'severe'])
        
        # Critical symptom combinations
        bleeding_cols = [col for col in symptom_cols if 'bleed' in col.lower()]
        fever_cols = [col for col in symptom_cols if 'fever' in col.lower()]
        
        if bleeding_cols and fever_cols:
            df['has_bleeding'] = df[bleeding_cols].max(axis=1)
            df['has_fever'] = df[fever_cols].max(axis=1)
            df['fever_bleeding_combo'] = df['has_fever'] * df['has_bleeding']
            print("Created fever-bleeding combination feature")
    
    # Contact/exposure features
    contact_keywords = ['contact', 'exposure', 'travel', 'rodent', 'case']
    contact_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in contact_keywords)]
    
    if contact_cols:
        df['exposure_risk'] = 0
        for col in contact_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.upper().isin(['YES', 'POSITIVE', 'TRUE', '1']).astype(int)
            df['exposure_risk'] += df[col]
        print(f"Created exposure risk score from {len(contact_cols)} contact features")
    
    # Geographic risk (Lassa fever endemic areas)
    state_cols = [col for col in df.columns if 'state' in col.lower()]
    if state_cols:
        # Nigerian states with high Lassa fever incidence
        high_risk_states = ['EDO', 'ONDO', 'BAUCHI', 'TARABA', 'EBONYI', 'KOGI', 'PLATEAU', 'RIVERS']
        
        for state_col in state_cols:
            df[f'{state_col}_endemic_risk'] = (
                df[state_col].astype(str).str.upper().isin(high_risk_states).astype(int)
            )
            print(f"Created endemic risk feature for {state_col}")
    
    # Age-related risk factors
    age_cols = [col for col in df.columns if 'age' in col.lower()]
    if age_cols:
        age_col = age_cols[0]
        if df[age_col].dtype in ['int64', 'float64']:
            # Age risk categories for Lassa fever
            df['age_risk_category'] = pd.cut(df[age_col], 
                                           bins=[0, 5, 15, 45, 65, 100], 
                                           labels=['infant', 'child', 'young_adult', 'adult', 'elderly'])
            
            # High-risk age groups (very young and elderly)
            df['high_risk_age'] = ((df[age_col] < 5) | (df[age_col] > 65)).astype(int)
            print("Created age-based risk features")
    
    # Temporal features (if date columns exist)
    date_cols = [col for col in df.columns if any(date_term in col.lower() 
                for date_term in ['date', 'time', 'onset'])]
    
    for date_col in date_cols:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            if not df[date_col].isnull().all():
                df[f'{date_col}_month'] = df[date_col].dt.month
                df[f'{date_col}_season'] = df[date_col].dt.month.map({
                    12: 'dry', 1: 'dry', 2: 'dry', 3: 'dry',
                    4: 'wet', 5: 'wet', 6: 'wet', 7: 'wet', 8: 'wet', 9: 'wet',
                    10: 'dry', 11: 'dry'
                })
                print(f"Created temporal features from {date_col}")
        except:
            pass
    
    return df

def enhanced_preprocessing_pipeline(df, target_col, test_size=0.2):
    """Complete enhanced preprocessing pipeline"""
    print("=== Enhanced Preprocessing Pipeline ===")
    
    # Step 1: Data quality improvement
    df = enhanced_data_quality_check(df, target_col)
    
    # Step 2: Feature engineering
    df = intelligent_feature_engineering(df, target_col)
    
    # Step 3: Handle target variable
    if target_col in df.columns:
        # Clean target labels
        df[target_col] = df[target_col].astype(str).str.upper().str.strip()
        
        # Remove pending/unknown cases for training
        valid_labels = ['POSITIVE', 'NEGATIVE', 'POS', 'NEG', '1', '0', 'TRUE', 'FALSE']
        mask = df[target_col].isin(valid_labels)
        df = df[mask]
        print(f"Filtered to {df.shape[0]} samples with valid labels")
        
        # Binary encoding
        positive_values = {'POSITIVE', 'POS', '1', 'TRUE'}
        df['binary_label'] = df[target_col].apply(lambda x: 1 if x in positive_values else 0)
        
        print(f"Final label distribution:\n{df['binary_label'].value_counts()}")
    
    # Step 4: Feature selection and cleaning
    # Remove ID columns and other non-predictive features
    id_patterns = ['id', 'uid', 'key', 'index', 'serial', 'number']
    id_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in id_patterns)]
    
    # Remove date columns that were converted to features
    date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
    
    cols_to_remove = list(set(id_cols + date_cols + [target_col]))
    feature_cols = [col for col in df.columns if col not in cols_to_remove and col != 'binary_label']
    
    print(f"Selected {len(feature_cols)} feature columns")
    
    # Step 5: Handle missing values intelligently
    # Separate numerical and categorical features
    numerical_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Advanced imputation for numerical features
    if numerical_features:
        # Use KNN imputation for better handling of missing values
        knn_imputer = KNNImputer(n_neighbors=5)
        df[numerical_features] = knn_imputer.fit_transform(df[numerical_features])
        print(f"Applied KNN imputation to {len(numerical_features)} numerical features")
    
    # Imputation for categorical features
    if categorical_features:
        for col in categorical_features:
            # Use mode for categorical variables
            mode_value = df[col].mode()
            if len(mode_value) > 0:
                df[col] = df[col].fillna(mode_value[0])
            else:
                df[col] = df[col].fillna('unknown')
        print(f"Applied mode imputation to {len(categorical_features)} categorical features")
    
    # Step 6: Encoding
    encoders = {}
    
    for col in categorical_features:
        if col in df.columns:
            # Use label encoding for categorical variables
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    # Step 7: Feature scaling
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    feature_matrix = df[feature_cols].values
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # Step 8: Final feature selection based on variance
    # Remove low-variance features
    from sklearn.feature_selection import VarianceThreshold
    variance_selector = VarianceThreshold(threshold=0.01)
    feature_matrix_final = variance_selector.fit_transform(feature_matrix_scaled)
    
    selected_feature_indices = variance_selector.get_support(indices=True)
    final_feature_names = [feature_cols[i] for i in selected_feature_indices]
    
    print(f"Final feature count after variance filtering: {len(final_feature_names)}")
    
    return {
        'X': feature_matrix_final,
        'y': df['binary_label'].values,
        'feature_names': final_feature_names,
        'encoders': encoders,
        'scaler': scaler,
        'variance_selector': variance_selector,
        'original_features': feature_cols,
        'df': df
    }

def create__graph(X, k=8, similarity_threshold=0.1):
    """Create  graph structure with better connectivity"""
    print(f"=== Creating Graph Structure ===")
    n_samples = X.shape[0]
    
    # Adaptive k based on dataset size and feature dimensionality
    k_adaptive = min(max(5, int(np.sqrt(n_samples) * 0.5)), 15)
    k_use = min(k_adaptive, k, n_samples - 1)
    
    print(f"Using k={k_use} for graph construction")
    
    # Memory-efficient graph construction
    if n_samples > 5000:
        return build_large_graph_efficient(X, k_use, similarity_threshold)
    else:
        return build_standard_graph(X, k_use, similarity_threshold)

def build_large_graph_efficient(X, k, similarity_threshold, chunk_size=1000):
    """Memory-efficient graph construction for large datasets"""
    n_samples = X.shape[0]
    edges = set()
    edge_weights = {}
    
    # Use ball tree for efficiency
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', algorithm='ball_tree')
    nbrs.fit(X)
    
    # Process in chunks
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        X_chunk = X[start_idx:end_idx]
        
        distances, indices = nbrs.kneighbors(X_chunk)
        
        for i_local, i_global in enumerate(range(start_idx, end_idx)):
            for j_idx in range(1, len(indices[i_local])):  # Skip self (index 0)
                j_global = indices[i_local][j_idx]
                distance = distances[i_local][j_idx]
                
                if distance > 0:  # Avoid identical samples
                    weight = 1.0 / (1.0 + distance)
                    if weight > similarity_threshold:
                        edges.add((int(i_global), int(j_global)))
                        edge_weights[(int(i_global), int(j_global))] = weight
        
        if (start_idx // chunk_size + 1) % 5 == 0:
            print(f"Processed {start_idx//chunk_size + 1}/{(n_samples-1)//chunk_size + 1} chunks")
    
    return finalize_graph(edges, edge_weights, n_samples)

def build_standard_graph(X, k, similarity_threshold):
    """Standard graph construction for smaller datasets"""
    n_samples = X.shape[0]
    
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    edges = set()
    edge_weights = {}
    
    for i in range(n_samples):
        for j_idx in range(1, len(indices[i])):  # Skip self
            j = indices[i][j_idx]
            distance = distances[i][j_idx]
            
            if distance > 0:
                weight = 1.0 / (1.0 + distance)
                if weight > similarity_threshold:
                    edges.add((int(i), int(j)))
                    edge_weights[(int(i), int(j))] = weight
    
    return finalize_graph(edges, edge_weights, n_samples)

def finalize_graph(edges, edge_weights, n_samples):
    """Finalize graph construction"""
    # Make graph undirected
    edges_undirected = set()
    weights_undirected = {}
    
    for (i, j) in edges:
        edges_undirected.add((i, j))
        edges_undirected.add((j, i))
        weight = edge_weights.get((i, j), edge_weights.get((j, i), 0.5))
        weights_undirected[(i, j)] = weight
        weights_undirected[(j, i)] = weight
    
    if len(edges_undirected) == 0:
        # Create minimal connectivity if no edges found
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
        edge_weights_tensor = torch.tensor([0.1, 0.1], dtype=torch.float)
    else:
        edge_list = list(edges_undirected)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weights_tensor = torch.tensor([weights_undirected[edge] for edge in edge_list], dtype=torch.float)
    
    avg_degree = len(edges_undirected) / n_samples if n_samples > 0 else 0
    print(f"✅ Graph created: {len(edges_undirected)} edges, avg degree: {avg_degree:.2f}")
    
    return edge_index, edge_weights_tensor

def df_to_pyg_data(csv_path, k=8):
    """Enhanced data preprocessing pipeline for PyG"""
    print("=== Enhanced PyG Data Preprocessing ===")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Enhanced preprocessing
    processed = enhanced_preprocessing_pipeline(df, 'InitialSampleFinalLaboratoryResultPathogentest')
    
    # Create graph
    edge_index, edge_weights = create__graph(processed['X'], k=k)
    
    return {
        'x': processed['X'],
        'y': processed['y'],
        'edge_index': edge_index,
        'edge_weights': edge_weights.numpy() if edge_weights is not None else None,
        'features_used': processed['feature_names'],
        'encoders': processed['encoders'],
        'scaler': processed['scaler'],
        'variance_selector': processed['variance_selector'],
        'raw_df': processed['df']
    }
