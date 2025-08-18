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

def create_enhanced_graph(X, df, k=8, similarity_threshold=0.1):
    """Create enhanced graph structure with medical domain knowledge"""
    print(f"=== Creating Enhanced Medical Graph Structure ===")
    n_samples = X.shape[0]
    
    # Adaptive k based on dataset size
    k_adaptive = min(max(5, int(np.sqrt(n_samples) * 0.3)), 12)
    k_use = min(k_adaptive, k, n_samples - 1)
    
    print(f"Using k={k_use} for graph construction")
    
    # Initialize edge sets
    edges = set()
    edge_weights = {}
    
    # 1. Geographic proximity edges (stronger medical relevance)
    if 'state_residence_new' in df.columns:
        state_groups = df.groupby('state_residence_new').groups
        for state, indices in state_groups.items():
            if len(indices) > 1:
                indices_list = indices.tolist()
                # Connect patients from same state with higher weight
                for i in range(len(indices_list)):
                    for j in range(i+1, min(i+6, len(indices_list))):  # Limit connections per state
                        idx1, idx2 = indices_list[i], indices_list[j]
                        edges.add((idx1, idx2))
                        edge_weights[(idx1, idx2)] = 0.8  # High weight for geographic proximity
    
    # 2. Contact tracing edges (highest medical relevance)
    contact_cols = ['contact_with_source_case_new', 'direct_contact_probable_case']
    for col in contact_cols:
        if col in df.columns:
            # Connect patients with confirmed contact history
            contact_indices = df[df[col] == 1].index.tolist()
            if len(contact_indices) > 1:
                for i in range(len(contact_indices)):
                    for j in range(i+1, min(i+4, len(contact_indices))):  # Limit contact connections
                        idx1, idx2 = contact_indices[i], contact_indices[j]
                        edges.add((idx1, idx2))
                        edge_weights[(idx1, idx2)] = 0.9  # Very high weight for contact tracing
    
    # 3. Temporal proximity edges (patients with similar onset dates)
    date_cols = ['date_symptom_onset2', 'initial_sample_date2']
    for col in date_cols:
        if col in df.columns and not df[col].isnull().all():
            # Group by similar dates (within 7 days)
            df_temp = df.dropna(subset=[col])
            if len(df_temp) > 1:
                df_temp = df_temp.sort_values(col)
                for i in range(len(df_temp)-1):
                    for j in range(i+1, min(i+4, len(df_temp))):
                        idx1, idx2 = df_temp.index[i], df_temp.index[j]
                        date_diff = abs(df_temp.iloc[j][col] - df_temp.iloc[i][col])
                        # Convert timedelta to days for comparison
                        if hasattr(date_diff, 'days'):
                            date_diff_days = date_diff.days
                        elif hasattr(date_diff, 'total_seconds'):
                            date_diff_days = date_diff.total_seconds() / (24 * 3600)
                        else:
                            date_diff_days = abs(float(date_diff)) / (24 * 3600)
                        
                        if date_diff_days <= 7:  # Within 7 days
                            edges.add((idx1, idx2))
                            edge_weights[(idx1, idx2)] = 0.7  # Medium-high weight for temporal proximity
    
    # 4. Enhanced feature similarity edges (improved from original)
    if n_samples > 5000:
        similarity_edge_index, similarity_weights = build_large_graph_efficient(X, k_use, similarity_threshold)
    else:
        similarity_edge_index, similarity_weights = build_standard_graph(X, k_use, similarity_threshold)
    
    # Convert tensor edge_index to edge tuples and combine with weights
    if isinstance(similarity_edge_index, torch.Tensor):
        # Convert edge_index tensor to list of tuples
        similarity_edges_array = similarity_edge_index.t().numpy()  # Transpose and convert to numpy
        similarity_edges_list = [(int(edge[0]), int(edge[1])) for edge in similarity_edges_array]
        
        # Create weights dictionary
        if isinstance(similarity_weights, torch.Tensor):
            similarity_weights_dict = {edge: similarity_weights[i].item() for i, edge in enumerate(similarity_edges_list)}
        else:
            similarity_weights_dict = {edge: 0.4 for edge in similarity_edges_list}
    else:
        similarity_edges_list = list(similarity_edges)
        similarity_weights_dict = similarity_weights if isinstance(similarity_weights, dict) else {}
    
    # Add similarity edges
    for edge in similarity_edges_list:
        if edge not in edges:  # Don't override medical edges
            edges.add(edge)
            edge_weights[edge] = similarity_weights_dict.get(edge, 0.4)  # Lower weight for pure similarity
    
    # 5. Symptom pattern edges (connect patients with similar critical symptoms)
    critical_symptoms = ['fever_new', 'bleeding_gums', 'bleeding_from_eyes', 'headache_new']
    available_symptoms = [col for col in critical_symptoms if col in df.columns]
    
    if available_symptoms:
        for symptom in available_symptoms:
            symptom_positive = df[df[symptom] == 1].index.tolist()
            if len(symptom_positive) > 1:
                # Connect patients with same critical symptoms (limited connections)
                for i in range(len(symptom_positive)):
                    for j in range(i+1, min(i+3, len(symptom_positive))):
                        idx1, idx2 = symptom_positive[i], symptom_positive[j]
                        if (idx1, idx2) not in edges:
                            edges.add((idx1, idx2))
                            edge_weights[(idx1, idx2)] = 0.6  # Medium weight for symptom similarity
    
    return finalize_enhanced_graph(edges, edge_weights, n_samples)

def finalize_enhanced_graph(edges, edge_weights, n_samples):
    """Finalize enhanced graph construction with better connectivity"""
    # Make graph undirected and ensure connectivity
    edges_undirected = set()
    weights_undirected = {}
    
    for (i, j) in edges:
        edges_undirected.add((i, j))
        edges_undirected.add((j, i))
        weight = edge_weights.get((i, j), edge_weights.get((j, i), 0.5))
        weights_undirected[(i, j)] = weight
        weights_undirected[(j, i)] = weight
    
    # Ensure minimum connectivity for isolated nodes
    if len(edges_undirected) == 0:
        # Create minimal connectivity
        for i in range(min(10, n_samples-1)):
            edges_undirected.add((i, i+1))
            edges_undirected.add((i+1, i))
            weights_undirected[(i, i+1)] = 0.3
            weights_undirected[(i+1, i)] = 0.3
    
    # Check for isolated nodes and connect them
    connected_nodes = set()
    for (i, j) in edges_undirected:
        connected_nodes.add(i)
        connected_nodes.add(j)
    
    isolated_nodes = set(range(n_samples)) - connected_nodes
    if isolated_nodes:
        print(f"Connecting {len(isolated_nodes)} isolated nodes")
        for node in isolated_nodes:
            # Connect to nearest connected node
            nearest_connected = min(connected_nodes, key=lambda x: abs(x - node))
            edges_undirected.add((node, nearest_connected))
            edges_undirected.add((nearest_connected, node))
            weights_undirected[(node, nearest_connected)] = 0.2
            weights_undirected[(nearest_connected, node)] = 0.2
    
    if len(edges_undirected) > 0:
        edge_list = list(edges_undirected)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weights_tensor = torch.tensor([weights_undirected[edge] for edge in edge_list], dtype=torch.float)
    else:
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
        edge_weights_tensor = torch.tensor([0.1, 0.1], dtype=torch.float)
    
    avg_degree = len(edges_undirected) / n_samples if n_samples > 0 else 0
    print(f"✅ Enhanced graph created: {len(edges_undirected)} edges, avg degree: {avg_degree:.2f}")
    
    # Print edge type distribution
    medical_edges = sum(1 for w in weights_undirected.values() if w >= 0.7)
    similarity_edges = sum(1 for w in weights_undirected.values() if w < 0.5)
    print(f"   Medical edges (contact/geographic): {medical_edges}")
    print(f"   Similarity edges: {similarity_edges}")
    
    return edge_index, edge_weights_tensor

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
    edge_index, edge_weights = create_enhanced_graph(processed['X'], processed['df'], k=k)
    
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
