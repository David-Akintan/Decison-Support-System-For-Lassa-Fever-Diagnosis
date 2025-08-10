# preprocess.py
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity # type: ignore
import torch # type: ignore

def load_data(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    return df

def pick_label_column(df):
    # preferred label column from your dataset
    candidates = [
        "InitialSampleFinalLaboratoryResultPathogentest",
        "LatestSampleFinalLaboratoryResultPathogentest",
        "Lab_result",
        "LabResult"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("No known label column found in dataset. Please check CSV.")

def build_feature_set(df):
    # Enhanced feature selection for medical data
    # Core demographic features
    demographic_candidates = [
        "sex_new2", "age_recode", "age", "age_group", "age grouped3",
        "pregnancy", "Pregnancy", "pregnant"
    ]
    
    # Clinical symptoms - most important for diagnosis
    symptom_candidates = [
        "fever_new", "headache_new", "vomiting_new", "cough_new",
        "bleeding_new", "weakness_new", "vomiting", "fever",
        "abdominal_pain", "acute_hearing_loss", "anorexia_loss_of_appetite",
        "backache_new", "bleeding_injection_site", "bleeding_from_eyes",
        "bleeding_from_vagina", "bleeding_gums", "bleeding_bruising",
        "blood_in_stool", "blood_urine", "chest_pain", "chills_sweats",
        "conjunctivities_new", "coughing_out_blood", "dark_urine",
        "diarrhea_new", "difficulty_breathing", "fatigue_weakness",
        "fresh_blood_vomit", "hiccups_new", "jaundice_new",
        "joint_pain_arthritis", "malaise_new", "muscle_pain",
        "nausea_new", "nose_bleeding", "pain_behind_eyes",
        "sore_throat", "side_pain"
    ]
    
    # Epidemiological features
    epi_candidates = [
        "contact_history", "contact_with_lassa_case", "travel_history",
        "contact_with_source_case_new", "direct_contact_probable_case",
        "rodents_excreta", "travelled_outside_district"
    ]
    
    # Geographic features
    geo_candidates = [
        "place of residence", "State_new", "state_residence_new",
        "States grouped into geopolitical zones", "Hotspot States and others",
        "LGA_of_residence", "lga_new", "area_type"
    ]
    
    # Combine all candidates
    all_candidates = demographic_candidates + symptom_candidates + epi_candidates + geo_candidates
    features = [c for c in all_candidates if c in df.columns]
    
    # If still not enough features, use pattern matching
    if len(features) < 10:
        for col in df.columns:
            if any(prefix in col.lower() for prefix in [
                "fever", "headache", "vomit", "cough", "bleed", "symptom", 
                "contact", "travel", "pain", "weakness", "diarrhea"
            ]):
                if col not in features:
                    features.append(col)
    
    # Final fallback to numerical columns
    if len(features) == 0:
        numerics = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numerics if c.lower() not in ("did", "id", "date")]
    
    print(f"Selected {len(features)} features for model training")
    return features

def clean_and_encode(df, features, label_col):
    """Enhanced data cleaning and encoding for medical data"""
    df_work = df[features + [label_col]].copy()
    
    # Advanced missing value handling
    # For symptoms: missing often means "No" (0)
    symptom_cols = [c for c in features if any(s in c.lower() for s in [
        'fever', 'headache', 'vomit', 'cough', 'bleed', 'pain', 'weakness',
        'diarrhea', 'symptom', 'nausea', 'fatigue', 'chest', 'sore'
    ])]
    
    for col in symptom_cols:
        if col in df_work.columns:
            # Fill missing symptoms with 0 (absent)
            df_work[col] = df_work[col].fillna(0)
    
    # Feature engineering: Create composite features
    if 'fever_new' in df_work.columns and 'headache_new' in df_work.columns:
        df_work['fever_headache_combo'] = df_work['fever_new'] * df_work['headache_new']
    
    if 'bleeding_new' in df_work.columns and 'weakness_new' in df_work.columns:
        df_work['severe_symptoms'] = df_work['bleeding_new'] * df_work['weakness_new']
    
    # Age-based risk factors
    if 'age_recode' in df_work.columns:
        df_work['age_risk_high'] = (df_work['age_recode'] > 50).astype(int)
        df_work['age_risk_child'] = (df_work['age_recode'] < 15).astype(int)
    
    # Symptom count feature
    symptom_count = 0
    for col in symptom_cols:
        if col in df_work.columns:
            symptom_count += df_work[col].fillna(0)
    df_work['total_symptoms'] = symptom_count
    
    # Update features list to include new engineered features
    new_features = [col for col in df_work.columns if col != label_col]
    
    # Outlier detection and handling for numerical features
    numerical_cols = df_work.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col != label_col and col not in symptom_cols:
            Q1 = df_work[col].quantile(0.25)
            Q3 = df_work[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df_work[col] = df_work[col].clip(lower=lower_bound, upper=upper_bound)
    
    # For other features, use median/mode imputation
    encoders = {}
    for col in new_features:
        if col in df_work.columns:
            if df_work[col].dtype == 'object':
                # Categorical encoding with frequency-based handling
                le = LabelEncoder()
                # Fill missing with 'Unknown'
                df_work[col] = df_work[col].fillna('Unknown')
                df_work[col] = le.fit_transform(df_work[col].astype(str))
                encoders[col] = le
            else:
                # Numerical imputation
                if col not in symptom_cols:  # Already handled above
                    median_val = df_work[col].median()
                    df_work[col] = df_work[col].fillna(median_val)
    
    # Handle label column - BINARY CLASSIFICATION FOR LASSA FEVER
    print(f"Original label distribution: {df_work[label_col].value_counts()}")
    
    # Enhanced binary label mapping
    df_work[label_col] = df_work[label_col].astype(str).str.upper().str.strip()
    positive_values = {'POSITIVE', 'POS', 'P', '1', 'TRUE', 'YES'}
    df_work['binary_label'] = df_work[label_col].apply(
        lambda x: 1 if x in positive_values else 0
    )
    
    print(f"Binary label distribution: {df_work['binary_label'].value_counts()}")
    
    # Remove rows with insufficient information (too many missing values)
    missing_threshold = 0.7  # Remove rows with >70% missing values
    missing_ratio = df_work[new_features].isnull().sum(axis=1) / len(new_features)
    df_work = df_work[missing_ratio <= missing_threshold]
    
    # Feature scaling with RobustScaler (better for outliers)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    
    # Only scale numerical features
    numerical_features = [col for col in new_features if col in df_work.columns and df_work[col].dtype != 'object']
    df_work[numerical_features] = scaler.fit_transform(df_work[numerical_features])
    
    return df_work[new_features].values, df_work['binary_label'].values, encoders, scaler, new_features

def build_edge_index_improved(X, k=12, similarity_threshold=0.1, use_adaptive_k=True, chunk_size=1000):
    """
    Memory-efficient graph construction with chunked processing to avoid OOM errors
    """
    n_samples = X.shape[0]
    
    # Adaptive k based on dataset size
    if use_adaptive_k:
        k = min(max(8, int(np.sqrt(n_samples))), 20)  # Adaptive k between 8-20
    
    k_use = min(k + 1, n_samples) if n_samples > 1 else 1
    
    print(f"Building graph for {n_samples} samples with k={k_use-1} (chunk_size={chunk_size})")
    
    # For large datasets, use only Euclidean distance to save memory
    # Cosine similarity is too memory-intensive for large datasets
    if n_samples > 5000 or X.shape[1] > 10000:
        print(" Large dataset detected. Using Euclidean distance only to prevent OOM.")
        return build_edge_index_euclidean_only(X, k_use, similarity_threshold, chunk_size)
    
    # Original approach for smaller datasets
    edges = set()
    edge_weights = {}
    
    # Process in chunks to avoid memory issues
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        X_chunk = X[start_idx:end_idx]
        
        # 1. Euclidean distance for current chunk
        nbrs_euclidean = NearestNeighbors(n_neighbors=min(k_use, n_samples), metric="euclidean").fit(X)
        dists_euc, inds_euc = nbrs_euclidean.kneighbors(X_chunk)
        
        # Process Euclidean neighbors for this chunk
        for i_local, i_global in enumerate(range(start_idx, end_idx)):
            for j_idx, j in enumerate(inds_euc[i_local][1:]):  # Skip self
                if j != i_global and dists_euc[i_local][j_idx + 1] > 0:  # Avoid identical samples
                    weight_euc = 1.0 / (1.0 + dists_euc[i_local][j_idx + 1])
                    if weight_euc > similarity_threshold:
                        edges.add((int(i_global), int(j)))
                        edge_weights[(int(i_global), int(j))] = weight_euc
        
        print(f"Processed chunk {start_idx//chunk_size + 1}/{(n_samples-1)//chunk_size + 1}")
    
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
        edge_index = torch.tensor([[], []], dtype=torch.long)
        edge_weights_tensor = torch.tensor([], dtype=torch.float)
    else:
        edge_list = list(edges_undirected)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weights_tensor = torch.tensor([weights_undirected[edge] for edge in edge_list], dtype=torch.float)
    
    print(f" Graph constructed with {len(edges_undirected)} edges, avg degree: {len(edges_undirected)/n_samples:.2f}")
    return edge_index, edge_weights_tensor


def build_edge_index_euclidean_only(X, k_use, similarity_threshold, chunk_size):
    """
    Memory-efficient graph construction using only Euclidean distance
    """
    n_samples = X.shape[0]
    edges = set()
    edge_weights = {}
    
    # Use approximate nearest neighbors for very large datasets
    if n_samples > 10000:
        print(" Using approximate nearest neighbors for very large dataset")
        from sklearn.neighbors import NearestNeighbors
        # Reduce k for very large datasets
        k_reduced = min(k_use, 15)
        nbrs = NearestNeighbors(n_neighbors=k_reduced, metric="euclidean", algorithm='ball_tree').fit(X)
        
        # Process in chunks
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            X_chunk = X[start_idx:end_idx]
            
            dists, inds = nbrs.kneighbors(X_chunk)
            
            for i_local, i_global in enumerate(range(start_idx, end_idx)):
                for j_idx, j in enumerate(inds[i_local][1:]):  # Skip self
                    if j != i_global and dists[i_local][j_idx + 1] > 0:
                        weight = 1.0 / (1.0 + dists[i_local][j_idx + 1])
                        if weight > similarity_threshold:
                            edges.add((int(i_global), int(j)))
                            edge_weights[(int(i_global), int(j))] = weight
            
            if (start_idx // chunk_size + 1) % 5 == 0:  # Print every 5 chunks
                print(f"Processed {start_idx//chunk_size + 1}/{(n_samples-1)//chunk_size + 1} chunks")
    
    else:
        # Standard approach for medium datasets
        nbrs = NearestNeighbors(n_neighbors=k_use, metric="euclidean").fit(X)
        dists, inds = nbrs.kneighbors(X)
        
        for i in range(n_samples):
            for j_idx, j in enumerate(inds[i][1:]):  # Skip self
                if dists[i][j_idx + 1] > 0:
                    weight = 1.0 / (1.0 + dists[i][j_idx + 1])
                    if weight > similarity_threshold:
                        edges.add((int(i), int(j)))
                        edge_weights[(int(i), int(j))] = weight
    
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
        edge_index = torch.tensor([[], []], dtype=torch.long)
        edge_weights_tensor = torch.tensor([], dtype=torch.float)
    else:
        edge_list = list(edges_undirected)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weights_tensor = torch.tensor([weights_undirected[edge] for edge in edge_list], dtype=torch.float)
    
    return edge_index, edge_weights_tensor

def df_to_pyg_data(df_path, k=12):
    """Enhanced data preprocessing pipeline"""
    df = load_data(df_path)
    label_col = pick_label_column(df)
    features = build_feature_set(df)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Using label column: {label_col}")
    print(f"Selected features: {len(features)}")
    
    X, y, encoders, scaler, new_features = clean_and_encode(df, features, label_col)
    edge_index, edge_weights = build_edge_index_improved(X, k=k)
    
    print(f"Final data shape: X={X.shape}, y={y.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    return {
        "x": torch.tensor(X, dtype=torch.float),
        "y": torch.tensor(y, dtype=torch.long),
        "edge_index": edge_index,
        "edge_weights": edge_weights,
        "encoders": encoders,
        "scaler": scaler,
        "features_used": new_features,
        "raw_df": df,
        "label_col": label_col
    }
