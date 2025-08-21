# app_enhanced.py - Enhanced Flask application with comprehensive features
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file
import torch
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from models import HybridGNN, GCNNet, GATNet
# from preprocess import build_feature_set  # Not needed for enhanced app
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from werkzeug.utils import secure_filename
from database import DatabaseManager

app = Flask(__name__)
app.secret_key = 'lassa_fever_decision_support_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db_manager = DatabaseManager()

class EnhancedLassaFeverPredictor:
    def __init__(self, model_path='model.pth', preprocess_path='preproc.pkl'):
        self.model_path = model_path
        self.preprocess_path = preprocess_path
        self.model = None
        self.preprocessor = None
        self.features_used = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_info = {}
        self.load_model()
    
    def load_model(self):
        """Load trained model and preprocessor with enhanced error handling"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.preprocess_path):
                # Load model
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                model_config = checkpoint['model_config']
                
                # Store model information
                self.model_info = {
                    'model_loaded': True,  # Key indicator for system status
                    'model_type': model_config['model_type'],
                    'architecture': model_config.get('architecture', 'Unknown'),
                    'training_date': checkpoint.get('training_date', 'Unknown'),
                    'performance_metrics': checkpoint.get('performance_metrics', {}),
                    'feature_count': model_config['in_channels'],
                    'device': str(self.device),
                    'features_count': model_config['in_channels']
                }
                
                # Recreate model architecture with enhanced models
                if model_config['model_type'] == 'hybrid':
                    self.model = HybridGNN(
                        in_channels=model_config['in_channels'],
                        hidden_channels=model_config['hidden_channels'],
                        out_channels=model_config['out_channels'],
                        num_layers=model_config['num_layers'],
                        dropout=model_config['dropout'],
                        fusion_method=model_config.get('fusion_method', 'medical_attention')
                    )
                elif model_config['model_type'] == 'gcn':
                    self.model = GCNNet(
                        in_channels=model_config['in_channels'],
                        hidden_channels=model_config['hidden_channels'],
                        out_channels=model_config['out_channels'],
                        num_layers=model_config['num_layers'],
                        dropout=model_config['dropout']
                    )
                else:  # gat model
                    self.model = GATNet(
                        in_channels=model_config['in_channels'],
                        hidden_channels=model_config['hidden_channels'],
                        out_channels=model_config['out_channels'],
                        num_layers=model_config['num_layers'],
                        dropout=model_config['dropout']
                    )
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                # Load preprocessor
                preprocess_data = joblib.load(self.preprocess_path)
                self.preprocessor = preprocess_data
                self.features_used = preprocess_data['features_used']
                
                print(f"✅ Enhanced model loaded: {model_config['model_type'].upper()}")
                print(f"✅ Features: {len(self.features_used)}")
                return True
            else:
                print("❌ Model files not found. Please train the model first.")
                self.model_info = {
                    'model_loaded': False,
                    'model_type': 'Unknown',
                    'device': str(self.device),
                    'features_count': 0,
                    'features_used': []
                }
                return False
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model_info = {
                'model_loaded': False,
                'model_type': 'Unknown', 
                'device': str(self.device),
                'features_count': 0,
                'features_used': []
            }
            return False
    
    def preprocess_patient_data(self, patient_data):
        """Enhanced preprocessing with proper feature mapping"""
        try:
            # Check if features_used is available
            if self.features_used is None or len(self.features_used) == 0:
                print("❌ No feature list available from trained model")
                return None
            
            print(f"Debug: Model expects {len(self.features_used)} features")
            print(f"Debug: Available patient data keys: {list(patient_data.keys())}")
            
            # Create a properly mapped feature vector
            feature_vector = []
            
            for expected_feature in self.features_used:
                # Map common feature names
                mapped_value = self.map_feature_value(expected_feature, patient_data)
                feature_vector.append(mapped_value)
            
            # Convert to tensor
            X = torch.tensor([feature_vector], dtype=torch.float).to(self.device)
            
            # Apply scaling if available
            scaler = self.preprocessor.get('scaler')
            if scaler:
                try:
                    X_scaled = scaler.transform(X.cpu().numpy())
                    X = torch.tensor(X_scaled, dtype=torch.float).to(self.device)
                    print(f"✅ Applied scaling to features")
                except Exception as e:
                    print(f"⚠️ Scaling failed, using unscaled features: {e}")
            
            print(f"✅ Preprocessed feature vector shape: {X.shape}")
            return X
            
        except Exception as e:
            print(f"❌ Error preprocessing data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def map_feature_value(self, expected_feature, patient_data):
        """Map expected feature names to patient data values"""
        
        # Direct mapping
        if expected_feature in patient_data:
            value = patient_data[expected_feature]
            return self.normalize_feature_value(value, expected_feature)
        
        # Common feature name variations
        feature_mapping = {
            # Demographics
            'age': ['age', 'age_recode'],
            'sex': ['sex', 'sex_new2', 'sex_new'],
            'pregnancy': ['pregnancy', 'pregnancy_status'],
            
            # Symptoms
            'fever_new': ['fever_new', 'fever', 'temperature'],
            'headache_new': ['headache_new', 'headache'],
            'vomiting_new': ['vomiting_new', 'vomiting', 'vomit'],
            'bleeding_new': ['bleeding_new', 'bleeding', 'bleeding_gums', 'bleeding_from_eyes'],
            'weakness_new': ['weakness_new', 'weakness', 'fatigue_weakness'],
            'abdominal_pain': ['abdominal_pain', 'abdominal_pain_new'],
            'diarrhea_new': ['diarrhea_new', 'diarrhea'],
            'sore_throat': ['sore_throat', 'sore_throat_new'],
            'cough_new': ['cough_new', 'cough'],
            'backache_new': ['backache_new', 'backache', 'back_pain'],
            'chest_pain': ['chest_pain', 'chest_pain_new'],
            'difficulty_breathing': ['difficulty_breathing', 'breathing_difficulty'],
            'joint_pain_arthritis': ['joint_pain_arthritis', 'joint_pain'],
            'muscle_pain': ['muscle_pain', 'muscle_pain_new'],
            'nausea_new': ['nausea_new', 'nausea'],
            
            # Epidemiological
            'contact_history': ['contact_history', 'contact_with_lassa_case', 'contact_with_source_case_new'],
            'travel_history': ['travel_history', 'travelled_outside_district'],
            'rodent_exposure': ['rodent_exposure', 'rodent_contact'],
            
            # Geographic
            'state': ['state', 'state_residence_new', 'State_new', 'state_of_residence']
        }
        
        # Try to find a mapping
        for feature_name, variations in feature_mapping.items():
            if expected_feature in variations:
                for variation in variations:
                    if variation in patient_data:
                        value = patient_data[variation]
                        return self.normalize_feature_value(value, expected_feature)
        
        # Default value if no mapping found
        print(f"⚠️ No mapping found for feature: {expected_feature}, using default 0")
        return 0.0

    def normalize_feature_value(self, value, feature_name):
        """Normalize feature values to expected format"""
        try:
            if pd.isna(value) or value is None:
                return 0.0
            
            # Handle string values
            if isinstance(value, str):
                value = value.lower().strip()
                if value in ['yes', 'true', '1', 'positive']:
                    return 1.0
                elif value in ['no', 'false', '0', 'negative']:
                    return 0.0
                elif value in ['unknown', 'unsure', '-1']:
                    return 0.5
                else:
                    # Try to convert to float
                    return float(value)
            
            # Handle numeric values
            if isinstance(value, (int, float)):
                if feature_name in ['age', 'temperature', 'pulse_rate', 'respiratory_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'weight', 'height', 'bmi']:
                    # Continuous features - normalize to 0-1 range
                    if feature_name == 'age':
                        return min(value / 100.0, 1.0)  # Age 0-100
                    elif feature_name == 'temperature':
                        return max(0.0, min(1.0, (value - 35.0) / 10.0))  # Temp 35-45°C
                    else:
                        return float(value)
                else:
                    # Binary features - ensure 0 or 1
                    return 1.0 if value > 0 else 0.0
            
            return 0.0
            
        except Exception as e:
            print(f"⚠️ Error normalizing {feature_name}={value}: {e}")
            return 0.0
    
    def predict_with_explanation(self, patient_data):
        """Enhanced prediction with medical explanation and proper thresholds"""
        if self.model is None:
            return None, "Model not loaded"
        
        try:
            # Preprocess data
            X = self.preprocess_patient_data(patient_data)
            if X is None:
                return None, "Error preprocessing data"
            
            # Create dummy edge index for single node
            edge_index = torch.tensor([[], []], dtype=torch.long).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(X, edge_index)
                probabilities = F.softmax(logits, dim=1)
                
                # Get probability for Lassa positive (class 1)
                lassa_probability = probabilities[0][1].item() if probabilities.shape[1] > 1 else 0.0
                
                # Apply proper medical decision threshold
                MEDICAL_THRESHOLD = 0.75  # 75% probability threshold for positive prediction
                
                if lassa_probability >= MEDICAL_THRESHOLD:
                    prediction = 1  # Lassa positive
                else:
                    prediction = 0  # Lassa negative
                
                # Validate prediction makes medical sense
                prediction = self.validate_medical_prediction(prediction, lassa_probability, patient_data)
                
                # Get enhanced diagnosis using medical knowledge
                enhanced_result = self.enhanced_lassa_diagnosis(patient_data)
                
                # CRITICAL FIX: Ensure confidence is always between 0-100%
                confidence = enhanced_result['confidence']
                
                # Double-check confidence bounds
                if not isinstance(confidence, (int, float)):
                    confidence = float(confidence) if confidence else 0.0
                
                # Ensure confidence is between 0-100
                confidence = max(0.0, min(100.0, confidence))
                
                # Debug output
                print(f"🔍 DEBUG - Raw confidence: {enhanced_result['confidence']}")
                print(f"🔍 DEBUG - Final confidence: {confidence}%")
                
                # Feature importance analysis
                feature_importance = self.analyze_feature_importance(X, patient_data)
                
                return {
                    'prediction': prediction,
                    'confidence': confidence,  # Fixed confidence value
                    'lassa_probability': lassa_probability,
                    'risk_level': enhanced_result['risk_level'],
                    'medical_advice': enhanced_result['diagnosis'],
                    'feature_importance': feature_importance,
                    'model_info': self.model_info,
                    'threshold_used': MEDICAL_THRESHOLD,
                    'raw_logits': logits[0].cpu().numpy().tolist(),
                    'raw_probabilities': probabilities[0].cpu().numpy().tolist(),
                    # Enhanced information
                    'total_score': enhanced_result['total_score'],
                    'epidemiological_score': enhanced_result['epidemiological_score'],
                    'clinical_score': enhanced_result['clinical_score'],
                    'laboratory_score': enhanced_result['laboratory_score'],
                    'differential_diagnosis': enhanced_result['differential_diagnosis'],
                    'enhanced_recommendations': enhanced_result['recommendations']
                }, None
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"

    def validate_medical_prediction(self, prediction, lassa_probability, patient_data):
        """Enhanced medical validation with stricter rules"""
        
        # Rule 1: Single mild symptom should NEVER predict Lassa fever
        mild_symptoms = ['headache_new', 'fatigue_weakness', 'muscle_pain', 'backache_new']
        severe_symptoms = ['bleeding_new', 'bleeding_gums', 'bleeding_from_eyes', 'difficulty_breathing']
        
        # Count symptoms
        mild_count = sum(1 for symptom in mild_symptoms if patient_data.get(symptom, 0) == 1)
        severe_count = sum(1 for symptom in severe_symptoms if patient_data.get(symptom, 0) == 1)
        
        # Rule 2: Check for epidemiological risk factors
        risk_factors = ['contact_history', 'travel_history', 'rodent_exposure', 'contact_with_source_case_new']
        has_risk_factors = any(patient_data.get(factor, 0) == 1 for factor in risk_factors)
        
        # Rule 3: CRITICAL - Single mild symptom with no risk factors = FORCE NEGATIVE
        if mild_count <= 1 and severe_count == 0 and not has_risk_factors:
            print(f"🚨 MEDICAL OVERRIDE: Single mild symptom ({mild_count}) with no risk factors - FORCING NEGATIVE")
            return 0  # Force negative prediction
        
        # Rule 4: Multiple symptoms required for positive prediction
        total_symptoms = mild_count + severe_count
        if total_symptoms < 2 and not has_risk_factors:
            if lassa_probability < 0.9:  # Extremely high threshold
                print(f"⚠️ MEDICAL OVERRIDE: Insufficient symptoms ({total_symptoms}) - forcing negative")
                return 0
        
        # Rule 5: Geographic location alone should not determine diagnosis
        if 'state' in patient_data and patient_data['state'] == 'Ondo':
            if total_symptoms < 2 and not has_risk_factors:
                print(f"⚠️ MEDICAL OVERRIDE: Ondo State location alone insufficient for diagnosis")
                return 0
        
        return prediction

    def get_medical_interpretation(self, lassa_probability, patient_data, final_prediction):
        """Fixed medical interpretation that considers the final validated prediction"""
        
        # If the medical validation forced a negative prediction, adjust risk accordingly
        if final_prediction == 0:  # Negative prediction
            # For negative predictions, risk should be LOW or VERY LOW
            if lassa_probability < 0.3:
                risk_level = "VERY LOW"
                advice = "✅ VERY LOW RISK: Routine care. Continue monitoring for symptom changes."
            elif lassa_probability < 0.5:
                risk_level = "LOW"
                advice = "📋 LOW RISK: Standard monitoring. Test if symptoms worsen."
            else:
                risk_level = "MODERATE"
                advice = "⚡ MODERATE RISK: Enhanced monitoring. Consider retesting if symptoms persist."
        else:  # Positive prediction
            # Apply stricter thresholds for positive predictions
            if lassa_probability >= 0.9:
                risk_level = "VERY HIGH"
                advice = "🚨 URGENT: Immediate isolation and confirmatory testing required."
            elif lassa_probability >= 0.75:
                risk_level = "HIGH"
                advice = "⚠️ HIGH RISK: Isolation recommended. Perform confirmatory testing."
            elif lassa_probability >= 0.6:
                risk_level = "MODERATE"
                advice = "⚡ MODERATE RISK: Enhanced monitoring required."
            else:
                risk_level = "LOW"
                advice = "📋 LOW RISK: Standard monitoring."
        
        return risk_level, advice
    
    def analyze_feature_importance(self, X, patient_data):
        """Analyze which features contributed most to the prediction"""
        # Simplified feature importance based on input values
        important_features = []
        
        # Map some key medical features
        key_features = {
            'fever_new': 'Fever',
            'bleeding_gums': 'Bleeding gums',
            'headache_new': 'Headache',
            'contact_with_source_case_new': 'Contact with confirmed case',
            'state_residence_new': 'Geographic location'
        }
        
        for feature, display_name in key_features.items():
            if feature in patient_data and patient_data[feature]:
                important_features.append({
                    'feature': display_name,
                    'value': patient_data[feature],
                    'impact': 'High' if patient_data[feature] == 1 else 'Moderate'
                })
        
        return important_features[:5]  # Return top 5
    
    def assess_uncertainty(self, patient_data):
        """Assess uncertainty in epidemiological data and provide recommendations"""
        uncertainty_factors = {
            'has_uncertainty': False,
            'warning': '',
            'recommend_higher_vigilance': False,
            'missing_factors': []
        }
        
        # Check for unknown/missing epidemiological factors
        epi_factors = {
            'contact_with_source_case_new': 'Contact with confirmed Lassa case',
            'direct_contact_probable_case': 'Direct contact with probable case',
            'travelled_outside_district': 'Travel to endemic areas',
            'travel_history': 'Recent travel history',
            'rodent_exposure': 'Exposure to rodents/excreta',
            'burial_of_case': 'Participation in burial of suspected case'
        }
        
        missing_count = 0
        for factor, description in epi_factors.items():
            value = patient_data.get(factor, -1)  # -1 indicates unknown
            if value == -1 or value == 'unknown':
                uncertainty_factors['missing_factors'].append(description)
                missing_count += 1
        
        # Assess uncertainty level
        if missing_count > 0:
            uncertainty_factors['has_uncertainty'] = True
            
            if missing_count >= 3:
                uncertainty_factors['recommend_higher_vigilance'] = True
                uncertainty_factors['warning'] = f"Critical epidemiological data missing ({missing_count} factors unknown). Consider higher risk category."
            elif missing_count >= 2:
                uncertainty_factors['recommend_higher_vigilance'] = True
                uncertainty_factors['warning'] = f"Important epidemiological data missing ({missing_count} factors unknown). Increased vigilance recommended."
            else:
                uncertainty_factors['warning'] = f"Some epidemiological data missing ({missing_count} factor unknown). Monitor closely."
        
        return uncertainty_factors
    
    def escalate_risk_level(self, current_risk):
        """Escalate risk level due to uncertainty"""
        escalation_map = {
            'VERY LOW': 'LOW',
            'LOW': 'MODERATE', 
            'MODERATE': 'HIGH',
            'HIGH': 'VERY HIGH',
            'VERY HIGH': 'VERY HIGH'  # Already at maximum
        }
        return escalation_map.get(current_risk, current_risk)
    
    def batch_predict(self, df):
        """Enhanced batch prediction with better error handling"""
        results = []
        
        for idx, row in df.iterrows():
            try:
                patient_data = row.to_dict()
                
                # Ensure required fields are present
                required_fields = ['age', 'sex']
                for field in required_fields:
                    if field not in patient_data or pd.isna(patient_data[field]):
                        results.append({
                            'patient_id': patient_data.get('patient_id', f'Patient_{idx}'),
                            'error': f'Missing required field: {field}'
                        })
                        continue
                
                # Convert data types
                patient_data['age'] = float(patient_data['age'])
                patient_data['sex'] = int(patient_data['sex']) if patient_data['sex'] in [0, 1, '0', '1'] else 0
                
                # Handle missing symptom fields
                symptom_fields = [
                    'fever_new', 'headache_new', 'vomiting_new', 'bleeding_new', 'weakness_new',
                    'abdominal_pain', 'diarrhea_new', 'sore_throat', 'cough_new', 'backache_new',
                    'chest_pain', 'difficulty_breathing', 'fatigue_weakness', 'joint_pain_arthritis',
                    'muscle_pain', 'nausea_new', 'bleeding_gums', 'bleeding_from_eyes'
                ]
                
                for field in symptom_fields:
                    if field not in patient_data or pd.isna(patient_data[field]):
                        patient_data[field] = 0
                    else:
                        patient_data[field] = int(patient_data[field]) if patient_data[field] in [0, 1, '0', '1'] else 0
                
                # Handle epidemiological fields (support unknown values)
                epi_fields = [
                    'contact_with_source_case_new', 'direct_contact_probable_case', 
                    'travelled_outside_district', 'contact_history', 'travel_history', 
                    'rodent_exposure', 'burial_of_case'
                ]
                
                for field in epi_fields:
                    if field not in patient_data or pd.isna(patient_data[field]):
                        patient_data[field] = -1  # Unknown
                    else:
                        value = patient_data[field]
                        if value in [-1, '-1', 'unknown', 'Unknown', '']:
                            patient_data[field] = -1
                        else:
                            patient_data[field] = int(value) if value in [0, 1, '0', '1'] else -1
                
                # Add default geographic data
                if 'state_residence_new' not in patient_data or pd.isna(patient_data['state_residence_new']):
                    patient_data['state_residence_new'] = 'UNKNOWN'
                
                # Add default medical values
                medical_defaults = {
                    'pregnancy': 0,
                    'area_type': 'Urban',
                    'temperature': 37.0,
                    'pulse_rate': 80,
                    'respiratory_rate': 16,
                    'blood_pressure_systolic': 120,
                    'blood_pressure_diastolic': 80,
                    'weight': 70.0,
                    'height': 170.0,
                    'bmi': 24.2
                }
                
                for field, default_value in medical_defaults.items():
                    if field not in patient_data or pd.isna(patient_data[field]):
                        patient_data[field] = default_value
                
                result, error = self.predict_with_explanation(patient_data)
                
                if result:
                    results.append({
                        'patient_id': patient_data.get('patient_id', f'Patient_{idx}'),
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'lassa_probability': result['lassa_probability'],
                        'risk_level': result['risk_level']
                    })
                else:
                    results.append({
                        'patient_id': patient_data.get('patient_id', f'Patient_{idx}'),
                        'error': error or 'Unknown prediction error'
                    })
                    
            except Exception as e:
                results.append({
                    'patient_id': patient_data.get('patient_id', f'Patient_{idx}') if 'patient_data' in locals() else f'Patient_{idx}',
                    'error': f'Processing error: {str(e)}'
                })
        
        return results

    def get_ai_analysis(self, prediction, lassa_probability, risk_level, patient_data):
        """AI analysis that matches the final diagnosis"""
        
        # Check symptom count
        mild_symptoms = ['headache_new', 'fatigue_weakness', 'muscle_pain', 'backache_new']
        severe_symptoms = ['bleeding_new', 'bleeding_gums', 'bleeding_from_eyes', 'difficulty_breathing']
        
        mild_count = sum(1 for symptom in mild_symptoms if patient_data.get(symptom, 0) == 1)
        severe_count = sum(1 for symptom in severe_symptoms if patient_data.get(symptom, 0) == 1)
        total_symptoms = mild_count + severe_count
        
        # For positive predictions, match the diagnosis
        if prediction == 1:  # Positive prediction
            if risk_level == "VERY HIGH":
                return "🚨 URGENT: Immediate isolation and confirmatory testing required."
            elif risk_level == "HIGH":
                return "⚠️ HIGH RISK: Isolation recommended. Perform confirmatory testing."
            else:
                return "⚡ MODERATE RISK: Enhanced monitoring required."
        else:  # Negative prediction
            if total_symptoms <= 1 and severe_count == 0:
                return "✅ LOW RISK: Single mild symptom with no risk factors. Continue routine monitoring."
            elif total_symptoms >= 2:
                return "⚡ MODERATE VIGILANCE: Multiple symptoms present. Enhanced monitoring recommended."
            else:
                return "📋 STANDARD MONITORING: Continue monitoring. Seek care if symptoms worsen."

    def calculate_epidemiological_score(self, patient_data):
        """Calculate epidemiological risk score based on medical knowledge"""
        
        score = 0
        risk_factors = {
            # Geographic risk (West Africa endemic)
            'geographic_risk': {
                'nigeria': 3, 'sierra_leone': 3, 'liberia': 3, 'guinea': 3,
                'ghana': 2, 'mali': 2, 'senegal': 1
            },
            
            # Seasonal risk (dry season Dec-Apr)
            'seasonal_risk': {
                'december': 2, 'january': 2, 'february': 2, 'march': 2, 'april': 2,
                'may': 1, 'june': 0, 'july': 0, 'august': 0, 'september': 0, 'october': 0, 'november': 1
            },
            
            # Exposure risk
            'rodent_exposure': 4,  # Mastomys natalensis contact
            'contact_with_source_case_new': 5,  # Human-to-human transmission
            'healthcare_setting': 3,  # Healthcare worker or patient
            'rural_setting': 2,  # Higher rodent exposure
            'food_water_contamination': 3  # Rodent urine/feces in food/water
        }
        
        # Calculate geographic score
        state = patient_data.get('state', '').lower()
        if state in risk_factors['geographic_risk']:
            score += risk_factors['geographic_risk'][state]
        
        # Calculate seasonal score
        from datetime import datetime
        current_month = datetime.now().strftime('%B').lower()
        if current_month in risk_factors['seasonal_risk']:
            score += risk_factors['seasonal_risk'][current_month]
        
        # Calculate exposure score
        for factor, points in risk_factors.items():
            if factor not in ['geographic_risk', 'seasonal_risk']:
                if patient_data.get(factor, 0) == 1:
                    score += points
        
        return min(score, 10)  # Cap at 10

    def calculate_clinical_lassa_score(self, patient_data):
        """Calculate clinical score based on Lassa-specific features"""
        
        score = 0
        
        # Early stage symptoms (nonspecific, lower weight)
        early_symptoms = {
            'fever_new': 2, 'weakness': 1, 'sore_throat': 2, 'headache_new': 1,
            'chest_pain': 2, 'backache_new': 1, 'nausea_new': 1, 'vomiting_new': 2
        }
        
        # Progressive Lassa-specific features (higher weight)
        progressive_features = {
            'persistent_sore_throat': 4,  # Pharyngeal ulcers/exudates
            'facial_swelling': 5,  # Facial edema
            'conjunctivitis': 3,  # Red eyes
            'retrosternal_pain': 4,  # Retro-sternal chest pain
            'proteinuria': 5,  # Protein in urine
            'hearing_loss': 4,  # Neurological involvement
            'tremors': 3,  # Neurological involvement
            'encephalitis': 5  # Severe neurological
        }
        
        # Hemorrhagic signs (mild in Lassa)
        hemorrhagic_signs = {
            'bleeding_new': 3, 'bleeding_gums': 3, 'bleeding_from_eyes': 3,
            'petechiae': 3, 'hematuria': 4, 'mucosal_bleeding': 3
        }
        
        # Calculate early symptom score
        for symptom, points in early_symptoms.items():
            if patient_data.get(symptom, 0) == 1:
                score += points
        
        # Calculate progressive feature score
        for feature, points in progressive_features.items():
            if patient_data.get(feature, 0) == 1:
                score += points
        
        # Calculate hemorrhagic score
        for sign, points in hemorrhagic_signs.items():
            if patient_data.get(sign, 0) == 1:
                score += points
        
        return min(score, 15)  # Cap at 15

    def integrate_laboratory_findings(self, patient_data):
        """Integrate laboratory findings for differential diagnosis"""
        
        lab_score = 0
        differential_diagnosis = []
        
        # Malaria test results
        malaria_test = patient_data.get('malaria_test', 'unknown')
        if malaria_test == 'negative':
            lab_score += 3  # Negative malaria increases Lassa suspicion
            differential_diagnosis.append("Malaria excluded - increases Lassa suspicion")
        elif malaria_test == 'positive':
            lab_score -= 2  # Positive malaria decreases Lassa probability
            differential_diagnosis.append("Malaria positive - consider co-infection or alternative diagnosis")
        
        # CBC findings
        if patient_data.get('leukopenia', 0) == 1:
            lab_score += 2
            differential_diagnosis.append("Leukopenia - consistent with Lassa fever")
        
        if patient_data.get('thrombocytopenia', 0) == 1:
            lab_score += 2
            differential_diagnosis.append("Thrombocytopenia - consistent with Lassa fever")
        
        # Liver function
        ast_level = patient_data.get('ast_level', 0)
        if ast_level > 150:
            lab_score += 4
            differential_diagnosis.append(f"AST >150 IU/L - strong marker for severe Lassa")
        elif ast_level > 100:
            lab_score += 2
            differential_diagnosis.append(f"AST >100 IU/L - suggestive of Lassa")
        
        return lab_score, differential_diagnosis

    def enhanced_lassa_diagnosis(self, patient_data):
        """Enhanced diagnosis using medical knowledge"""
        
        # Calculate all scores
        epi_score = self.calculate_epidemiological_score(patient_data)
        clinical_score = self.calculate_clinical_lassa_score(patient_data)
        lab_score, differential_diagnosis = self.integrate_laboratory_findings(patient_data)
        
        # Total risk score (0-30)
        total_score = epi_score + clinical_score + lab_score
        
        # DEBUG: Print scores to identify the issue
        print(f" DEBUG - epi_score: {epi_score}, clinical_score: {clinical_score}, lab_score: {lab_score}")
        print(f"🔍 DEBUG - total_score: {total_score}")
        
        # FIXED: Proper confidence calculation with bounds
        if total_score >= 20:
            diagnosis = "HIGHLY SUGGESTIVE of Lassa Fever"
            risk_level = "VERY HIGH"
            confidence = 0.95  # Fixed: Cap at 95%
        elif total_score >= 15:
            diagnosis = "SUSPICIOUS for Lassa Fever"
            risk_level = "HIGH"
            confidence = 0.85  # Fixed: Cap at 85%
        elif total_score >= 10:
            diagnosis = "POSSIBLE Lassa Fever"
            risk_level = "MODERATE"
            confidence = 0.70  # Fixed: Cap at 70%
        elif total_score >= 5:
            diagnosis = "LOW PROBABILITY of Lassa Fever"
            risk_level = "LOW"
            confidence = 0.50  # Fixed: Cap at 50%
        else:
            diagnosis = "UNLIKELY to be Lassa Fever"
            risk_level = "VERY LOW"
            confidence = 0.25  # Fixed: Cap at 25%
        
        # CRITICAL: Ensure confidence is always between 0-100
        confidence = max(0.0, min(100.0, float(confidence)))
        
        print(f"🔍 DEBUG - Final confidence: {confidence}%")
        
        return {
            'diagnosis': diagnosis,
            'risk_level': risk_level,
            'confidence': confidence,
            'total_score': total_score,
            'epidemiological_score': epi_score,
            'clinical_score': clinical_score,
            'laboratory_score': lab_score,
            'differential_diagnosis': differential_diagnosis,
            'recommendations': self.get_enhanced_recommendations(total_score, epi_score, clinical_score)
        }

    def get_enhanced_recommendations(self, total_score, epi_score, clinical_score):
        """Get enhanced medical recommendations based on scores"""
        
        recommendations = []
        
        if total_score >= 20:
            recommendations.extend([
                "🚨 IMMEDIATE ISOLATION required",
                "🔬 Confirmatory RT-PCR testing urgent",
                " Consider Ribavirin therapy",
                "📞 Contact health authorities immediately",
                "👥 Contact tracing for recent exposures"
            ])
        elif total_score >= 15:
            recommendations.extend([
                "⚠️ Isolation recommended",
                "🔬 RT-PCR testing required",
                "📊 Enhanced monitoring",
                " Hospital admission consideration"
            ])
        elif total_score >= 10:
            recommendations.extend([
                "⚡ Enhanced monitoring",
                "🔬 Consider Lassa testing",
                "📋 Watch for symptom progression",
                "🏥 Outpatient monitoring"
            ])
        elif total_score >= 5:
            recommendations.extend([
                "📋 Standard monitoring",
                "🔍 Watch for new symptoms",
                "🏥 Routine care",
                "📞 Seek care if symptoms worsen"
            ])
        else:
            recommendations.extend([
                "✅ Routine care",
                "📋 Standard monitoring",
                "🔍 Watch for symptom changes",
                "🏥 No special precautions needed"
            ])
        
        # Add epidemiological recommendations
        if epi_score >= 5:
            recommendations.append("🌍 High endemic area - increased vigilance recommended")
        
        # Add clinical recommendations
        if clinical_score >= 8:
            recommendations.append("🏥 Multiple symptoms - consider specialist consultation")
        
        return recommendations

    def debug_score_calculation(self, patient_data):
        """Debug method to see what's causing high scores"""
        
        print("🔍 DEBUGGING SCORE CALCULATION:")
        print(f"Patient data keys: {list(patient_data.keys())}")
        
        # Check epidemiological factors
        epi_factors = ['rodent_exposure', 'contact_with_source_case_new', 'state']
        for factor in epi_factors:
            value = patient_data.get(factor, 'NOT_FOUND')
            print(f"  {factor}: {value}")
        
        # Check clinical factors
        clinical_factors = ['fever_new', 'headache_new', 'bleeding_new', 'weakness', 'sore_throat']
        for factor in clinical_factors:
            value = patient_data.get(factor, 'NOT_FOUND')
            print(f"  {factor}: {value}")
        
        # Calculate individual scores
        epi_score = self.calculate_epidemiological_score(patient_data)
        clinical_score = self.calculate_clinical_lassa_score(patient_data)
        
        print(f"  Epidemiological score: {epi_score}")
        print(f"  Clinical score: {clinical_score}")
        print(f"  Total score: {epi_score + clinical_score}")

    def validate_confidence_format(self, confidence_value):
        """Ensure confidence is properly formatted and within bounds"""
        
        try:
            # Convert to float
            if isinstance(confidence_value, str):
                # Remove any % symbols and convert
                confidence_value = float(confidence_value.replace('%', ''))
            else:
                confidence_value = float(confidence_value)
            
            # Ensure it's between 0-100
            confidence_value = max(0.0, min(100.0, confidence_value))
            
            # Round to 1 decimal place for display
            confidence_value = round(confidence_value, 1)
            
            return confidence_value
            
        except (ValueError, TypeError):
            print(f"⚠️ Invalid confidence value: {confidence_value}, defaulting to 50.0%")
            return 50.0

# Initialize predictor
predictor = EnhancedLassaFeverPredictor()

@app.route('/')
def index():
    """Enhanced home page with dashboard"""
    stats = db_manager.get_prediction_statistics()
    return render_template('dashboard.html', stats=stats)

@app.route('/predict')
def predict_form():
    """Individual prediction form"""
    return render_template('patient_form.html')
    
@app.route('/single_prediction')
def single_prediction():
    """Single prediction form page"""
    return render_template('patient_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle individual prediction"""
    try:
        # Get form data
        patient_data = {}
        
        # Demographics
        patient_data['patient_id'] = request.form.get('patient_id', '')
        patient_data['age'] = float(request.form.get('age', 0))
        patient_data['sex'] = int(request.form.get('sex', 0))
        patient_data['pregnancy'] = int(request.form.get('pregnancy', 0))
        
        # Symptoms (checkboxes return '1' if checked, None if not)
        symptoms = [
            'fever_new', 'headache_new', 'vomiting_new', 'bleeding_new', 'weakness_new',
            'abdominal_pain', 'diarrhea_new', 'sore_throat', 'cough_new', 'backache_new',
            'chest_pain', 'difficulty_breathing', 'fatigue_weakness', 'joint_pain_arthritis',
            'muscle_pain', 'nausea_new'
        ]
        
        for symptom in symptoms:
            patient_data[symptom] = 1 if request.form.get(symptom) == '1' else 0
        
        # Epidemiological factors (now supports -1 for unknown)
        patient_data['contact_history'] = int(request.form.get('contact_history', 0))
        patient_data['travel_history'] = int(request.form.get('travel_history', 0))
        patient_data['rodent_exposure'] = int(request.form.get('rodent_exposure', 0))
        
        # Map form fields to model expected fields
        patient_data['contact_with_source_case_new'] = patient_data['contact_history']
        patient_data['travelled_outside_district'] = patient_data['travel_history']
        patient_data['direct_contact_probable_case'] = patient_data.get('contact_history', 0)
        
        # Add additional required fields with defaults
        patient_data['burial_of_case'] = 0
        patient_data['area_type'] = 'Urban'  # Default
        patient_data['bleeding_gums'] = patient_data.get('bleeding_new', 0)
        patient_data['bleeding_from_eyes'] = patient_data.get('bleeding_new', 0)
        
        # Ensure all common medical fields are present
        medical_defaults = {
            'temperature': 37.0,  # Normal temperature
            'pulse_rate': 80,     # Normal pulse
            'respiratory_rate': 16, # Normal respiratory rate
            'blood_pressure_systolic': 120,
            'blood_pressure_diastolic': 80,
            'weight': 70.0,
            'height': 170.0,
            'bmi': 24.2
        }
        
        for field, default_value in medical_defaults.items():
            if field not in patient_data:
                patient_data[field] = default_value
        
        # Geographic
        patient_data['state_residence_new'] = request.form.get('state', '')
        
        # Make prediction
        result, error = predictor.predict_with_explanation(patient_data)
        
        if error:
            flash(f'Prediction error: {error}', 'error')
            return redirect(url_for('predict_form'))
        
        # Save to database
        db_manager.save_prediction(patient_data, result)
        
        return render_template('prediction_result.html', 
                             patient_data=patient_data, 
                             result=result)
    
    except Exception as e:
        flash(f'Error processing prediction: {str(e)}', 'error')
        return redirect(url_for('predict_form'))

@app.route('/batch_predict')
def batch_predict_form():
    """Batch prediction upload form"""
    return render_template('batch_predict.html')

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions"""
    try:
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('batch_predict_form'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('batch_predict_form'))
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and process CSV
            df = pd.read_csv(filepath)
            batch_name = request.form.get('batch_name', filename)
            
            # Make predictions
            results = predictor.batch_predict(df)
            
            # Save batch results
            db_manager.save_batch_prediction(batch_name, results, filepath)
            
            return render_template('batch_results.html', 
                                 results=results, 
                                 batch_name=batch_name)
        else:
            flash('Please upload a CSV file', 'error')
            return redirect(url_for('batch_predict_form'))
    
    except Exception as e:
        flash(f'Error processing batch: {str(e)}', 'error')
        return redirect(url_for('batch_predict_form'))

@app.route('/dashboard')
def dashboard():
    """Enhanced dashboard with analytics"""
    stats = db_manager.get_prediction_statistics()
    recent_predictions = db_manager.get_recent_predictions(20)
    
    # Generate visualization
    chart_data = generate_dashboard_charts(recent_predictions)
    
    return render_template('dashboard.html', 
                         stats=stats, 
                         recent_predictions=recent_predictions.to_dict('records'),
                         chart_data=chart_data)

@app.route('/model_info')
def model_info():
    """Model information and performance metrics"""
    return render_template('model_info.html', model_info=predictor.model_info)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        patient_data = request.json
        result, error = predictor.predict_with_explanation(patient_data)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_dashboard_charts(df):
    """Generate charts for dashboard"""
    if df.empty:
        return {'prediction_trend': '', 'risk_distribution': ''}
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Prediction trend over time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Convert prediction_date to datetime
    df['prediction_date'] = pd.to_datetime(df['prediction_date'])
    daily_predictions = df.groupby(df['prediction_date'].dt.date).size()
    
    ax1.plot(daily_predictions.index, daily_predictions.values, marker='o')
    ax1.set_title('Daily Predictions Trend')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Predictions')
    ax1.tick_params(axis='x', rotation=45)
    
    # Risk level distribution
    risk_counts = df['risk_level'].value_counts()
    colors = ['#ff4444', '#ff8800', '#ffcc00', '#88cc00', '#44cc44']
    ax2.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=colors)
    ax2.set_title('Risk Level Distribution')
    
    plt.tight_layout()
    
    # Convert to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)
    chart_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return {'dashboard_chart': chart_data}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
