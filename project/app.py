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
from .models import HybridGNN, GCNNet, GATNet
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
        """Enhanced preprocessing with better error handling"""
        try:
            # Check if features_used is available
            if self.features_used is None or len(self.features_used) == 0:
                print("❌ No feature list available from trained model")
                return None
            
            print(f"Debug: Model expects {len(self.features_used)} features")
            
            # Check scaler expected features
            scaler = self.preprocessor.get('scaler')
            if scaler and hasattr(scaler, 'n_features_in_'):
                scaler_features = scaler.n_features_in_
                print(f"Debug: Scaler expects {scaler_features} features")
                
                if len(self.features_used) != scaler_features:
                    print(f"⚠️ Model/Scaler mismatch detected. Using fallback preprocessing.")
                    # Use a simple feature mapping approach
                    return self._fallback_preprocessing(patient_data)
            
            # Standard preprocessing (same as original app.py)
            df = pd.DataFrame([patient_data])
            
            # Fill missing values with defaults
            for feature in self.features_used:
                if feature not in patient_data:
                    patient_data[feature] = 0
            
            # Select only the features used in training
            df_features = pd.DataFrame([patient_data])[self.features_used].copy()
            
            # Handle categorical encoding
            encoders = self.preprocessor.get('encoders', {})
            for col in df_features.columns:
                if col in encoders:
                    try:
                        df_features[col] = encoders[col].transform(df_features[col].astype(str))
                    except ValueError:
                        df_features[col] = 0
                else:
                    try:
                        # Handle epidemiological uncertainty
                        if col in ['contact_with_source_case_new', 'travelled_outside_district', 
                                 'contact_history', 'travel_history', 'rodent_exposure']:
                            df_features[col] = df_features[col].apply(lambda x: 0.5 if x == -1 else float(x) if pd.notnull(x) else 0.0)
                        else:
                            df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)
                    except:
                        df_features[col] = 0
            
            df_features = df_features.astype(np.float32)
            X_scaled = scaler.transform(df_features.values)
            
            return torch.tensor(X_scaled, dtype=torch.float).to(self.device)
        
        except Exception as e:
            print(f"❌ Error preprocessing data: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            print(f"❌ Patient data keys: {list(patient_data.keys()) if patient_data else 'None'}")
            print(f"❌ Features used: {self.features_used[:5] if self.features_used else 'None'}...")
            import traceback
            traceback.print_exc()
            return None
    
    def _fallback_preprocessing(self, patient_data):
        """Fallback preprocessing when model/scaler mismatch occurs"""
        try:
            print("Using fallback preprocessing - creating basic feature vector")
            
            # Create a basic feature vector from available patient data
            basic_features = [
                'age', 'sex', 'fever_new', 'headache_new', 'vomiting_new', 
                'bleeding_new', 'weakness_new', 'abdominal_pain', 'diarrhea_new',
                'contact_history', 'travel_history', 'rodent_exposure'
            ]
            
            feature_vector = []
            for feature in basic_features:
                if feature in patient_data:
                    value = patient_data[feature]
                    if feature in ['contact_history', 'travel_history', 'rodent_exposure'] and value == -1:
                        feature_vector.append(0.5)  # Unknown
                    elif feature == 'sex' and isinstance(value, str):
                        feature_vector.append(1.0 if value.lower() == 'male' else 0.0)
                    else:
                        try:
                            feature_vector.append(float(value))
                        except:
                            feature_vector.append(0.0)
                else:
                    feature_vector.append(0.0)
            
            # Pad or truncate to match model expectations
            while len(feature_vector) < len(self.features_used):
                feature_vector.append(0.0)
            feature_vector = feature_vector[:len(self.features_used)]
            
            print(f"Debug: Created fallback feature vector with {len(feature_vector)} features")
            return torch.tensor([feature_vector], dtype=torch.float).to(self.device)
        
        except Exception as e:
            print(f"❌ Error in fallback preprocessing: {str(e)}")
            return None
    
    def predict_with_explanation(self, patient_data):
        """Enhanced prediction with medical explanation"""
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
                prediction = torch.argmax(logits, dim=1).item()
                confidence = probabilities[0][prediction].item()
                
                # Get probability for Lassa positive (class 1)
                lassa_probability = probabilities[0][1].item() if probabilities.shape[1] > 1 else 0.0
                
                # Determine risk level and medical interpretation
                risk_level, medical_advice = self.get_medical_interpretation(
                    lassa_probability, patient_data
                )
                
                # Feature importance analysis
                feature_importance = self.analyze_feature_importance(X, patient_data)
                
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'lassa_probability': lassa_probability,
                    'risk_level': risk_level,
                    'medical_advice': medical_advice,
                    'feature_importance': feature_importance,
                    'model_info': self.model_info
                }, None
        
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def get_medical_interpretation(self, lassa_probability, patient_data):
        """Provide medical interpretation based on probability and symptoms with uncertainty handling"""
        # Check for missing epidemiological data
        uncertainty_factors = self.assess_uncertainty(patient_data)
        
        if lassa_probability >= 0.8:
            risk_level = "VERY HIGH"
            advice = "🚨 URGENT: Immediate isolation and confirmatory testing required. High probability of Lassa fever."
        elif lassa_probability >= 0.6:
            risk_level = "HIGH"
            advice = "⚠️ HIGH RISK: Isolation recommended. Perform confirmatory testing and monitor closely."
        elif lassa_probability >= 0.4:
            risk_level = "MODERATE"
            advice = "⚡ MODERATE RISK: Enhanced monitoring required. Consider testing if symptoms persist."
        elif lassa_probability >= 0.2:
            risk_level = "LOW"
            advice = "📋 LOW RISK: Standard monitoring. Test if symptoms worsen or new symptoms appear."
        else:
            risk_level = "VERY LOW"
            advice = "✅ VERY LOW RISK: Routine care. Monitor for symptom changes."
        
        # Add uncertainty warnings
        if uncertainty_factors['has_uncertainty']:
            advice += f" ⚠️ UNCERTAINTY: {uncertainty_factors['warning']}"
            if uncertainty_factors['recommend_higher_vigilance']:
                risk_level = self.escalate_risk_level(risk_level)
                advice += " Risk level elevated due to missing epidemiological data."
        
        # Add symptom-specific advice
        symptoms = patient_data.get('symptoms', {})
        if symptoms.get('fever_new', 0) and symptoms.get('bleeding_gums', 0):
            advice += " Note: Fever + bleeding symptoms increase concern for viral hemorrhagic fever."
        
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
