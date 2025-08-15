from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import torch
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json
from models import HybridGNN, GCNNet, GATNet
from preprocess import build_feature_set
import torch.nn.functional as F

app = Flask(__name__)
app.secret_key = 'lassa_fever_decision_support_2024'

class LassaFeverPredictor:
    def __init__(self, model_path='model.pth', preprocess_path='preproc.pkl'):
        self.model_path = model_path
        self.preprocess_path = preprocess_path
        self.model = None
        self.preprocessor = None
        self.features_used = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load trained model and preprocessor"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.preprocess_path):
                # Load model
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                model_config = checkpoint['model_config']
                
                # Recreate model architecture
                if model_config['model_type'] == 'hybrid':
                    self.model = HybridGNN(
                        in_channels=model_config['in_channels'],
                        hidden_channels=model_config['hidden_channels'],
                        out_channels=model_config['out_channels'],
                        num_layers=model_config['num_layers'],
                        dropout=model_config['dropout'],
                        fusion_method=model_config.get('fusion_method', 'attention')
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
                
                print(f"✅ Model loaded successfully: {model_config['model_type'].upper()}")
                print(f"✅ Features: {len(self.features_used)}")
                return True
            else:
                print("❌ Model files not found. Please train the model first.")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def preprocess_patient_data(self, patient_data):
        """Preprocess single patient data for prediction"""
        try:
            # Create a DataFrame with the patient data
            df = pd.DataFrame([patient_data])
            
            # Fill missing values with defaults
            for feature in self.features_used:
                if feature not in df.columns:
                    df[feature] = 0  # Default value for missing features
            
            # Select only the features used in training
            df_features = df[self.features_used].copy()
            
            # Handle categorical encoding first
            encoders = self.preprocessor['encoders']
            for col in df_features.columns:
                if col in encoders:
                    # Handle unseen categories
                    try:
                        df_features[col] = encoders[col].transform(df_features[col].astype(str))
                    except ValueError:
                        # If unseen category, use the most frequent class (0)
                        df_features[col] = 0
                else:
                    # For numeric columns, ensure they are numeric
                    try:
                        df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)
                    except:
                        df_features[col] = 0
            
            # Ensure all values are numeric before scaling
            df_features = df_features.astype(np.float32)
            
            # Scale features
            scaler = self.preprocessor['scaler']
            X_scaled = scaler.transform(df_features.values)
            
            return torch.tensor(X_scaled, dtype=torch.float).to(self.device)
        
        except Exception as e:
            print(f"❌ Error preprocessing data: {str(e)}")
            print(f"Debug - Patient data keys: {list(patient_data.keys())}")
            print(f"Debug - Features used: {self.features_used[:10]}...")  # Show first 10 features
            return None
    
    def predict(self, patient_data):
        """Make prediction for a single patient"""
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
                
            return {
                'prediction': prediction,
                'confidence': confidence,
                'lassa_probability': lassa_probability,
                'probabilities': probabilities[0].cpu().numpy().tolist()
            }, None
            
        except Exception as e:
            return None, f"Error making prediction: {str(e)}"

# Initialize predictor
predictor = LassaFeverPredictor()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/patient_form')
def patient_form():
    """Patient data input form"""
    return render_template('patient_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        patient_data = {}
        
        # Demographics with validation
        sex = int(request.form.get('sex', 0))
        age = int(request.form.get('age', 0))
        pregnancy = int(request.form.get('pregnancy', 0))
        
        # Validation: Males cannot be pregnant
        if sex == 1 and pregnancy == 1:  # Male (1) and Pregnant (1)
            flash('Error: Male patients cannot be marked as pregnant. Please correct the form.', 'error')
            return redirect(url_for('patient_form'))
        
        patient_data['sex_new2'] = sex
        patient_data['age_recode'] = age
        patient_data['pregnancy'] = pregnancy
        
        # Clinical symptoms
        symptoms = [
            'fever_new', 'headache_new', 'vomiting_new', 'cough_new',
            'bleeding_new', 'weakness_new', 'abdominal_pain', 'backache_new',
            'chest_pain', 'diarrhea_new', 'difficulty_breathing', 'fatigue_weakness',
            'joint_pain_arthritis', 'muscle_pain', 'nausea_new', 'sore_throat'
        ]
        
        for symptom in symptoms:
            patient_data[symptom] = int(request.form.get(symptom, 0))
        
        # Epidemiological factors
        patient_data['contact_with_lassa_case'] = int(request.form.get('contact_history', 0))
        patient_data['travel_history'] = int(request.form.get('travel_history', 0))
        patient_data['rodents_excreta'] = int(request.form.get('rodent_exposure', 0))
        
        # Geographic
        patient_data['State_new'] = request.form.get('state', 'Unknown')
        
        # Make prediction
        result, error = predictor.predict(patient_data)
        
        if error:
            flash(f'Error: {error}', 'error')
            return redirect(url_for('patient_form'))
        
        # Interpret results
        interpretation = interpret_prediction(result, patient_data)
        
        return render_template('results.html', 
                             result=result, 
                             interpretation=interpretation,
                             patient_data=patient_data)
    
    except Exception as e:
        flash(f'Error processing request: {str(e)}', 'error')
        return redirect(url_for('patient_form'))

def interpret_prediction(result, patient_data):
    """Provide clinical interpretation of prediction"""
    prediction = result['prediction']
    lassa_prob = result['lassa_probability']
    confidence = result['confidence']
    
    # Risk level assessment
    if lassa_prob >= 0.7:
        risk_level = "HIGH"
        risk_color = "danger"
        recommendation = "Immediate isolation and confirmatory testing recommended. Treat as suspected Lassa fever case."
    elif lassa_prob >= 0.3:
        risk_level = "MODERATE"
        risk_color = "warning"
        recommendation = "Close monitoring required. Consider Lassa fever testing based on clinical judgment."
    else:
        risk_level = "LOW"
        risk_color = "success"
        recommendation = "Low probability of Lassa fever. Continue standard care and monitor symptoms."
    
    # Key symptoms present
    key_symptoms = []
    symptom_mapping = {
        'fever_new': 'Fever',
        'headache_new': 'Headache',
        'vomiting_new': 'Vomiting',
        'bleeding_new': 'Bleeding',
        'weakness_new': 'Weakness',
        'abdominal_pain': 'Abdominal Pain',
        'diarrhea_new': 'Diarrhea',
        'sore_throat': 'Sore Throat'
    }
    
    for symptom, display_name in symptom_mapping.items():
        if patient_data.get(symptom, 0) == 1:
            key_symptoms.append(display_name)
    
    return {
        'diagnosis': 'POSITIVE for Lassa Fever' if prediction == 1 else 'NEGATIVE for Lassa Fever',
        'risk_level': risk_level,
        'risk_color': risk_color,
        'lassa_probability_percent': round(lassa_prob * 100, 1),
        'confidence_percent': round(confidence * 100, 1),
        'recommendation': recommendation,
        'key_symptoms': key_symptoms,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        result, error = predictor.predict(data)
        
        if error:
            return jsonify({'error': error}), 400
        
        interpretation = interpret_prediction(result, data)
        
        return jsonify({
            'prediction': result,
            'interpretation': interpretation,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Display model information and performance metrics"""
    model_loaded = predictor.model is not None
    
    info = {
        'model_loaded': model_loaded,
        'features_count': len(predictor.features_used) if predictor.features_used else 0,
        'device': str(predictor.device),
        'features_used': predictor.features_used if predictor.features_used else []
    }
    
    return render_template('model_info.html', info=info)

if __name__ == '__main__':
    print("🏥 Starting Lassa Fever Decision Support System...")
    print(f"📊 Model Status: {'✅ Loaded' if predictor.model else '❌ Not Loaded'}")
    app.run(debug=True, host='0.0.0.0', port=5000)
