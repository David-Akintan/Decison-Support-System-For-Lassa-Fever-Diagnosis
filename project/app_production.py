# app_production.py - Production-ready Flask app
import os
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import torch
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json
from models import HybridGNN, GCNNet, GATNet
from preprocess import build_feature_set
import torch.nn.functional as F
import logging

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'lassa_fever_decision_support_2024_production')

class LassaFeverPredictor:
    def __init__(self, model_path='model.pth', preprocess_path='preproc.pkl'):
        self.model_path = model_path
        self.preprocess_path = preprocess_path
        self.model = None
        self.preprocessor = None
        self.features_used = None
        self.device = torch.device("cpu")  # Use CPU for production stability
        self.load_model()
    
    def load_model(self):
        """Load trained model and preprocessor with production error handling"""
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
                else:  # gat
                    self.model = GATNet(
                        in_channels=model_config['in_channels'],
                        hidden_channels=model_config['hidden_channels'],
                        out_channels=model_config['out_channels'],
                        num_layers=model_config['num_layers'],
                        dropout=model_config['dropout']
                    )
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                # Load preprocessor
                self.preprocessor = joblib.load(self.preprocess_path)
                self.features_used = self.preprocessor.get('features_used', [])
                
                logger.info("✅ Model and preprocessor loaded successfully!")
                
            else:
                logger.warning("⚠️ Model files not found. Running in demo mode.")
                self.model = None
                self.preprocessor = None
                self.features_used = []
                
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            self.model = None
            self.preprocessor = None
            self.features_used = []
    
    def preprocess_patient_data(self, patient_data):
        """Preprocess single patient data for prediction"""
        if self.preprocessor is None:
            return None
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame([patient_data])
            
            # Build feature set
            df_features = build_feature_set(df)
            df_features = df[self.features_used].copy()
            
            # Handle categorical encoding
            encoders = self.preprocessor.get('encoders', {})
            for col in df_features.columns:
                if col in encoders:
                    try:
                        df_features[col] = encoders[col].transform(df_features[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        df_features[col] = 0
            
            # Ensure all features are numerical
            df_features = df_features.astype(np.float32)
            
            # Scale features
            scaler = self.preprocessor.get('scaler', None)
            if scaler:
                X_scaled = scaler.transform(df_features.values)
            else:
                X_scaled = df_features.values
            
            return torch.tensor(X_scaled, dtype=torch.float).to(self.device)
            
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return None
    
    def predict(self, patient_data):
        """Make prediction for a single patient with production error handling"""
        if self.model is None or self.preprocessor is None:
            return {
                'error': 'Model not available. Please contact system administrator.',
                'probability': 0.0,
                'prediction': 'Unknown',
                'confidence': 'N/A'
            }
            
        try:
            # Preprocess data
            X = self.preprocess_patient_data(patient_data)
            if X is None:
                return {
                    'error': 'Error processing patient data',
                    'probability': 0.0,
                    'prediction': 'Error',
                    'confidence': 'N/A'
                }
            
            # Create dummy edge index for single node
            edge_index = torch.tensor([[], []], dtype=torch.long).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(X, edge_index)
                probabilities = F.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                
                lassa_probability = probabilities[0][1].item()
                confidence = torch.max(probabilities, dim=1)[0].item()
                
            return {
                'prediction': prediction,
                'confidence': confidence,
                'lassa_probability': lassa_probability,
                'probabilities': probabilities[0].cpu().numpy().tolist()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'error': f'Prediction failed: {str(e)}',
                'probability': 0.0,
                'prediction': 'Error',
                'confidence': 'N/A'
            }

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
        # Collect patient data from form
        patient_data = {}
        
        # Basic demographics
        patient_data['age'] = int(request.form.get('age', 0))
        patient_data['sex_new2'] = request.form.get('gender', 'Unknown')
        
        # Symptoms (convert checkboxes to binary)
        symptoms = ['fever_new', 'headache_new', 'vomiting_new', 'cough_new', 
                   'bleeding_new', 'weakness_new', 'diarrhea_new', 'muscle_pain']
        
        for symptom in symptoms:
            patient_data[symptom] = 1 if request.form.get(symptom) == 'on' else 0
        
        # Contact history
        patient_data['contact_history'] = 1 if request.form.get('contact_history') == 'on' else 0
        patient_data['travel_history'] = 1 if request.form.get('travel_history') == 'on' else 0
        
        # Location
        patient_data['State_new'] = request.form.get('state', 'Unknown')
        
        # Make prediction
        result = predictor.predict(patient_data)
        
        if 'error' in result:
            flash(f'Error: {result["error"]}', 'error')
            return redirect(url_for('patient_form'))
        
        # Interpret results
        interpretation = interpret_prediction(result, patient_data)
        
        return render_template('results.html', 
                             result=result, 
                             interpretation=interpretation,
                             patient_data=patient_data)
        
    except Exception as e:
        logger.error(f"Prediction route error: {str(e)}")
        flash(f'System error: {str(e)}', 'error')
        return redirect(url_for('patient_form'))

def interpret_prediction(result, patient_data):
    """Provide clinical interpretation of prediction"""
    if 'error' in result:
        return {
            'risk_level': 'Unknown',
            'recommendation': f'Unable to assess: {result["error"]}',
            'clinical_notes': 'Model not available for prediction.',
            'follow_up': 'Please ensure the model is trained and loaded properly.'
        }
    
    probability = result['lassa_probability']
    prediction = result['prediction']
    confidence = result['confidence']
    
    # Risk level assessment
    if probability >= 0.7:
        risk_level = "HIGH"
        risk_color = "danger"
        recommendation = "Immediate isolation and confirmatory testing recommended. Treat as suspected Lassa fever case."
    elif probability >= 0.3:
        risk_level = "MODERATE"
        risk_color = "warning"
        recommendation = "Close monitoring required. Consider Lassa fever testing based on clinical judgment."
    else:
        risk_level = "LOW"
        risk_color = "success"
        recommendation = "Low probability of Lassa fever. Continue standard care and monitoring."
    
    # Key symptoms present
    key_symptoms = []
    symptom_mapping = {
        'fever_new': 'Fever',
        'headache_new': 'Headache',
        'bleeding_new': 'Bleeding',
        'weakness_new': 'Weakness'
    }
    
    for key, label in symptom_mapping.items():
        if patient_data.get(key, 0) == 1:
            key_symptoms.append(label)
    
    return {
        'diagnosis': 'POSITIVE for Lassa Fever' if prediction == 1 else 'NEGATIVE for Lassa Fever',
        'risk_level': risk_level,
        'risk_color': risk_color,
        'lassa_probability_percent': round(probability * 100, 1),
        'confidence_percent': round(confidence * 100, 1),
        'recommendation': recommendation,
        'key_symptoms': key_symptoms,
        'clinical_notes': f"Based on {len(key_symptoms)} key symptoms and patient demographics.",
        'follow_up': "Please follow institutional protocols for suspected Lassa fever cases."
    }

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        result = predictor.predict(data)
        
        if 'error' in result:
            return jsonify({'error': result["error"]}), 400
        
        interpretation = interpret_prediction(result, data)
        
        return jsonify({
            'prediction': result,
            'interpretation': interpretation,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model_info')
def model_info():
    """Display model information"""
    info = {
        'model_loaded': predictor.model is not None,
        'features_count': len(predictor.features_used) if predictor.features_used else 0,
        'features_used': predictor.features_used if predictor.features_used else []
    }
    
    return render_template('model_info.html', info=info)

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return render_template('error.html', error='Internal server error'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info("🏥 Starting Lassa Fever Decision Support System (Production)...")
    logger.info(f"📊 Model Status: {'✅ Loaded' if predictor.model else '❌ Not Loaded'}")
    logger.info(f"🌐 Running on port: {port}")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
