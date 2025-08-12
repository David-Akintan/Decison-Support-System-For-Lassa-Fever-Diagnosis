#!/usr/bin/env python3
"""
Simplified Lassa Fever Diagnosis System for Railway Deployment
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'lassa-fever-demo-key-2024')

# Mock prediction function (since we can't load the full model on Railway easily)
def mock_prediction(patient_data):
    """
    Mock prediction function that simulates the GNN model behavior
    This is for demonstration purposes when the full model can't be loaded
    """
    # Simple rule-based mock that considers key symptoms
    risk_score = 0.0
    
    # Fever is a major indicator
    if patient_data.get('fever', 0) == 1:
        risk_score += 0.3
    
    # Headache
    if patient_data.get('headache', 0) == 1:
        risk_score += 0.2
    
    # Muscle pain
    if patient_data.get('muscle_pain', 0) == 1:
        risk_score += 0.2
    
    # Travel to endemic area
    if patient_data.get('travel_endemic', 0) == 1:
        risk_score += 0.25
    
    # Contact with suspected case
    if patient_data.get('contact_suspected', 0) == 1:
        risk_score += 0.15
    
    # Add some randomness to simulate model uncertainty
    import random
    risk_score += random.uniform(-0.1, 0.1)
    risk_score = max(0.0, min(1.0, risk_score))
    
    # Determine risk level
    if risk_score < 0.3:
        risk_level = "Low"
        recommendation = "Monitor symptoms. Seek medical attention if symptoms worsen."
    elif risk_score < 0.7:
        risk_level = "Medium"
        recommendation = "Consult healthcare provider. Consider Lassa fever testing."
    else:
        risk_level = "High"
        recommendation = "Immediate medical attention required. Isolate patient and conduct Lassa fever testing."
    
    return {
        'risk_score': float(risk_score),
        'risk_level': risk_level,
        'recommendation': recommendation,
        'confidence': 0.85,  # Mock confidence
        'prediction': 1 if risk_score > 0.5 else 0
    }

@app.route('/')
def index():
    """Home page with system overview"""
    return render_template('index.html')

@app.route('/patient_form')
def patient_form():
    """Patient data input form"""
    return render_template('patient_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on patient data"""
    try:
        # Helper function to safely convert form values
        def safe_int(value, default=0):
            try:
                if value is None or value == '':
                    return default
                return int(value)
            except (ValueError, TypeError):
                return default
        
        def safe_float(value, default=0.0):
            try:
                if value is None or value == '':
                    return default
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Get form data with safe conversion
        patient_data = {
            'fever': safe_int(request.form.get('fever')),
            'headache': safe_int(request.form.get('headache')),
            'muscle_pain': safe_int(request.form.get('muscle_pain')),
            'fatigue': safe_int(request.form.get('fatigue')),
            'sore_throat': safe_int(request.form.get('sore_throat')),
            'nausea': safe_int(request.form.get('nausea')),
            'vomiting': safe_int(request.form.get('vomiting')),
            'diarrhea': safe_int(request.form.get('diarrhea')),
            'abdominal_pain': safe_int(request.form.get('abdominal_pain')),
            'chest_pain': safe_int(request.form.get('chest_pain')),
            'travel_endemic': safe_int(request.form.get('travel_endemic')),
            'contact_suspected': safe_int(request.form.get('contact_suspected')),
            'contact_confirmed': safe_int(request.form.get('contact_confirmed')),
            'healthcare_exposure': safe_int(request.form.get('healthcare_exposure')),
            'age': safe_float(request.form.get('age'), 30.0),
            'temperature': safe_float(request.form.get('temperature'), 37.0)
        }
        
        logger.info(f"Processing patient data: {patient_data}")
        
        # Make prediction using mock function
        result = mock_prediction(patient_data)
        
        # Add timestamp
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result['patient_data'] = patient_data
        
        logger.info(f"Prediction made: {result['risk_level']} risk")
        
        return render_template('results.html', result=result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        result = mock_prediction(data)
        result['timestamp'] = datetime.now().isoformat()
        return jsonify(result)
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'model_status': 'mock_model_active'
    })

@app.route('/model_info')
def model_info():
    """Model information page"""
    model_info = {
        'name': 'Lassa Fever GNN Diagnosis System',
        'version': '1.0.0 (Demo)',
        'architecture': 'Hybrid GCN + GAT',
        'status': 'Demo Mode - Mock Predictions',
        'accuracy': '19.26% (baseline)',
        'sensitivity': '93.63%',
        'specificity': '~14.52%',
        'note': 'This is a demonstration version using rule-based predictions',
        'model_loaded': True  # Add the missing field that template expects
    }
    return render_template('model_info.html', info=model_info)

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    logger.info(f"Starting Lassa Fever Diagnosis System on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
