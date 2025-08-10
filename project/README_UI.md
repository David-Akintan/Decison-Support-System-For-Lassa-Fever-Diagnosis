# Lassa Fever Decision Support System - Web Interface

## Overview

A comprehensive web-based decision support system for Lassa fever diagnosis using advanced Hybrid Graph Neural Networks (GNN). The system combines Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) to provide healthcare professionals with AI-powered diagnostic assistance.

## Features

- **🧠 Hybrid GNN Architecture**: Combines GCN and GAT for superior performance
- **⚖️ Class Imbalance Handling**: Advanced techniques for rare disease detection
- **🏥 Clinical Focus**: Medical-aware preprocessing and clinical interpretation
- **📊 Risk Assessment**: Probability-based risk stratification with clinical recommendations
- **🖥️ User-Friendly Interface**: Modern, responsive web interface for healthcare professionals
- **🔒 Privacy Compliant**: Designed with healthcare data security in mind

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (if not already done)

```bash
python train.py --csv sample-data.csv --model_type hybrid --epochs 150 --lr 1e-3
```

### 3. Launch the Web Interface

```bash
python app.py
```

The application will be available at: `http://localhost:5000`

## System Architecture

### Model Components
- **GCN Layers**: Capture local neighborhood patterns in patient similarity graphs
- **GAT Layers**: Learn adaptive attention weights for relevant patient features  
- **Fusion Layer**: Attention-based combination of GCN and GAT outputs
- **Classification Head**: Binary classification for Lassa fever diagnosis

### Training Features
- **SMOTE Oversampling**: Synthetic minority class augmentation
- **Enhanced Class Weights**: 3x boost for minority class detection
- **Combined Loss**: Focal Loss + CrossEntropy for imbalanced data
- **Minority-Focused Early Stopping**: Based on Lassa F1 score for better sensitivity

## Web Interface Pages

### 1. Dashboard (`/`)
- System overview and quick actions
- Performance metrics and capabilities
- Navigation to assessment and model info

### 2. Patient Assessment (`/patient_form`)
- Comprehensive clinical data input form
- Demographics, symptoms, epidemiological factors
- Interactive symptom selection with visual feedback

### 3. Results Display (`/predict`)
- AI diagnosis with risk stratification
- Clinical recommendations and interpretation
- Probability visualization and confidence metrics

### 4. Model Information (`/model_info`)
- System status and technical specifications
- Architecture details and performance metrics
- Feature list and training strategies

## API Endpoints

### POST `/api/predict`
JSON API for programmatic access:

```json
{
  "age_recode": 35,
  "sex_new2": 1,
  "fever_new": 1,
  "headache_new": 1,
  "contact_with_lassa_case": 0,
  "State_new": "Edo"
}
```

Response:
```json
{
  "prediction": {
    "prediction": 0,
    "confidence": 0.85,
    "lassa_probability": 0.15
  },
  "interpretation": {
    "diagnosis": "NEGATIVE for Lassa Fever",
    "risk_level": "LOW",
    "recommendation": "Low probability of Lassa fever..."
  }
}
```

## Clinical Features

The system analyzes 51 clinical parameters including:

### Demographics
- Age, Sex, Pregnancy status

### Clinical Symptoms
- Fever, Headache, Vomiting, Bleeding
- Weakness, Abdominal pain, Diarrhea
- Respiratory symptoms, Joint/muscle pain

### Epidemiological Factors
- Contact with confirmed Lassa cases
- Travel history to endemic areas
- Rodent/excreta exposure

### Geographic Information
- State/region of residence

## Risk Stratification

- **HIGH RISK (≥70%)**: Immediate isolation and testing recommended
- **MODERATE RISK (30-69%)**: Close monitoring and clinical judgment
- **LOW RISK (<30%)**: Standard care with symptom monitoring

## Performance Metrics

- **Overall Accuracy**: 80-85%
- **Lassa Detection Rate**: 40-70% (sensitivity for minority class)
- **AUC Score**: 0.75-0.85
- **F1 Score**: 0.70-0.80

## Important Medical Disclaimer

⚠️ **This system is a decision support tool only**
- Clinical judgment and confirmatory testing are required for diagnosis
- Follow established medical protocols and consult specialists
- Consider patient history, physical examination, and laboratory results
- Not intended for autonomous diagnosis or treatment decisions

## Technical Requirements

### Minimum
- Python 3.8+
- PyTorch 1.9+
- 4GB RAM
- CPU: Intel i5 or equivalent

### Recommended
- Python 3.9+
- PyTorch 2.0+
- CUDA-enabled GPU
- 8GB+ RAM
- SSD storage

## File Structure

```
project/
├── app.py                 # Flask web application
├── models.py             # GNN model architectures
├── preprocess.py         # Data preprocessing pipeline
├── train.py              # Model training script
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates
│   ├── index.html        # Main dashboard
│   ├── patient_form.html # Patient input form
│   ├── results.html      # Results display
│   └── model_info.html   # Model information
├── model.pth            # Trained model (generated)
├── preproc.pkl          # Preprocessor (generated)
└── sample-data.csv      # Training data
```

## Usage Workflow

1. **Train Model**: Run `train.py` with your dataset
2. **Launch Interface**: Start `app.py` to open web interface
3. **Input Patient Data**: Use the assessment form to enter clinical information
4. **Review Results**: Analyze AI predictions and clinical recommendations
5. **Clinical Decision**: Use results to support medical decision-making

## Security Considerations

- No patient data is stored permanently
- All processing occurs locally
- Secure session management
- Input validation and sanitization
- HTTPS recommended for production deployment

## Troubleshooting

### Model Not Loading
- Ensure `model.pth` and `preproc.pkl` exist
- Check model was trained successfully
- Verify PyTorch version compatibility

### Prediction Errors
- Validate input data format
- Check feature compatibility with training data
- Ensure all required fields are provided

### Performance Issues
- Use GPU acceleration if available
- Optimize batch sizes for your hardware
- Consider model quantization for deployment

## Support

For technical issues or questions about the decision support system, please refer to the training logs and model validation results. The system is designed to be robust and provide consistent performance across different deployment environments.
