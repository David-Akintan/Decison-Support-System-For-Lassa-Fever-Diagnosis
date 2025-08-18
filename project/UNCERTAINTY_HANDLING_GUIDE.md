# Uncertainty Handling in Lassa Fever Diagnosis System

## Overview

The enhanced Lassa fever diagnosis system now includes comprehensive uncertainty handling for epidemiological data. This addresses real-world clinical scenarios where patients may not know their contact history, travel history, or exposure to risk factors.

## Key Features

### 1. Enhanced Patient Form
- **Unknown Options**: Added "Unknown" selections for critical epidemiological factors:
  - Contact with Lassa Case
  - Recent Travel to Endemic Area
  - Rodent/Excreta Exposure
- **User Guidance**: Helper text instructs when to select "Unknown"
- **Value Encoding**: Unknown = -1, No = 0, Yes = 1

### 2. Intelligent Risk Assessment

#### Uncertainty Detection
The system automatically detects missing epidemiological data and categorizes uncertainty levels:

- **1 Unknown Factor**: "Some epidemiological data missing - Monitor closely"
- **2 Unknown Factors**: "Important epidemiological data missing - Increased vigilance recommended"
- **3+ Unknown Factors**: "Critical epidemiological data missing - Consider higher risk category"

#### Risk Level Escalation
When significant uncertainty is detected (≥2 unknown factors), the system automatically escalates risk levels:

```
VERY LOW → LOW
LOW → MODERATE  
MODERATE → HIGH
HIGH → VERY HIGH
```

### 3. Advanced Preprocessing

#### Neutral Value Assignment
Unknown epidemiological factors are converted to neutral values (0.5) during model preprocessing:

- **No (0)** → 0.0 (definitive negative)
- **Unknown (-1)** → 0.5 (neutral/uncertain)
- **Yes (1)** → 1.0 (definitive positive)

This approach distinguishes between definitive "No" responses and genuine uncertainty.

### 4. Enhanced Medical Interpretation

#### Uncertainty Warnings
Medical advice now includes specific warnings about missing data:

```
⚠️ UNCERTAINTY: Important epidemiological data missing (2 factors unknown). 
Increased vigilance recommended. Risk level elevated due to missing epidemiological data.
```

#### Clinical Decision Support
- Maintains clinical safety by being conservative with uncertain data
- Provides specific guidance on monitoring and testing protocols
- Escalates care recommendations when key information is missing

## Implementation Details

### Form Processing
```python
# Epidemiological factors support -1 for unknown
patient_data['contact_history'] = int(request.form.get('contact_history', 0))
patient_data['travel_history'] = int(request.form.get('travel_history', 0))
patient_data['rodent_exposure'] = int(request.form.get('rodent_exposure', 0))
```

### Uncertainty Assessment
```python
def assess_uncertainty(self, patient_data):
    # Checks for -1 values in epidemiological factors
    # Returns uncertainty metrics and recommendations
    # Triggers risk escalation when appropriate
```

### Preprocessing Logic
```python
# Convert unknown values to neutral (0.5) for model input
if col in epidemiological_fields:
    numeric_values = pd.to_numeric(df_features[col], errors='coerce')
    df_features[col] = numeric_values.apply(
        lambda x: 0.5 if x == -1 else (1.0 if x == 1 else 0.0)
    ).fillna(0)
```

## Clinical Benefits

1. **Real-World Applicability**: Handles common scenarios where patients don't know their exposure history
2. **Safety-First Approach**: Escalates risk when uncertainty could impact diagnosis
3. **Transparent Decision Making**: Clearly communicates uncertainty to healthcare providers
4. **Graduated Response**: Provides proportional warnings based on amount of missing data

## Usage Guidelines

### For Healthcare Providers
- Select "Unknown" when patients genuinely don't know their exposure history
- Pay attention to uncertainty warnings in the medical advice
- Consider additional testing/monitoring when uncertainty is high
- Use escalated risk levels to guide clinical decisions

### For System Administrators
- Monitor prediction logs for patterns in uncertain data
- Consider additional data collection strategies for high-uncertainty cases
- Review model performance with uncertain vs. definitive data

## Testing and Validation

The uncertainty handling has been validated through:
- Form processing with unknown values
- Risk escalation logic verification  
- Preprocessing value conversion testing
- Medical interpretation with uncertainty scenarios

## Future Enhancements

Potential improvements include:
- Machine learning models specifically trained on uncertain data
- Confidence intervals for predictions with missing data
- Integration with external data sources to reduce uncertainty
- Advanced imputation techniques for missing epidemiological factors
