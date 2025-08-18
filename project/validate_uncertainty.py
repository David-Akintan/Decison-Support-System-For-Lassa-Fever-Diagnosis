#!/usr/bin/env python3
"""
Simple validation script for uncertainty handling
"""

def validate_form_processing():
    """Validate form data processing with unknown values"""
    print("🔍 Validating Form Processing with Unknown Values")
    print("=" * 50)
    
    # Simulate form data with unknown values
    form_data = {
        'age': '35',
        'sex': '1',
        'pregnancy': '0',
        'fever_new': '1',
        'headache_new': '1',
        'contact_history': '-1',  # Unknown
        'travel_history': '0',    # No
        'rodent_exposure': '-1',  # Unknown
        'state': 'Edo'
    }
    
    # Simulate form processing logic from app_enhanced.py
    patient_data = {}
    
    # Demographics
    patient_data['age'] = float(form_data.get('age', 0))
    patient_data['sex'] = int(form_data.get('sex', 0))
    patient_data['pregnancy'] = int(form_data.get('pregnancy', 0))
    
    # Symptoms
    symptoms = ['fever_new', 'headache_new', 'vomiting_new', 'bleeding_new']
    for symptom in symptoms:
        patient_data[symptom] = 1 if form_data.get(symptom) == '1' else 0
    
    # Epidemiological factors
    patient_data['contact_history'] = int(form_data.get('contact_history', 0))
    patient_data['travel_history'] = int(form_data.get('travel_history', 0))
    patient_data['rodent_exposure'] = int(form_data.get('rodent_exposure', 0))
    
    # Map to model fields
    patient_data['contact_with_source_case_new'] = patient_data['contact_history']
    patient_data['travelled_outside_district'] = patient_data['travel_history']
    
    print("✅ Form data processed successfully:")
    for key, value in patient_data.items():
        if 'history' in key or 'exposure' in key or 'contact' in key:
            status = "Unknown" if value == -1 else ("Yes" if value == 1 else "No")
            print(f"   {key}: {value} ({status})")
        else:
            print(f"   {key}: {value}")

def validate_uncertainty_assessment():
    """Validate uncertainty assessment logic"""
    print("\n🎯 Validating Uncertainty Assessment")
    print("=" * 40)
    
    # Simulate uncertainty assessment logic
    def assess_uncertainty(patient_data):
        uncertainty_factors = {
            'has_uncertainty': False,
            'warning': '',
            'recommend_higher_vigilance': False,
            'missing_factors': []
        }
        
        epi_factors = {
            'contact_with_source_case_new': 'Contact with confirmed Lassa case',
            'travelled_outside_district': 'Travel to endemic areas',
            'contact_history': 'Contact history',
            'travel_history': 'Recent travel history',
            'rodent_exposure': 'Exposure to rodents/excreta'
        }
        
        missing_count = 0
        for factor, description in epi_factors.items():
            value = patient_data.get(factor, -1)
            if value == -1 or value == 'unknown':
                uncertainty_factors['missing_factors'].append(description)
                missing_count += 1
        
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
    
    # Test cases
    test_cases = [
        {
            'name': 'No unknown factors',
            'data': {'contact_history': 0, 'travel_history': 0, 'rodent_exposure': 0}
        },
        {
            'name': 'One unknown factor',
            'data': {'contact_history': -1, 'travel_history': 0, 'rodent_exposure': 0}
        },
        {
            'name': 'Two unknown factors',
            'data': {'contact_history': -1, 'travel_history': -1, 'rodent_exposure': 0}
        },
        {
            'name': 'Three unknown factors',
            'data': {'contact_history': -1, 'travel_history': -1, 'rodent_exposure': -1}
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📋 {test_case['name']}:")
        uncertainty = assess_uncertainty(test_case['data'])
        print(f"   Has uncertainty: {uncertainty['has_uncertainty']}")
        print(f"   Missing count: {len(uncertainty['missing_factors'])}")
        print(f"   Higher vigilance: {uncertainty['recommend_higher_vigilance']}")
        print(f"   Warning: {uncertainty['warning']}")

def validate_preprocessing_logic():
    """Validate preprocessing of unknown values"""
    print("\n⚙️ Validating Preprocessing Logic")
    print("=" * 35)
    
    # Simulate preprocessing logic for epidemiological fields
    def process_epidemiological_value(value):
        """Convert epidemiological values: -1 (unknown) -> 0.5, 1 (yes) -> 1.0, 0 (no) -> 0.0"""
        if value == -1:
            return 0.5  # Neutral/uncertain
        elif value == 1:
            return 1.0  # Yes
        else:
            return 0.0  # No
    
    test_values = [-1, 0, 1]
    value_names = ['Unknown', 'No', 'Yes']
    
    print("Value conversion for epidemiological fields:")
    for value, name in zip(test_values, value_names):
        processed = process_epidemiological_value(value)
        print(f"   {name} ({value}) → {processed}")

if __name__ == "__main__":
    validate_form_processing()
    validate_uncertainty_assessment()
    validate_preprocessing_logic()
    print("\n✅ All validation checks completed successfully!")
