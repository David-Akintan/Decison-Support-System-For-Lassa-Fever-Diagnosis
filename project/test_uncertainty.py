#!/usr/bin/env python3
"""
Test script for uncertainty handling in Lassa fever diagnosis system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app_enhanced import EnhancedLassaFeverPredictor
import pandas as pd

def test_uncertainty_handling():
    """Test the uncertainty handling functionality"""
    print("🧪 Testing Uncertainty Handling for Lassa Fever Diagnosis")
    print("=" * 60)
    
    # Initialize predictor
    predictor = EnhancedLassaFeverPredictor()
    
    # Test cases with different uncertainty scenarios
    test_cases = [
        {
            'name': 'Patient with known epidemiological data',
            'data': {
                'age': 35,
                'sex': 1,
                'fever_new': 1,
                'headache_new': 1,
                'contact_history': 1,  # Known contact
                'travel_history': 0,   # No travel
                'rodent_exposure': 0,  # No exposure
                'state': 'Edo'
            }
        },
        {
            'name': 'Patient with unknown contact history',
            'data': {
                'age': 28,
                'sex': 0,
                'fever_new': 1,
                'headache_new': 1,
                'bleeding_new': 1,
                'contact_history': -1,  # Unknown
                'travel_history': 0,
                'rodent_exposure': 0,
                'state': 'Ondo'
            }
        },
        {
            'name': 'Patient with multiple unknown factors',
            'data': {
                'age': 42,
                'sex': 1,
                'fever_new': 1,
                'vomiting_new': 1,
                'weakness_new': 1,
                'contact_history': -1,  # Unknown
                'travel_history': -1,   # Unknown
                'rodent_exposure': -1,  # Unknown
                'state': 'Bauchi'
            }
        },
        {
            'name': 'Patient with all unknown epidemiological data',
            'data': {
                'age': 25,
                'sex': 0,
                'fever_new': 1,
                'headache_new': 1,
                'muscle_pain': 1,
                'contact_history': -1,  # Unknown
                'travel_history': -1,   # Unknown
                'rodent_exposure': -1,  # Unknown
                'state': 'Plateau'
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print("-" * 50)
        
        # Test uncertainty assessment
        uncertainty_factors = predictor.assess_uncertainty(test_case['data'])
        
        print(f"Has Uncertainty: {uncertainty_factors['has_uncertainty']}")
        print(f"Missing Factors: {len(uncertainty_factors['missing_factors'])}")
        if uncertainty_factors['missing_factors']:
            print(f"Missing: {', '.join(uncertainty_factors['missing_factors'])}")
        print(f"Warning: {uncertainty_factors['warning']}")
        print(f"Recommend Higher Vigilance: {uncertainty_factors['recommend_higher_vigilance']}")
        
        # Test medical interpretation with uncertainty
        mock_probability = 0.6  # Moderate risk
        risk_level, advice = predictor.get_medical_interpretation(mock_probability, test_case['data'])
        
        print(f"Risk Level: {risk_level}")
        print(f"Medical Advice: {advice}")
        
        # Test preprocessing with unknown values
        try:
            X = predictor.preprocess_patient_data(test_case['data'])
            if X is not None:
                print(f"✅ Preprocessing successful - Shape: {X.shape}")
                
                # Check how unknown values are handled
                epidemiological_fields = ['contact_history', 'travel_history', 'rodent_exposure']
                for field in epidemiological_fields:
                    if field in test_case['data'] and test_case['data'][field] == -1:
                        print(f"   Unknown {field} converted to neutral value (0.5)")
            else:
                print("❌ Preprocessing failed")
        except Exception as e:
            print(f"❌ Preprocessing error: {str(e)}")

def test_risk_escalation():
    """Test risk level escalation due to uncertainty"""
    print("\n🔺 Testing Risk Level Escalation")
    print("=" * 40)
    
    predictor = EnhancedLassaFeverPredictor()
    
    risk_levels = ['VERY LOW', 'LOW', 'MODERATE', 'HIGH', 'VERY HIGH']
    
    for risk in risk_levels:
        escalated = predictor.escalate_risk_level(risk)
        print(f"{risk} → {escalated}")

if __name__ == "__main__":
    try:
        test_uncertainty_handling()
        test_risk_escalation()
        print("\n✅ All uncertainty handling tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
