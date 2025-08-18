# database.py - Database management for patient history and model performance tracking
import sqlite3
import json
from datetime import datetime
import pandas as pd

class DatabaseManager:
    def __init__(self, db_path='lassa_fever_app.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Patient predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                prediction INTEGER,
                confidence REAL,
                lassa_probability REAL,
                risk_level TEXT,
                symptoms TEXT,
                demographics TEXT,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT
            )
        ''')
        
        # Model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                auc_score REAL,
                training_date TIMESTAMP,
                notes TEXT
            )
        ''')
        
        # Batch predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batch_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_name TEXT,
                total_patients INTEGER,
                positive_predictions INTEGER,
                negative_predictions INTEGER,
                avg_confidence REAL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_path TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, patient_data, prediction_result):
        """Save individual patient prediction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO patient_predictions 
            (patient_id, prediction, confidence, lassa_probability, risk_level, symptoms, demographics, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_data.get('patient_id', 'Unknown'),
            prediction_result['prediction'],
            prediction_result['confidence'],
            prediction_result['lassa_probability'],
            prediction_result['risk_level'],
            json.dumps(patient_data.get('symptoms', {})),
            json.dumps(patient_data.get('demographics', {})),
            'Enhanced_GNN_v1'
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_predictions(self, limit=50):
        """Get recent predictions for dashboard"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM patient_predictions 
            ORDER BY prediction_date DESC 
            LIMIT ?
        ''', conn, params=(limit,))
        conn.close()
        return df
    
    def get_prediction_statistics(self):
        """Get prediction statistics for dashboard"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) FROM patient_predictions')
        total_predictions = cursor.fetchone()[0]
        
        # Positive predictions
        cursor.execute('SELECT COUNT(*) FROM patient_predictions WHERE prediction = 1')
        positive_predictions = cursor.fetchone()[0]
        
        # Average confidence
        cursor.execute('SELECT AVG(confidence) FROM patient_predictions')
        avg_confidence = cursor.fetchone()[0] or 0
        
        # Recent predictions (last 7 days)
        cursor.execute('''
            SELECT COUNT(*) FROM patient_predictions 
            WHERE prediction_date >= datetime('now', '-7 days')
        ''')
        recent_predictions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_predictions': total_predictions,
            'positive_predictions': positive_predictions,
            'negative_predictions': total_predictions - positive_predictions,
            'positive_rate': (positive_predictions / total_predictions * 100) if total_predictions > 0 else 0,
            'avg_confidence': avg_confidence,
            'recent_predictions': recent_predictions
        }
    
    def save_batch_prediction(self, batch_name, results, file_path):
        """Save batch prediction results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        total_patients = len(results)
        positive_predictions = sum(1 for r in results if r['prediction'] == 1)
        negative_predictions = total_patients - positive_predictions
        avg_confidence = sum(r['confidence'] for r in results) / total_patients if total_patients > 0 else 0
        
        cursor.execute('''
            INSERT INTO batch_predictions 
            (batch_name, total_patients, positive_predictions, negative_predictions, avg_confidence, file_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (batch_name, total_patients, positive_predictions, negative_predictions, avg_confidence, file_path))
        
        conn.commit()
        conn.close()
