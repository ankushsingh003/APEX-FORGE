import os
import sys
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from datetime import datetime

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

app = Flask(__name__, template_folder='template', static_folder='static')

# Constants & Paths
MODEL_PATH = "artifacts/model/model.joblib"

# Mappings for categorical variables (Alphabetical order as per LabelEncoder)
MAPPINGS = {
    'type_of_meal': ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'],
    'room_type': ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'],
    'market_segment_type': ['Aviation', 'Complementary', 'Corporate', 'Offline', 'Online']
}

def preprocess_input(data):
    """
    Transforms raw form data into model-ready features.
    Matches the 10 features selected during training.
    """
    # Map form data to model features
    features = {
        'lead_time': int(data.get('lead time', 0)),
        'average_price': float(data.get('average price', 0)),
        'special_requests': int(data.get('special requests', 0)),
        'number_of_week_nights': int(data.get('number of week nights', 0)),
        'number_of_weekend_nights': int(data.get('number of weekend nights', 0)),
        'market_segment_type': data.get('market segment type', 'Online'),
        'room_type': data.get('room type', 'Room_Type 1'),
        'number_of_adults': int(data.get('number of adults', 2)),
        'type_of_meal': data.get('type of meal', 'Meal Plan 1'),
        'car_parking_space': int(data.get('car parking space', 0))
    }
    
    proc_df = pd.DataFrame([features])
    
    # Apply Label Encoding (Manual mapping to match LabelEncoder)
    for col, classes in MAPPINGS.items():
        if col in proc_df.columns:
            try:
                proc_df[col] = classes.index(proc_df[col])
            except ValueError:
                proc_df[col] = -1 # Handle unknown
    
    # Re-order columns to match the exact order expected by the model
    # ['lead_time', 'average_price', 'special_requests', 'number_of_week_nights', 'number_of_weekend_nights', 'market_segment_type', 'room_type', 'number_of_adults', 'type_of_meal', 'car_parking_space']
    feature_order = [
        'lead_time', 'average_price', 'special_requests', 'number_of_week_nights',
        'number_of_weekend_nights', 'market_segment_type', 'room_type',
        'number_of_adults', 'type_of_meal', 'car_parking_space'
    ]
    proc_df = proc_df[feature_order]
    
    return proc_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({
                'success': False,
                'error': 'Model file not found. Please run the training pipeline first.'
            })

        data = request.json
        processed_data = preprocess_input(data)
        
        # Load model
        model = joblib.load(MODEL_PATH)
        
        # Prediction
        prediction = model.predict(processed_data)[0]
        
        # Get Probabilities
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(processed_data)[0]
            # Mapping depends on internal class order: 0: Not_Canceled, 1: Canceled
            prob_not_canceled = probs[0]
            prob_canceled = probs[1]
        else:
            prob_not_canceled = 0.9 if prediction == 0 else 0.1
            prob_canceled = 1 - prob_not_canceled

        result = {
            'success': True,
            'prediction': 'Canceled' if prediction == 1 else 'Not_Canceled',
            'probabilities': {
                'not_canceled': float(prob_not_canceled),
                'canceled': float(prob_canceled)
            }
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Use port 5000 as requested
    app.run(debug=True, port=5000)
