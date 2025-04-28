from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

MODEL_FILE = 'lung_disease_model.plk'
SCALER_FILE = 'scaler.plk' 
FEATURE_COLS_FILE = 'feature_cols.pkl'

model = None
scaler = None
feature_cols = None

if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        print(f"Model '{MODEL_FILE}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model '{MODEL_FILE}': {e}")
else:
    print(f"Error: Model file '{MODEL_FILE}' not found.")

# Load the scaler 
if model is not None and os.path.exists(SCALER_FILE):
    try:
        scaler = joblib.load(SCALER_FILE)
        print(f"Scaler '{SCALER_FILE}' loaded successfully.")
    except Exception as e:
        print(f"Error loading scaler '{SCALER_FILE}': {e}")
        model = None
else:
     if model is not None:
         print(f"Error: Scaler file '{SCALER_FILE}' not found. Model requires scaler.")
         model = None

# Load feature columns
if os.path.exists(FEATURE_COLS_FILE):
    try:
        feature_cols = joblib.load(FEATURE_COLS_FILE)
        print(f"Feature columns loaded successfully from '{FEATURE_COLS_FILE}'")
    except Exception as e:
        print(f"Error loading feature columns from '{FEATURE_COLS_FILE}': {e}")
        feature_cols = None
else:
    print(f"Feature columns file '{FEATURE_COLS_FILE}' not found. Using default feature list.")
    feature_cols = None

if model is None:
     print("Application will not be able to predict. Please run generate_model.py")

# Base features received from the form
base_feature_order = [
    'Age', 'Gender', 'Smoking_History', 
    'Family_History', 'Chronic_Cough', 'Shortness_of_Breath', 'Wheezing', 
    'Sputum_Production', 'BMI', 'Physical_Activity', 'Occupational_Exposure', 
    'Respiratory_Infections', 'Allergies', 'Medication_Use'
]

# Define risk factors and symptoms for clinical validation
risk_factors = ['Smoking_History', 'Family_History', 'Occupational_Exposure']
symptoms = ['Chronic_Cough', 'Shortness_of_Breath', 'Wheezing', 'Sputum_Production', 'Respiratory_Infections', 'Allergies']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': f"Model or Scaler not loaded. Cannot predict. Please check server logs and run generate_model.py."}), 500

    try:
        form_data = request.form.to_dict()
        print(f"Received form data: {form_data}")

        # Extract base features from form
        base_features = {}
        for feature in base_feature_order:
            value = form_data.get(feature)
            if value is None or value == '': 
                 if feature in form_data:
                      return jsonify({'error': f'Empty input value for {feature}'}), 400
                 else:
                      return jsonify({'error': f'Missing input value for {feature}'}), 400
            try:
                 base_features[feature] = float(value)
            except ValueError:
                return jsonify({'error': f'Invalid input value for {feature}: \'{value}\'. Please enter a valid number or selection.'}), 400
        
        # Create DataFrame with base features
        input_df = pd.DataFrame([base_features])
        
        # --- Clinical validation check ---
        # Count risk factors and symptoms
        risk_factor_count = sum(1 for factor in risk_factors if factor in input_df.columns and input_df[factor].iloc[0] == 1)
        symptom_count = sum(1 for symptom in symptoms if symptom in input_df.columns and input_df[symptom].iloc[0] == 1)
        
        # --- Apply only the lightweight feature engineering ---
        # Calculate symptom score
        symptom_cols = ['Chronic_Cough', 'Shortness_of_Breath', 'Wheezing', 
                       'Sputum_Production', 'Respiratory_Infections', 'Allergies']
        input_df['Symptom_Score'] = input_df[symptom_cols].sum(axis=1)
        
        # Convert categorical features
        input_df = pd.get_dummies(input_df, columns=['Gender'], drop_first=True)
        
        # Ensure correct columns if feature_cols is loaded
        if feature_cols is not None:
            # Add missing columns with zeros
            for col in feature_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match training data
            input_df = input_df[feature_cols]
        
        # Reshape and scale features
        features_array = input_df.values
        scaled_features = scaler.transform(features_array)
        
        # Get model prediction and probability
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]
        
        # --- Clinical override for edge cases ---
        # If no symptoms and no risk factors, override to negative
        if symptom_count == 0 and risk_factor_count == 0:
            prediction = 0
            probability = max(0.05, min(probability, 0.15))  # Cap probability between 5-15%
            clinical_override = True
        # If many symptoms but model says negative with very low probability
        elif symptom_count >= 3 and prediction == 0 and probability < 0.2:
            prediction = 1
            probability = max(0.6, probability)
            clinical_override = True
        else:
            clinical_override = False
            
        # Format output
        prediction_text = "Positive" if prediction == 1 else "Negative"
        probability_percentage_text = f"{probability * 100:.2f}%"
        
        # Simple confidence level
        confidence = "High" if probability > 0.75 or probability < 0.25 else "Medium"
        
        # Return response
        response_data = {
            'prediction_status': prediction_text,
            'probability': probability,
            'probability_text': probability_percentage_text,
            'confidence': confidence,
            'symptom_count': symptom_count,
            'risk_factor_count': risk_factor_count,
            'clinical_override': clinical_override
        }
        
        return jsonify(response_data)

    except Exception as e:
        import traceback
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': 'An unexpected error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(debug=True) 