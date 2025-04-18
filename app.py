from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

MODEL_FILE = 'lung_disease_model.plk'
SCALER_FILE = 'scaler.plk' # Scaler file path

model = None
scaler = None # Initialize scaler

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
        model = None # Invalidate model if scaler fails to load
else:
     if model is not None:
         print(f"Error: Scaler file '{SCALER_FILE}' not found. Model requires scaler.")
         model = None # Invalidate model if scaler is missing

if model is None:
     print("Application will not be able to predict. Please run generate_model.py")

feature_order = [
    'Age', 'Gender', 'Smoking_History', 'Air_Pollution_Exposure', 
    'Family_History', 'Chronic_Cough', 'Shortness_of_Breath', 'Wheezing', 
    'Sputum_Production', 'BMI', 'Physical_Activity', 'Occupational_Exposure', 
    'Respiratory_Infections', 'Allergies', 'Medication_Use'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None: # Check both model and scaler
        return jsonify({'error': f"Model or Scaler not loaded. Cannot predict. Please check server logs and run generate_model.py."}), 500

    try:
        form_data = request.form.to_dict()
        print(f"Received form data: {form_data}")

        input_features = []
        for feature in feature_order:
            value = form_data.get(feature)
            if value is None or value == '': 
                 if feature in form_data:
                      return jsonify({'error': f'Empty input value for {feature}'}), 400
                 else:
                      return jsonify({'error': f'Missing input value for {feature}'}), 400
            try:
                 input_features.append(float(value))
            except ValueError:
                return jsonify({'error': f'Invalid input value for {feature}: \'{value}\'. Please enter a valid number or selection.'}), 400

        # Reshape features for scaler
        features_array = np.array(input_features).reshape(1, -1)
        
        # Scale the input features using the loaded scaler
        print(f"Features before scaling: {features_array}")
        scaled_features = scaler.transform(features_array)
        print(f"Features after scaling: {scaled_features}")

        # --- Get Probability --- 
        probability_percentage_text = 'N/A' 
        risk_proba = None # Initialize risk_proba
        prediction_fallback = None # For case where predict_proba fails

        if 'predict_proba' in dir(model):
            try:
                probability = model.predict_proba(scaled_features)[0][1]
                risk_proba = probability
                probability_percentage_text = f"{risk_proba * 100:.2f}%"
                print(f"DEBUG: Probability from predict_proba: {probability}")
            except:
                print("DEBUG: predict_proba failed, using predict method")
                prediction_fallback = model.predict(scaled_features)[0]
                print(f"Direct Prediction: {prediction_fallback}")
        else:
            print("DEBUG: predict_proba not available, using predict method")
            prediction_fallback = model.predict(scaled_features)[0]
            print(f"Direct Prediction: {prediction_fallback}")

        # --- Determine Final Prediction Status --- 
        if risk_proba is not None:
            # Count how many symptomatic features have value = 1 (Yes)
            symptom_features = ["Chronic_Cough", "Shortness_of_Breath", "Wheezing", 
                               "Sputum_Production", "Respiratory_Infections", "Allergies"]
            symptom_indices = [feature_order.index(feature) for feature in symptom_features 
                              if feature in feature_order]
            symptom_count = sum(1 for i in symptom_indices if input_features[i] == 1)
            total_symptoms = len(symptom_indices)
            
            print(f"DEBUG: Risk probability: {risk_proba}")
            print(f"DEBUG: Symptom count: {symptom_count} out of {total_symptoms}")
            print(f"DEBUG: Threshold check: risk_proba >= 0.4 = {risk_proba >= 0.4}")
            print(f"DEBUG: Symptom check: symptom_count >= total_symptoms/2 and >= 3 = {symptom_count >= total_symptoms/2 and symptom_count >= 3}")
            
            # New logic: Predict Positive if probability >= 40% OR majority of symptoms are "Yes"
            if risk_proba >= 0.4 or (symptom_count >= total_symptoms / 2 and symptom_count >= 3):
                prediction_text_output = "Positive"
                print(f"DEBUG: Setting status to Positive. Prob: {risk_proba}, Threshold: 0.4, Symptom count: {symptom_count}/{total_symptoms}")
            else:
                prediction_text_output = "Negative"
                print(f"DEBUG: Setting status to Negative. Prob: {risk_proba}, Threshold: 0.4, Symptom count: {symptom_count}/{total_symptoms}")
        elif prediction_fallback is not None:
             # Use direct predict output if predict_proba failed
             prediction_text_output = "Positive" if prediction_fallback == 1 else "Negative"
             print(f"Using Fallback Direct Prediction: Result='{prediction_text_output}'")
        else:
            # If both failed
            prediction_text_output = "Error"
            print("Error: Could not determine prediction status.")

        print(f"DEBUG: Final prediction_status: '{prediction_text_output}'")
        print(f"DEBUG: Final probability: {risk_proba}")
        print(f"DEBUG: Final probability_text: {probability_percentage_text}")
        
        # --- Return JSON Response --- 
        response_data = {
            'prediction_status': prediction_text_output, 
            'probability': risk_proba,
            'probability_text': probability_percentage_text 
        }
        print(f"DEBUG: Final response: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        import traceback
        print(f"Error during prediction processing: {e}")
        traceback.print_exc()
        return jsonify({'error': 'An unexpected error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(debug=True) 