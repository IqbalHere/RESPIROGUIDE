# RESPIROGUIDE: COPD Risk Prediction

RESPIROGUIDE is a web application built with Flask that predicts the likelihood of Chronic Obstructive Pulmonary Disease (COPD) based on patient data. It utilizes a machine learning model (RandomForestClassifier) trained on the `copd_15_features_1000_rows.xlsx` dataset.

The application takes 15 patient characteristics as input and predicts the COPD diagnosis (Positive/Negative) along with an estimated probability.

## Performance Optimizations

This application has been optimized to run on lower-performance computers with:
- Simplified feature engineering that focuses on the most important predictors
- Lightweight model training with optimized hyperparameter search
- Reduced computational requirements for both training and prediction
- Clinical validation overrides to ensure medically sound predictions

## Setup and Running Locally

1.  **Prerequisites:**
    *   Python (>=3.7 recommended)
    *   pip (Python package installer)
    *   Git (optional, for cloning)

2.  **Get the Code:**
    *   Clone the repository (if available):
        ```bash
        git clone https://github.com/IqbalHere/RESPIROGUIDE.git
        cd RESPIROGUIDE
        
        ```
    *   Or, ensure all project files (`app.py`, `generate_model.py`, `copd_15_features_1000_rows.xlsx`, `requirements.txt`, `static/`, `templates/`) are in a single directory.

3.  **Create and Activate Virtual Environment (Recommended):**
    Navigate to the project directory (`copd__proj`) in your terminal and run:
    ```bash
    # Create environment
    python -m venv .venv
    
    # Activate (Windows)
    .venv\Scripts\activate
    
  
  

4.  **Install Dependencies:**
    While the virtual environment is active, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Train the Model:**
    Run the model training script. This reads the dataset and creates the `lung_disease_model.plk` file:
    ```bash
    python generate_model.py
    ```
    *Ensure `copd_15_features_1000_rows.xlsx` is in the same directory.* 

6.  **Run the Flask Application:**
    Start the web server:
    ```bash
    flask run
    ```
    Or alternatively:
    ```bash
    python app.py
    ```
    Look for a message in the terminal indicating the address where the app is running (usually `http://127.0.0.1:5000`).

7.  **Access the App:**
    Open your web browser and go to the address shown in the terminal (e.g., `http://127.0.0.1:5000`). You should see the COPD prediction form.

## Development Mode

For developers working on the application, you can run the Flask app in debug mode to enable automatic reloading on code changes and enhanced error messages:

```bash
# On Windows
set FLASK_DEBUG=1
flask run

# On Unix/Linux/Mac
export FLASK_DEBUG=1
flask run
```

This is particularly useful when making changes to the Flask routes or templates as it eliminates the need to manually restart the server after each change.

## Prediction Functionality

The application uses a combination of factors to determine COPD risk:

1. **Machine Learning Model**: The RandomForest model calculates a probability score (0-100%) for COPD risk based on all input features.

2. **Clinical Validation Override**: 
   - If a patient has NO symptoms AND NO risk factors, the prediction is automatically set to "Negative" regardless of the model output
   - If a patient has 3+ symptoms but the model predicts "Negative" with low confidence, the prediction is overridden to "Positive"
   - These clinical overrides ensure medically sound predictions in edge cases

3. **Decision Factors**: The model considers both risk factors and symptoms:
   - **Risk Factors**: Smoking History, Family History, Occupational Exposure
   - **Key Symptoms**: Chronic Cough, Shortness of Breath, Wheezing, Sputum Production, Respiratory Infections, Allergies

This hybrid approach combines the statistical power of machine learning with clinical knowledge to provide more accurate predictions.

