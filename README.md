# RESPIROGUIDE: COPD Risk Prediction

RESPIROGUIDE is a web application built with Flask that predicts the likelihood of Chronic Obstructive Pulmonary Disease (COPD) based on patient data. It utilizes a machine learning model (RandomForestClassifier) trained on the `copd_15_features_1000_rows.xlsx` dataset.

The application takes 15 patient characteristics as input and predicts the COPD diagnosis (Positive/Negative) along with an estimated probability.

## Setup and Running Locally

1.  **Prerequisites:**
    *   Python (>=3.7 recommended)
    *   pip (Python package installer)
    *   Git (optional, for cloning)

2.  **Get the Code:**
    *   Clone the repository (if available):
        ```bash
        git clone <repository_url>
        cd copd__proj 
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

1. **Probability Assessment**: The model calculates a probability score (0-100%) for COPD risk based on all input features.

2. **Decision Criteria**: 
   - If the calculated probability is ≥40%, the prediction status is set to "Positive" (Suffering from COPD)
   - If the patient has a majority of respiratory symptoms (at least 3 out of 6 key symptoms), the prediction may also be "Positive" even with a lower probability score
   - Otherwise, the prediction status is "Negative" (Not Suffering from COPD)

3. **Key symptoms** considered in the assessment:
   - Chronic Cough
   - Shortness of Breath
   - Wheezing
   - Sputum Production
   - Respiratory Infections
   - Allergies

This multi-factor approach helps ensure that both the statistical probability and actual clinical symptoms are considered in the final prediction.

