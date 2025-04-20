import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np
import time

# --- Configuration --- 
DATA_FILES = ['copd_15_features_1000_rows.xlsx'] 
MODEL_FILE = 'lung_disease_model.plk'
SCALER_FILE = 'scaler.plk'

# Target column from the compatible dataset
target_col = 'COPD_Diagnosis' 

# Columns to drop (if any)
cols_to_drop = [] 
# --- End Configuration --- 

def load_data(file_path):
    """Loads data from a single Excel file."""
    print(f"Attempting to load data from: {file_path}")
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path)
            print(f"Successfully loaded {file_path}. Shape: {df.shape}")
            print(f"Initial columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}. Cannot proceed.")
            return None
    else:
        print(f"File not found: {file_path}. Cannot proceed.")
        return None

def train_model():
    """
    Loads data, defines features, preprocesses, tunes with lightweight GridSearchCV, 
    evaluates, and saves the final model and scaler.
    """
    df = load_data(DATA_FILES[0]) # Load the single specified file

    if df is None:
        return
        
    try:
        # --- Define Feature list and Target column --- 
        if target_col not in df.columns:
            print(f"Error: Target column '{target_col}' not found in the dataset.")
            print(f"Available columns: {df.columns.tolist()}")
            return
            
        # Use all columns except the target as features
        feature_cols = [col for col in df.columns if col != target_col]
        print(f"Using target column: '{target_col}'")
        print(f"Using features: {feature_cols}")

        # --- Drop specified columns (if any) --- 
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
             print(f"Dropping columns: {existing_cols_to_drop}")
             df = df.drop(existing_cols_to_drop, axis=1)
             feature_cols = [col for col in feature_cols if col not in existing_cols_to_drop]
             print(f"Features after dropping: {feature_cols}")

        # --- Check if final features and target exist --- 
        final_required_cols = feature_cols + [target_col]
        missing_cols = [col for col in final_required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Required columns are missing after processing: {missing_cols}")
            return

        X = df[feature_cols]
        y = df[target_col]
        
        # --- Lightweight Feature Engineering ---
        # Add only the most important engineered features
        
        # Calculate symptom score (sum of all symptoms)
        symptom_cols = ['Chronic_Cough', 'Shortness_of_Breath', 'Wheezing', 
                        'Sputum_Production', 'Respiratory_Infections', 'Allergies']
        X['Symptom_Score'] = X[symptom_cols].sum(axis=1)
        
        # Add most important interaction
        X['Smoking_Air_Pollution'] = X['Smoking_History'] * X['Air_Pollution_Exposure']
        
        # Convert Gender to dummy variable
        X = pd.get_dummies(X, columns=['Gender'], drop_first=True)
        
        # Update feature cols after engineering
        feature_cols = X.columns.tolist()
        print(f"Features after engineering: {feature_cols}")

        # --- Handle potential missing values (NaNs) --- 
        print("Checking for and handling potential missing values (NaNs)...")
        if y.isnull().sum() > 0:
            print(f"Warning: Found {y.isnull().sum()} missing values in target '{target_col}'. Filling with mode.")
            y = y.fillna(y.mode()[0])
            
        if X.isnull().sum().sum() > 0:
            print(f"Warning: Found {X.isnull().sum().sum()} missing values in features. Filling with column means.")
            X = X.fillna(X.mean())

        # Ensure data types are suitable
        X = X.astype(float)
        y = y.astype(int)

        # --- Data Splitting with Stratification --- 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Data split into training ({len(X_train)}) and testing ({len(X_test)}).")
        
        # --- Feature Scaling --- 
        print("Applying StandardScaler to features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # --- Lightweight Hyperparameter Tuning --- 
        print("\nStarting lightweight GridSearchCV...")
        start_time = time.time()
        
        # Simplified parameter grid with fewer options
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, None],
            'min_samples_split': [5],
            'class_weight': ['balanced']
        }
        
        # Use 3-fold CV instead of 5 for speed
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42), 
            param_grid=param_grid, 
            cv=3, 
            n_jobs=-1, 
            scoring='roc_auc',
            verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        end_time = time.time()
        print(f"GridSearchCV finished in {end_time - start_time:.2f} seconds.")
        
        best_model = grid_search.best_estimator_
        print(f"\nBest parameters found by GridSearchCV: {grid_search.best_params_}")
        print(f"Best cross-validation AUC: {grid_search.best_score_:.4f}")

        # --- Evaluating Best Model on Test Set --- 
        print("\n--- Evaluating Best Model on Test Set ---")
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Accuracy on Test Set: {accuracy:.4f}")
        print(f"F1-Score on Test Set: {f1:.4f}")
        print(f"ROC-AUC on Test Set: {auc:.4f}")
        
        print("\nClassification Report on Test Set:")
        try:
            target_names = ['Class 0 (Negative)', 'Class 1 (Positive)'] 
            print(classification_report(y_test, y_pred, target_names=target_names))
        except Exception as report_e:
            print(f"Could not generate classification report: {report_e}")
            print(classification_report(y_test, y_pred))
        
        # --- Feature Importance Analysis ---
        print("\nFeature Importance Analysis:")
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Feature ranking:")
        for i, idx in enumerate(indices[:10]):  # Show top 10
            print(f"{i+1}. {feature_cols[idx]} ({importances[idx]:.4f})")

        # --- Save the final model and scaler ---
        print(f"\nSaving the scaler object to {SCALER_FILE}...")
        joblib.dump(scaler, SCALER_FILE)
        
        print(f"Saving the best trained model to {MODEL_FILE}...")
        joblib.dump(best_model, MODEL_FILE)
        
        # Also save feature columns list for prediction
        feature_cols_file = 'feature_cols.pkl'
        joblib.dump(feature_cols, feature_cols_file)
        print(f"Saving feature columns to {feature_cols_file}")
        
        print(f"Model successfully saved to {MODEL_FILE}")

    except Exception as e:
        import traceback
        print(f"\n--- An error occurred during model training --- ")
        traceback.print_exc()

if __name__ == '__main__':
    train_model() 