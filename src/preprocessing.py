import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define categorical mappings globally
CATEGORICAL_MAPPINGS = {
    'Gender': {'Male': 0, 'Female': 1},
    'Parent_Education': {
        'None': 0,
        'Primary Education': 1,
        '5th to 9th Grade': 2,
        'Secondary Education': 3,
        'Higher Education': 4
    },
    'Internet_Access': {'No': 0, 'Yes': 1},
    'Extra_Activities': {'No': 0, 'Yes': 1},
    'Family_Support': {'No': 0, 'Yes': 1}
}

# Define column groups
NUMERIC_COLS = ['Age', 'Study_Hours', 'Attendance', 'Previous_Score', 'Sleep_Hours']
CATEGORICAL_COLS = ['Gender', 'Parent_Education', 'Internet_Access', 'Extra_Activities', 'Family_Support']
TARGET_COL = 'Final_Score'

def preprocess_train_data(df: pd.DataFrame, models_dir: str = 'models') -> tuple:
    """
    Cleans raw data, trains and saves the imputer values & scaler, 
    encodes features, and splits into train and test sets.
    
    Args:
        df (pd.DataFrame): Raw input data.
        models_dir (str): Directory where preprocessing assets will be saved.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    try:
        print("Starting preprocessing for training data...")
        df_clean = df.copy()
        
        # 1. Duplicate Handling
        duplicate_count = df_clean.duplicated().sum()
        if duplicate_count > 0:
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)
            print(f"Dropped {duplicate_count} duplicate rows.")
            
        # 2. Missing Value Analysis & Storage
        imputation_values = {}
        for col in NUMERIC_COLS:
            # Impute numeric features with median
            median_val = float(df_clean[col].median(skipna=True))
            imputation_values[col] = median_val
            df_clean[col] = df_clean[col].fillna(median_val)
            
        for col in CATEGORICAL_COLS:
            # Impute categorical features with mode
            mode_series = df_clean[col].mode(dropna=True)
            mode_val = mode_series.iloc[0] if not mode_series.empty else 'Unknown'
            imputation_values[col] = mode_val
            df_clean[col] = df_clean[col].fillna(mode_val)
            
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(imputation_values, os.path.join(models_dir, 'imputation_values.joblib'))
        print("Imputed missing values and saved imputation parameters.")
        
        # 3. Categorical Encoding
        for col, mapping in CATEGORICAL_MAPPINGS.items():
            if col in df_clean.columns:
                # Handle unseen values by mapping them to median or default (0)
                df_clean[col] = df_clean[col].map(mapping).fillna(0).astype(int)
        print("Categorical features encoded successfully.")
        
        # 4. Feature Scaling (Only on numeric features)
        scaler = StandardScaler()
        # Scale numeric features
        df_clean[NUMERIC_COLS] = scaler.fit_transform(df_clean[NUMERIC_COLS])
        joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
        print("Numeric features scaled and scaler saved.")
        
        # 5. Split Features and Target
        X = df_clean[NUMERIC_COLS + CATEGORICAL_COLS]
        y = df_clean[TARGET_COL]
        
        # 6. Train-Test Split (80:20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Train/Test split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error in training preprocessing pipeline: {e}")
        raise e

def preprocess_inference_data(df: pd.DataFrame, models_dir: str = 'models') -> pd.DataFrame:
    """
    Applies saved imputer parameters, custom encodings, and saved scaler
    to inference/prediction data.
    
    Args:
        df (pd.DataFrame): New raw student data.
        models_dir (str): Directory where preprocessing assets are saved.
        
    Returns:
        pd.DataFrame: Scaled and encoded features ready for prediction.
    """
    try:
        df_clean = df.copy()
        
        # Load preprocessing assets
        imputation_path = os.path.join(models_dir, 'imputation_values.joblib')
        scaler_path = os.path.join(models_dir, 'scaler.joblib')
        
        if not os.path.exists(imputation_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Preprocessing assets not found. Run training first.")
            
        imputation_values = joblib.load(imputation_path)
        scaler = joblib.load(scaler_path)
        
        # 1. Missing Value Imputation
        for col in NUMERIC_COLS + CATEGORICAL_COLS:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(imputation_values[col])
            else:
                # If feature is missing completely, inject imputed training default
                df_clean[col] = imputation_values[col]
                
        # 2. Categorical Encoding
        for col, mapping in CATEGORICAL_MAPPINGS.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map(mapping).fillna(0).astype(int)
                
        # 3. Feature Scaling (Using the fit scaler)
        df_clean[NUMERIC_COLS] = scaler.transform(df_clean[NUMERIC_COLS])
        
        # Reorder columns to match the model training layout
        final_cols = NUMERIC_COLS + CATEGORICAL_COLS
        return df_clean[final_cols]
        
    except Exception as e:
        print(f"Error in inference preprocessing pipeline: {e}")
        raise e
