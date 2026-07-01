import os
import argparse
import joblib
import pandas as pd
import numpy as np

# Define categorical mappings globally (same as in preprocessing.py)
from preprocessing import preprocess_inference_data

def get_performance_category(score: float) -> str:
    """
    Categorizes the score into a qualitative label.
    """
    if score >= 85.0:
        return 'Excellent'
    elif score >= 70.0:
        return 'Good'
    elif score >= 55.0:
        return 'Average'
    else:
        return 'Needs Improvement'

def predict_single(input_data: dict, models_dir: str = 'models') -> dict:
    """
    Predicts the final score and performance category for a single student.
    
    Args:
        input_data (dict): Dictionary with student features.
        models_dir (str): Directory containing saved preprocessing assets and models.
        
    Returns:
        dict: Predictions with confidence margins.
    """
    try:
        # Load best model and metadata
        model_path = os.path.join(models_dir, 'best_model.joblib')
        metadata_path = os.path.join(models_dir, 'best_model_metadata.joblib')
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Saved model or metadata not found. Run train.py first.")
            
        model = joblib.load(model_path)
        metadata = joblib.load(metadata_path)
        rmse = metadata.get('rmse', 12.0) # standard error default
        
        # Convert single input dict to DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Preprocess inference data (impute, encode, scale)
        df_preprocessed = preprocess_inference_data(df_input, models_dir=models_dir)
        
        # Predict score
        raw_pred = float(model.predict(df_preprocessed)[0])
        
        # Clip score between 0 and 100
        predicted_score = round(max(0.0, min(100.0, raw_pred)), 1)
        
        # Classify score
        category = get_performance_category(predicted_score)
        
        # Calculate confidence interval (95% confidence interval using 1.96 * RMSE)
        lower_bound = round(max(0.0, predicted_score - 1.96 * rmse), 1)
        upper_bound = round(min(100.0, predicted_score + 1.96 * rmse), 1)
        
        return {
            'predicted_score': predicted_score,
            'performance_category': category,
            'prediction_margin': round(1.96 * rmse, 1),
            'confidence_interval': (lower_bound, upper_bound),
            'model_used': metadata.get('best_model_name', 'Unknown')
        }
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        raise e

def main():
    parser = argparse.ArgumentParser(description="Predict student final score from input variables.")
    parser.add_argument('--sample', action='store_true', help="Run prediction on a sample student.")
    args = parser.parse_args()
    
    if args.sample:
        print("Running prediction on sample student...")
        # Create a sample student input dictionary
        sample_student = {
            'Gender': 'Female',
            'Age': 17,
            'Study_Hours': 12.5,
            'Attendance': 95.0,
            'Previous_Score': 78.0,
            'Parent_Education': 'Higher Education',
            'Internet_Access': 'Yes',
            'Extra_Activities': 'Yes',
            'Family_Support': 'Yes',
            'Sleep_Hours': 8.0
        }
        
        print("\nStudent Input Profile:")
        for k, v in sample_student.items():
            print(f"  {k}: {v}")
            
        result = predict_single(sample_student)
        
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Predicted Final Score: {result['predicted_score']}%")
        print(f"Performance Category:  {result['performance_category']}")
        print(f"Model Employed:        {result['model_used']}")
        print(f"95% Confidence Range:  {result['predicted_score']}% ± {result['prediction_margin']}% "
              f"({result['confidence_interval'][0]}% to {result['confidence_interval'][1]}%)")
        print("="*50 + "\n")
    else:
        print("Usage: python src/predict.py --sample")

if __name__ == '__main__':
    main()
