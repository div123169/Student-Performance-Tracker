import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils import load_data
from preprocessing import preprocess_train_data

def train_and_compare():
    print("Starting model training and comparison...")
    
    # 1. Load data
    data_path = 'dataset/student_performance.csv'
    df = load_data(data_path)
    
    # 2. Preprocess data
    X_train, X_test, y_train, y_test = preprocess_train_data(df, models_dir='models')
    
    # Save the splits for evaluation script to use directly
    os.makedirs('dataset', exist_ok=True)
    X_train.to_csv('dataset/X_train.csv', index=False)
    X_test.to_csv('dataset/X_test.csv', index=False)
    y_train.to_csv('dataset/y_train.csv', index=False)
    y_test.to_csv('dataset/y_test.csv', index=False)
    print("Saved train-test splits for evaluation script.")
    
    # 3. Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        'Support Vector Regressor': SVR(kernel='rbf', C=10.0, epsilon=0.1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    }
    
    # 4. Train and evaluate each model
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict on test set
        preds = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        
        results.append({
            'Model': name,
            'R2': round(r2, 4),
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4)
        })
        
    # 5. Display comparison
    results_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("MODEL COMPARISON (Sorted by R² Descending)")
    print("="*50)
    results_df_sorted = results_df.sort_values(by='R2', ascending=False).reset_index(drop=True)
    print(results_df_sorted)
    print("="*50 + "\n")
    
    # Save comparison dataframe to models folder
    results_df_sorted.to_csv('models/model_comparison.csv', index=False)
    
    # 6. Automatic model selection
    best_model_name = results_df_sorted.iloc[0]['Model']
    best_r2 = results_df_sorted.iloc[0]['R2']
    best_rmse = results_df_sorted.iloc[0]['RMSE']
    best_model = trained_models[best_model_name]
    
    print(f"Automatically selected the Best Model: {best_model_name} (R² = {best_r2:.4f}, RMSE = {best_rmse:.4f})")
    
    # 7. Save the best model
    best_model_path = 'models/best_model.joblib'
    joblib.dump(best_model, best_model_path)
    
    # Save metadata about best model
    metadata = {
        'best_model_name': best_model_name,
        'r2': best_r2,
        'rmse': best_rmse
    }
    joblib.dump(metadata, 'models/best_model_metadata.joblib')
    print(f"Saved the best model to {best_model_path}")
    print("Training process complete!")

if __name__ == '__main__':
    train_and_compare()
