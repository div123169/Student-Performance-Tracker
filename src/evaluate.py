import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

from utils import configure_plotting_style

def evaluate_best_model():
    print("Starting evaluation of the best model...")
    
    # 1. Setup paths and directories
    os.makedirs('notebooks/plots', exist_ok=True)
    configure_plotting_style()
    
    # Load dataset train-test splits
    try:
        X_test = pd.read_csv('dataset/X_test.csv')
        y_test = pd.read_csv('dataset/y_test.csv').squeeze()
        X_train = pd.read_csv('dataset/X_train.csv')
        y_train = pd.read_csv('dataset/y_train.csv').squeeze()
    except FileNotFoundError:
        print("Train/test splits not found. Please run training script first.")
        return
        
    # Load the best model and metadata
    if not os.path.exists('models/best_model.joblib') or not os.path.exists('models/best_model_metadata.joblib'):
        print("Saved best model or metadata not found. Please run training script first.")
        return
        
    best_model = joblib.load('models/best_model.joblib')
    metadata = joblib.load('models/best_model_metadata.joblib')
    best_name = metadata['best_model_name']
    
    # 2. Make predictions
    preds = best_model.predict(X_test)
    
    # 3. Calculate evaluation metrics
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    
    print("\n" + "="*50)
    print(f"EVALUATION SUMMARY FOR: {best_name}")
    print("="*50)
    print(f"R² Score:                  {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE):  {mse:.4f}")
    print(f"Root Mean Sq. Error (RMSE): {rmse:.4f}")
    print("="*50 + "\n")
    
    # Save test evaluation metrics
    with open('models/evaluation_summary.txt', 'w') as f:
        f.write(f"Best Model Name: {best_name}\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
    
    # 4. Generate & Save Plots
    
    # Plot 1: Actual vs. Predicted Scores
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=preds, alpha=0.7, color='#4E79A7', s=80)
    # 45-degree reference line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    plt.title(f'Actual vs. Predicted Final Scores\n(Model: {best_name})', fontsize=14)
    plt.xlabel('Actual Final Score (%)', fontsize=12)
    plt.ylabel('Predicted Final Score (%)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    actual_vs_pred_path = 'notebooks/plots/actual_vs_predicted.png'
    plt.savefig(actual_vs_pred_path)
    plt.close()
    print(f"Saved: {actual_vs_pred_path}")
    
    # Plot 2: Residual Plot
    residuals = y_test - preds
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=preds, y=residuals, alpha=0.7, color='#E15759', s=80)
    plt.axhline(0, color='black', linestyle='--', lw=2)
    plt.title('Residuals Plot (Homoscedasticity Check)', fontsize=14)
    plt.xlabel('Predicted Final Score (%)', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    residuals_plot_path = 'notebooks/plots/residuals_plot.png'
    plt.savefig(residuals_plot_path)
    plt.close()
    print(f"Saved: {residuals_plot_path}")
    
    # Plot 3: Feature Importance or Coefficients
    plt.figure(figsize=(9, 6))
    features = X_train.columns
    
    if hasattr(best_model, 'feature_importances_'):
        # For tree-based models: Random Forest, Decision Tree, Gradient Boosting
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        sns.barplot(x=importances[indices], y=features[indices], palette='viridis', hue=features[indices], legend=False)
        plt.title(f'Feature Importances (Model: {best_name})', fontsize=14)
        plt.xlabel('Relative Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
    elif hasattr(best_model, 'coef_'):
        # For Linear Regression
        coefs = best_model.coef_
        indices = np.argsort(np.abs(coefs))[::-1]
        
        sns.barplot(x=coefs[indices], y=features[indices], palette='coolwarm', hue=features[indices], legend=False)
        plt.title(f'Model Coefficients (Model: {best_name})', fontsize=14)
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
    else:
        # Default fallback: Permutation Importance (useful for SVR)
        print("Calculating permutation importance for SVR...")
        perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
        importances = perm_importance.importances_mean
        indices = np.argsort(importances)[::-1]
        
        sns.barplot(x=importances[indices], y=features[indices], palette='magma', hue=features[indices], legend=False)
        plt.title(f'Permutation Feature Importances (Model: {best_name})', fontsize=14)
        plt.xlabel('Importance Decrease in R²', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
    plt.tight_layout()
    feature_importance_path = 'notebooks/plots/feature_importance.png'
    plt.savefig(feature_importance_path)
    plt.close()
    print(f"Saved: {feature_importance_path}")
    
    print("\nModel evaluation completed successfully! Evaluation summaries and plots exported.")

if __name__ == '__main__':
    evaluate_best_model()
