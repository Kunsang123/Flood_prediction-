
import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

# Configuration
ARTIFACTS_PATH = "artifacts"
MODELS_PATH = os.path.join(ARTIFACTS_PATH, "models")
PROCESSED_PATH = "data/processed"

def generate_shap_plots():
    if not os.path.exists(MODELS_PATH):
        print(f"Error: Models path {MODELS_PATH} does not exist.")
        return

    # Load test data
    X_test_path = os.path.join(PROCESSED_PATH, "X_test.csv")
    X_train_path = os.path.join(PROCESSED_PATH, "X_train.csv")
    
    if not os.path.exists(X_test_path):
        print(f"Error: Test data {X_test_path} not found.")
        return
        
    X_test = pd.read_csv(X_test_path)
    X_train = pd.read_csv(X_train_path)

    for model_type in ["xgboost", "rf", "mlp"]:
        model_path = os.path.join(MODELS_PATH, f"{model_type}_model.pkl")
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found. Skipping.")
            continue
            
        print(f"Generating SHAP plot for {model_type}...")
        try:
            model = joblib.load(model_path)
            
            # Using a subset of X_train for the explainer if it's too large, 
            # but here 50k is manageable or we can use X_test for some explainers.
            # For tree models, TreeExplainer is fast.
            
            if model_type in ["xgboost", "rf"]:
                explainer = shap.TreeExplainer(model)
                # TreeExplainer usually doesn't need X_train as background if it's a tree model
                shap_values = explainer.shap_values(X_test)
            else:
                # For MLP, use KernelExplainer or DeepExplainer. 
                # KernelExplainer is slow, so we use a small background sample.
                background = shap.sample(X_train, 100)
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_test.head(200)) # Just 200 for speed if needed

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test if model_type != "mlp" else X_test.head(200), show=False)
            output_path = os.path.join(ARTIFACTS_PATH, f"{model_type}_shap.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            print(f"Successfully saved {output_path}")
            
        except Exception as e:
            print(f"Failed to generate SHAP plot for {model_type}: {e}")

if __name__ == "__main__":
    generate_shap_plots()
