import joblib
import pandas as pd
import os

# Define the path where the trained model is saved
MODEL_PATH = 'PRME_App/random_forest_model.joblib'

def load_model(model_path=MODEL_PATH):
    """
    Loads the trained machine learning model from the specified path.

    Args:
        model_path (str): The path to the saved model file.

    Returns:
        object or None: The loaded model object, or None if loading fails.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def assign_risk_category(score):
    """
    Assigns a risk category (low, medium, high) based on the risk score.

    Args:
        score (float): The predicted risk score.

    Returns:
        str: The risk category ('low', 'medium', 'high').
    """
    if score < 40:
        return 'low'
    elif 40 <= score <= 70:
        return 'medium'
    else:
        return 'high'

def predict_risk(data, model):
    """
    Predicts risk scores and assigns risk categories for new machine data.

    Args:
        data (pd.DataFrame): The preprocessed machine data (features).
        model: The trained machine learning model.

    Returns:
        tuple: A tuple containing:
            - predicted_scores (np.ndarray): Array of predicted risk scores.
            - risk_categories (pd.Series): Series of corresponding risk categories.
    """
    if model is None:
        print("Error: Model is not loaded.")
        return None, None

    if data is None or data.empty:
        print("Error: Input data is empty or None.")
        return None, None

    try:
        predicted_scores = model.predict(data)
        risk_categories = pd.Series([assign_risk_category(score) for score in predicted_scores], index=data.index)
        return predicted_scores, risk_categories
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# Example usage within an if __name__ == "__main__": block
if __name__ == "__main__":
    # Assume a dummy model has been trained and saved by train_model.py
    # and utils.py with preprocess_data exists

    # Create dummy data for prediction
    dummy_prediction_data = {
        'temperature': [55, 90, 30, 75, 45],
        'vibration': [1.1, 3.5, 0.8, 2.9, 1.5],
        'pressure': [100, 150, 70, 180, 95],
        'runtime': [2000, 3500, 500, 4000, 1500],
        'maintenance_history': [2, 4, 1, 3, 0]
    }
    sample_data_for_prediction = pd.DataFrame(dummy_prediction_data)

    print("Sample data for prediction:")
    display(sample_data_for_prediction)

    # Load the trained model
    loaded_model = load_model()

    if loaded_model:
        # Preprocess the sample data (using preprocess_data from utils)
        # Ensure utils is accessible in the Python path or copy the function here for this example
        try:
             from utils import preprocess_data
             preprocessed_sample_data = preprocess_data(sample_data_for_prediction.copy())
             print("\nPreprocessed sample data:")
             display(preprocessed_sample_data)

             # Make predictions
             predicted_scores, risk_categories = predict_risk(preprocessed_sample_data, loaded_model)

             if predicted_scores is not None and risk_categories is not None:
                 print("\nPrediction Results:")
                 results_df = pd.DataFrame({
                     'Predicted_Risk_Score': predicted_scores,
                     'Risk_Category': risk_categories
                 }, index=sample_data_for_prediction.index)
                 display(results_df)

        except ImportError:
            print("\nCould not import preprocess_data from utils. Please ensure utils.py is in the same directory.")
            print("Skipping preprocessing and prediction for this example.")
        except Exception as e:
             print(f"\nAn error occurred during preprocessing or prediction: {e}")
