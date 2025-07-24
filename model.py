# PRME_App/train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Assuming utils.py is in the same directory or accessible in the Python path
# If not, you might need to adjust the import or add the directory to the path
try:
    from utils import load_data, preprocess_data
except ImportError:
    print("Error: utils.py not found. Make sure it's in the same directory.")
    load_data = None
    preprocess_data = None

def train_model(data_path='PRME_App/data/sample_data.csv', model_save_path='PRME_App/random_forest_model.joblib'):
    """
    Trains a RandomForestRegressor model, evaluates it, and saves the trained model.

    Args:
        data_path (str): Path to the training data CSV file.
        model_save_path (str): Path to save the trained model.
    """
    if load_data is None or preprocess_data is None:
        print("Training aborted due to missing utils functions.")
        return

    print(f"Loading data from {data_path}...")
    df = load_data(data_path)

    if df is None:
        print("Failed to load data. Training aborted.")
        return

    print("Preprocessing data...")
    preprocessed_df = preprocess_data(df.copy()) # Use .copy() to avoid modifying original

    if preprocessed_df is None:
        print("Failed to preprocess data. Training aborted.")
        return

    # Define features (X) and target (y)
    # Ensure 'Machine ID' is not included in features
    if 'Machine ID' in preprocessed_df.columns:
        preprocessed_df = preprocessed_df.drop('Machine ID', axis=1)

    # Define the target variable 'risk_score'
    # Assuming 'risk_score' is present in the training data
    target_column = 'risk_score'
    if target_column not in preprocessed_df.columns:
        print(f"Error: Target column '{target_column}' not found in the data.")
        print("Training aborted.")
        return

    X = preprocessed_df.drop(target_column, axis=1)
    y = preprocessed_df[target_column]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Instantiate the model
    print("Instantiating RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42) # Using RandomForestRegressor as chosen

    # Train the model
    print("Training the model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Make predictions
    print("Making predictions on the test set...")
    y_pred = model.predict(X_test)
    print("Predictions complete.")

    # Evaluate the model
    print("Evaluating the model...")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # Save the trained model
    print(f"Saving the trained model to {model_save_path}...")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(model, model_save_path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving the model: {e}")

# Example usage: Run the training process
if __name__ == "__main__":
    # Create a dummy CSV file with 'risk_score' for testing purposes
    dummy_data = {
        'Machine ID': [f'M{i}' for i in range(1000)],
        'temperature': [49.963210, 96.057145, 78.559515, 67.892679, 32.481491] * 200,
        'vibration': [1.007151, 2.755315, 4.377435, 3.687902, 4.052150] * 200,
        'pressure': [89.255853, 87.046820, 185.938187, 87.431930, 90.792459] * 200,
        'runtime': [3396.244672, 4003.738846, 1327.292704, 3161.883088, 2901.555317] * 200,
        'maintenance_history': [1, 4, 4, 2, 3] * 200,
        'risk_score': [97.916431, 91.730073, 79.465822, 100.000000, 84.851845] * 200 # Dummy risk scores
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_csv_path = 'PRME_App/data/sample_data.csv'
    os.makedirs(os.path.dirname(dummy_csv_path), exist_ok=True)
    dummy_df.to_csv(dummy_csv_path, index=False)

    print(f"Created dummy training data at: {dummy_csv_path}")

    # Run the training process
    train_model(data_path=dummy_csv_path, model_save_path='PRME_App/random_forest_model.joblib')

