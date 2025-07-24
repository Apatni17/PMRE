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
    preprocessed_df = preprocess_data(df.copy())

    if preprocessed_df is None:
        print("Failed to preprocess data. Training aborted.")
        return

    # Drop non-feature columns
    for col in ['Machine ID', 'risk_score']:
        if col in preprocessed_df.columns:
            preprocessed_df = preprocessed_df.drop(col, axis=1)

    # Features and target
    X = preprocessed_df.copy()
    y = df['risk_score']

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Print risk score distribution for verification
    print("Risk score distribution (bins):")
    print(pd.cut(y, bins=[0,39,70,100]).value_counts())

    # Split data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Train model
    print("Instantiating RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("Training the model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate
    print("Making predictions on the test set...")
    y_pred = model.predict(X_test)
    print("Predictions complete.")
    print("Evaluating the model...")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # Save model
    print(f"Saving the trained model to {model_save_path}...")
    try:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(model, model_save_path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving the model: {e}")

if __name__ == "__main__":
    train_model(data_path='PRME_App/data/sample_data.csv', model_save_path='PRME_App/random_forest_model.joblib')

