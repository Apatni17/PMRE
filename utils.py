# PRME_App/utils.py
import pandas as pd
import os

def load_data(file_path):
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame or None: The loaded DataFrame, or None if an error occurs.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def preprocess_data(df):
    """
    Preprocesses the input DataFrame by handling missing values.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    if df is None:
        print("Error: Input DataFrame is None.")
        return None

    # Handle missing numerical values by filling with the mean
    numerical_cols = df.select_dtypes(include=['number']).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)

    # Note: Add more preprocessing steps here as needed in the future (e.g., one-hot encoding for categorical features)

    return df

# Example usage (for testing during development)
# if __name__ == "__main__":
#     # Create a dummy CSV file for testing
#     dummy_data = {
#         'Machine ID': ['M1', 'M2', 'M3', 'M4'],
#         'Age': [5, 10, None, 8],
#         'Hours Used/Day': [12, None, 10, 15],
#         'Temperature': [35.5, 40.1, 38.0, None],
#         'Vibration Level': [1.2, 1.5, None, 1.1],
#         'Load %': [70, 85, 78, 90],
#         'Time Since Last Service': [100, 150, 120, None],
#         'Failure History': [0, 1, 0, 0]
#     }
#     dummy_df = pd.DataFrame(dummy_data)
#     dummy_csv_path = 'PRME_App/data/sample_data.csv'
#     os.makedirs(os.path.dirname(dummy_csv_path), exist_ok=True)
#     dummy_df.to_csv(dummy_csv_path, index=False)

#     print(f"Created dummy data at: {dummy_csv_path}")

#     # Load the dummy data
#     loaded_df = load_data(dummy_csv_path)
#     if loaded_df is not None:
#         print("\nOriginal DataFrame:")
#         display(loaded_df)

#         # Preprocess the loaded data
#         preprocessed_df = preprocess_data(loaded_df.copy()) # Use .copy() to avoid modifying original
#         print("\nPreprocessed DataFrame:")
#         display(preprocessed_df)
