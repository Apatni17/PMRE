import streamlit as st
import pandas as pd
import os
import sys
import json

try:
    # Attempt direct import assuming files are accessible
    from utils import load_data, preprocess_data
    from predict import load_model, predict_risk, assign_risk_category # Also import assign_risk_category for coloring
    from chatbot import get_maintenance_advice
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop() # Stop the app if essential modules can't be imported

# Define paths - adjust to current directory if necessary for notebook execution
# Assuming the model and data directories are relative to where this notebook is run
DATA_DIR = 'PRME_App/data' # Assuming PRME_App directory is in the current path
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, 'sample_data.csv')
MODEL_PATH = 'PRME_App/random_forest_model.joblib' # Assuming PRME_App directory is in the current path
KNOWLEDGE_BASE_PATH = os.path.join(DATA_DIR, 'knowledge_base.json')

# Ensure the PRME_App directory and data subdirectory exist for saving/loading
os.makedirs(DATA_DIR, exist_ok=True)
# Create dummy sample data and knowledge base if they don't exist, for demonstration
if not os.path.exists(SAMPLE_DATA_PATH):
    dummy_data = {
        'Machine ID': [f'M{i}' for i in range(10)],
        'temperature': [49.96, 96.05, 78.55, 67.89, 32.48, 55.0, 90.0, 30.0, 75.0, 45.0],
        'vibration': [1.00, 2.75, 4.37, 3.68, 4.05, 1.1, 3.5, 0.8, 2.9, 1.5],
        'pressure': [89.25, 87.04, 185.93, 87.43, 90.79, 100.0, 150.0, 70.0, 180.0, 95.0],
        'runtime': [3396, 4003, 1327, 3161, 2901, 2000, 3500, 500, 4000, 1500],
        'maintenance_history': [1, 4, 4, 2, 3, 2, 4, 1, 3, 0],
        'risk_score': [97.91, 91.73, 79.46, 100.00, 84.85, 60.0, 80.0, 30.0, 90.0, 50.0] # Dummy risk scores
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv(SAMPLE_DATA_PATH, index=False)
    st.info(f"Created dummy sample data at: {SAMPLE_DATA_PATH}")

if not os.path.exists(KNOWLEDGE_BASE_PATH):
    dummy_kb_data = {
        "high_risk_causes": ["Excessive vibration indicates bearing wear.", "High temperature suggests friction."],
        "preventive_actions_high_risk": ["Inspect bearings.", "Check lubrication."],
        "medium_risk_causes": ["Slightly elevated temperature.", "Intermittent vibration."],
        "preventive_actions_medium_risk": ["Monitor closely.", "Check fasteners."],
        "low_risk_advice": ["Continue routine monitoring."]
    }
    with open(KNOWLEDGE_BASE_PATH, 'w') as f:
        json.dump(dummy_kb_data, f, indent=2)
    st.info(f"Created dummy knowledge base at: {KNOWLEDGE_BASE_PATH}")


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="PRME: Predictive Risk Maintenance Evaluator",
    layout="wide"
)

st.title("🏭 PRME: Predictive Risk Maintenance Evaluator")

# --- File Upload ---
st.header("Upload Machine Data")
uploaded_file = st.file_uploader("Upload your machine data CSV file (e.g., sample_data.csv)", type="csv")


# Use Streamlit's session state to manage data across reruns
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'preprocessed_df' not in st.session_state:
    st.session_state['preprocessed_df'] = None
if 'results_df' not in st.session_state:
    st.session_state['results_df'] = None
if 'model' not in st.session_state:
     # Load model once and store in session state
     st.session_state['model'] = load_model(model_path=MODEL_PATH)
if 'uploaded_file_name' not in st.session_state:
     st.session_state['uploaded_file_name'] = None

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


model = st.session_state['model']

# Flag to indicate if data is being loaded (uploaded or sample)
loading_data = False

if uploaded_file is not None:
    # Check if a new file is uploaded by comparing the file name
    if st.session_state['uploaded_file_name'] != uploaded_file.name:
         st.info("Processing uploaded file...")
         loading_data = True
         st.session_state['uploaded_file_name'] = uploaded_file.name # Store file name
         st.session_state['chat_history'] = [] # Clear chat history on new file upload

         try:
             # Load data using utils function
             # Streamlit's file uploader provides a file-like object
             st.session_state['df'] = load_data(uploaded_file)
             if st.session_state['df'] is not None:
                 st.success("Data loaded successfully!")
                 # Reset preprocessed and results when new data is loaded
                 st.session_state['preprocessed_df'] = None
                 st.session_state['results_df'] = None
             else:
                 st.error("Failed to load data from the uploaded file.")
                 st.session_state['df'] = None # Reset state if loading fails
                 st.session_state['preprocessed_df'] = None
                 st.session_state['results_df'] = None

         except Exception as e:
             st.error(f"An error occurred during data loading: {e}")
             st.session_state['df'] = None
             st.session_state['preprocessed_df'] = None
             st.session_state['results_df'] = None

    else:
         st.info("Using previously uploaded data.")
         loading_data = True # Assume data is loaded if using previous

elif st.session_state['df'] is None:
    st.info("Please upload a CSV file to get started.")
    # Option to load sample data if no file is uploaded and no data in state
    if st.button("Load Sample Data"):
         if os.path.exists(SAMPLE_DATA_PATH):
             st.session_state['load_sample'] = True
         else:
             st.error(f"Sample data file not found at {SAMPLE_DATA_PATH}")


# Handle sample data loading if button was clicked and no uploaded file
if 'load_sample' in st.session_state and st.session_state['load_sample'] and uploaded_file is None:
    del st.session_state['load_sample'] # Clear the state
    if os.path.exists(SAMPLE_DATA_PATH):
        st.info("Loading sample data...")
        loading_data = True
        st.session_state['chat_history'] = [] # Clear chat history on loading sample data

        try:
            st.session_state['df'] = load_data(SAMPLE_DATA_PATH)
            if st.session_state['df'] is not None:
                st.success("Sample data loaded successfully!")
                # Reset preprocessed and results when new data is loaded
                st.session_state['preprocessed_df'] = None
                st.session_state['results_df'] = None
                 # Also reset uploaded file info if sample data is loaded
                st.session_state['uploaded_file_name'] = None

            else:
                 st.error("Failed to load sample data file.")
                 st.session_state['df'] = None
                 st.session_state['preprocessed_df'] = None
                 st.session_state['results_df'] = None

        except Exception as e:
            st.error(f"An error occurred while loading sample data: {e}")
            st.session_state['df'] = None
            st.session_state['preprocessed_df'] = None
            st.session_state['results_df'] = None


# --- Data Processing and Prediction (only if data is loaded or in state) ---
if st.session_state['df'] is not None:
    st.write("Original Data Preview:")
    st.dataframe(st.session_state['df'].head())

    if st.session_state['preprocessed_df'] is None:
        st.info("Preprocessing data...")
        st.session_state['preprocessed_df'] = preprocess_data(st.session_state['df'].copy())
        if st.session_state['preprocessed_df'] is not None:
             st.success("Data preprocessed successfully!")
        else:
             st.error("Failed to preprocess data.")

    if st.session_state['preprocessed_df'] is not None:
        # Drop non-feature columns like 'Machine ID' before prediction
        df_display = st.session_state['df'].copy() # Keep a copy with Machine ID for display

        preprocessed_features = st.session_state['preprocessed_df'].copy()
        for col in ['Machine ID', 'risk_score']:
            if col in preprocessed_features.columns:
                preprocessed_features = preprocessed_features.drop(col, axis=1)
        
        st.write("Preprocessed Data Preview (Features for Model):")
        st.dataframe(preprocessed_features.head())

        # --- Model Prediction ---
        st.header("Machine Risk Dashboard")

        if st.session_state['results_df'] is None:
             if model is not None:
                 st.info("Predicting risk...")
                 # Predict risk
                 predicted_scores, risk_categories = predict_risk(preprocessed_features, model)

                 if predicted_scores is not None and risk_categories is not None:
                     st.success("Risk prediction complete!")
                     # Combine results with original data for display
                     st.session_state['results_df'] = df_display.copy()
                     st.session_state['results_df']['Predicted_Risk_Score'] = predicted_scores
                     st.session_state['results_category'] = risk_categories # Store series separately
                     st.session_state['results_df']['Risk_Category'] = risk_categories

                 else:
                      st.error("Failed to predict risk scores.")
             else:
                 st.warning(f"Could not load the trained model from {MODEL_PATH}. Please ensure 'train_model.py' has been run.")
        else:
            st.info("Using previously computed risk predictions.")


    # --- Display Results and Dashboard (if results_df is available) ---
    if st.session_state['results_df'] is not None:
        st.write("Risk Evaluation Results:")

        # Add Filtering Options
        all_risk_categories = ['low', 'medium', 'high']
        selected_risk_categories = st.multiselect(
            "Filter by Risk Category",
            options=all_risk_categories,
            default=all_risk_categories # Default to selecting all
        )

        # Apply Filtering
        filtered_results_df = st.session_state['results_df'].copy()
        if selected_risk_categories: # Only filter if categories are selected
            filtered_results_df = filtered_results_df[filtered_results_df['Risk_Category'].isin(selected_risk_categories)]
        else:
            st.info("Select at least one risk category to display results.")
            filtered_results_df = pd.DataFrame() # Show empty if nothing is selected


        # Define a function for row coloring
        def color_risk_category(row):
           if row['Risk_Category'] == 'high':
               return ['background-color: #ffcccc'] * len(row) # Light red
           elif row['Risk_Category'] == 'medium':
               return ['background-color: #ffffcc'] * len(row) # Light yellow
           elif row['Risk_Category'] == 'low':
               return ['background-color: #ccffcc'] * len(row) # Light green
           else:
               return [''] * len(row)

        # Apply the coloring function to the filtered DataFrame
        if not filtered_results_df.empty:
            styled_filtered_results_df = filtered_results_df.style.apply(color_risk_category, axis=1)
            st.dataframe(styled_filtered_results_df, use_container_width=True)

            # Optional: Display summary statistics for filtered data
            if not filtered_results_df.empty:
                risk_counts = filtered_results_df['Risk_Category'].value_counts().reset_index()
                risk_counts.columns = ['Risk Category', 'Number of Machines']
                st.write("Summary of Machines by Filtered Risk Category:")
                st.dataframe(risk_counts)
            else:
                 st.write("No machines match the selected filter criteria.")

        else:
            st.write("No data to display based on the selected filters.")


        # --- Maintenance Chatbot ---
        st.header("Maintenance Chatbot")

        selected_machine_id = None
        selected_machine_data = None
        selected_risk_category = None

        # Use the filtered_results_df for machine selection
        if not filtered_results_df.empty:
            # Ensure 'Machine ID' column exists before using it for selection
            if 'Machine ID' in filtered_results_df.columns:
                machine_ids_list = filtered_results_df['Machine ID'].tolist()
                selected_machine_id = st.selectbox("Select a Machine to get advice:", machine_ids_list, key='chatbot_machine_select')

                if selected_machine_id:
                    selected_row = filtered_results_df[filtered_results_df['Machine ID'] == selected_machine_id].iloc[0]
                    # Extract relevant parameters for the chatbot
                    # Exclude Machine ID, Predicted_Risk_Score, Risk_Category
                    selected_machine_data = selected_row.drop(['Machine ID', 'Predicted_Risk_Score', 'Risk_Category']).to_dict()
                    selected_risk_category = selected_row['Risk_Category']

                    st.write(f"Selected Machine: **{selected_machine_id}**")
                    st.write(f"Predicted Risk Level: **{selected_risk_category}**")
                    st.write("Parameters:")
                    st.json(selected_machine_data)

            else:
                 st.warning("'Machine ID' column not found in the filtered data. Cannot select a specific machine for the chatbot.")
        else:
             st.info("No data available in the filtered view to select a machine for the chatbot.")


        # --- Interactive Chat Interface ---
        if selected_machine_data is not None:
            st.subheader("Chat with the Maintenance Assistant")

            # Display chat messages from history
            for message in st.session_state['chat_history']:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            user_input = st.chat_input("Enter your message:")

            # Process user input
            if user_input:
                # Add user message to chat history
                st.session_state['chat_history'].append({"role": "user", "content": user_input})

                # Get chatbot response
                with st.spinner("Thinking..."):
                     # Call the chatbot function
                     # Pass the selected machine's *parameters* and risk level
                     advice = get_maintenance_advice(selected_machine_data, selected_risk_category)

                # Add chatbot response to chat history
                st.session_state['chat_history'].append({"role": "assistant", "content": advice})

                # Rerun the app to display the new messages
                st.rerun()

            # Button to clear chat history
            if st.button("Clear Chat History"):
                st.session_state['chat_history'] = []
                st.rerun()

        # End of Chat Interface
# End of Data Loaded Block


# Add some basic instructions
st.sidebar.header("Instructions")
st.sidebar.write("1. Upload your machine data CSV file or load sample data.")
st.sidebar.write("2. The app will predict risk scores and display results.")
st.sidebar.write("3. Use the filter options to narrow down the displayed machines.")
st.sidebar.write("4. Select a machine from the dropdown in the Chatbot section.")
st.sidebar.write("5. Use the chat input field to ask questions about maintenance for the selected machine.")
st.sidebar.write("6. Click 'Clear Chat History' to start a new conversation.")
