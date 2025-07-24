import openai
import os
import json
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_knowledge_base(kb_path='PRME_App/data/knowledge_base.json'):
    """
    Loads a knowledge base from a JSON file.

    Args:
        kb_path (str): The path to the JSON knowledge base file.

    Returns:
        dict or None: The loaded knowledge base dictionary, or None if loading fails.
    """
    if not os.path.exists(kb_path):
        print(f"Warning: Knowledge base file not found at {kb_path}. Chatbot will rely solely on general knowledge.")
        return None
    try:
        with open(kb_path, 'r') as f:
            knowledge_base = json.load(f)
        print(f"Knowledge base loaded successfully from {kb_path}")
        return knowledge_base
    except Exception as e:
        print(f"Error loading knowledge base from {kb_path}: {e}")
        return None

# Load the knowledge base when the module is imported
knowledge_base = load_knowledge_base()

def get_maintenance_advice(machine_params, risk_category):
    """
    Gets maintenance advice from the OpenAI API based on machine parameters and risk level.

    Args:
        machine_params (dict): A dictionary of machine parameters (e.g., {'temperature': 60, 'vibration': 2.5, ...}).
        risk_category (str): The predicted risk category ('low', 'medium', 'high').

    Returns:
        str: The chatbot's generated maintenance advice.
    """
    if not openai.api_key:
        return "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."

    prompt = f"""You are an AI assistant specializing in predictive maintenance for industrial machines.
Your goal is to provide maintenance advice based on machine parameters and a predicted risk level.
Explain potential failure causes for the given risk level and suggest preventive actions.
Keep the advice concise and actionable.

Machine Parameters:
{json.dumps(machine_params, indent=2)}

Predicted Risk Level: {risk_category}

"""

    if knowledge_base:
        prompt += f"""\nReference the following knowledge base for specific advice if relevant:
{json.dumps(knowledge_base, indent=2)}
"""

    prompt += """\nBased on the above information, provide maintenance advice.
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", # You can choose a different model if needed
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in machine maintenance."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300 # Limit response length
        )
        # Accessing the content correctly from the new API response object
        advice = response.choices[0].message.content.strip()
        return advice
    except Exception as e:
        return f"Error getting advice from OpenAI API: {e}"

# Example usage (for testing during development)
if __name__ == "__main__":
    # Create a dummy knowledge base file for testing if it doesn't exist
    dummy_kb_path = 'PRME_App/data/knowledge_base.json'
    if not os.path.exists(dummy_kb_path):
        dummy_kb_data = {
            "high_risk_causes": [
                "Excessive vibration often indicates bearing wear or misalignment.",
                "High temperature can be caused by friction, poor lubrication, or cooling system issues.",
                "Low pressure in hydraulic systems might point to leaks or pump problems."
            ],
            "preventive_actions_high_risk": [
                "Schedule immediate inspection of components showing abnormal readings.",
                "Check lubrication levels and quality.",
                "Verify cooling system functionality.",
                "Inspect for leaks and unusual noises."
            ],
            "medium_risk_causes": [
                 "Slightly elevated temperature might indicate increased load or minor friction.",
                 "Intermittent vibration could be due to minor imbalances or loose parts."
            ],
            "preventive_actions_medium_risk": [
                 "Monitor parameters closely.",
                 "Perform scheduled maintenance proactively.",
                 "Check for loose fasteners."
            ],
             "low_risk_advice": [
                 "Machine is operating within normal parameters. Continue routine monitoring and scheduled maintenance."
            ]
        }
        os.makedirs(os.path.dirname(dummy_kb_path), exist_ok=True)
        with open(dummy_kb_path, 'w') as f:
            json.dump(dummy_kb_data, f, indent=2)
        print(f"Created dummy knowledge base at: {dummy_kb_path}")
        # Reload knowledge base after creating the file
        knowledge_base = load_knowledge_base(dummy_kb_path)


    # Sample machine data and risk levels for testing
    sample_machine_data_row = {'temperature': 49.96, 'vibration': 4.85, 'pressure': 54.71, 'runtime': 4550.50, 'maintenance_history': 3}
    sample_machine_index = 0 # Use an index from the original dataframe if available

    # Example 1: High Risk
    print("\n--- High Risk Example ---")
    high_risk_params = {'temperature': 95.0, 'vibration': 4.5, 'pressure': 80.0, 'runtime': 3000, 'maintenance_history': 5}
    high_risk_advice = get_maintenance_advice(high_risk_params, 'high')
    print(f"Machine Parameters: {high_risk_params}")
    print(f"Risk Level: high")
    print(f"Maintenance Advice:\n{high_risk_advice}")

    # Example 2: Medium Risk
    print("\n--- Medium Risk Example ---")
    medium_risk_params = {'temperature': 65.0, 'vibration': 2.0, 'pressure': 120.0, 'runtime': 1500, 'maintenance_history': 2}
    medium_risk_advice = get_maintenance_advice(medium_risk_params, 'medium')
    print(f"Machine Parameters: {medium_risk_params}")
    print(f"Risk Level: medium")
    print(f"Maintenance Advice:\n{medium_risk_advice}")

    # Example 3: Low Risk
    print("\n--- Low Risk Example ---")
    low_risk_params = {'temperature': 30.0, 'vibration': 0.5, 'pressure': 100.0, 'runtime': 500, 'maintenance_history': 1}
    low_risk_advice = get_maintenance_advice(low_risk_params, 'low')
    print(f"Machine Parameters: {low_risk_params}")
    print(f"Risk Level: low")
    print(f"Maintenance Advice:\n{low_risk_advice}")
