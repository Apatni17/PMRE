import os
import json
import requests

def load_knowledge_base(kb_path='PRME_App/data/knowledge_base.json'):
    """
    Loads a knowledge base from a JSON file.
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

knowledge_base = load_knowledge_base()

OLLAMA_URL = 'http://localhost:11434/api/chat'
OLLAMA_MODEL = 'llama2'  # You can change to 'mistral' or another model if you prefer

def get_maintenance_advice(machine_params, risk_category, chat_history=None):
    """
    Uses a local LLM (Ollama) to provide conversational maintenance advice.
    chat_history: list of dicts with 'role' and 'content' (Streamlit session state)
    """
    if chat_history is None:
        chat_history = []

    # Compose system prompt with knowledge base
    system_prompt = (
        "You are a helpful AI assistant specializing in predictive maintenance for industrial machines. "
        "You have access to a knowledge base for risk causes and preventive actions. "
        "When given machine parameters and a risk category, explain possible causes and suggest preventive actions. "
        "Be conversational and answer follow-up questions.\n\n"
    )
    if knowledge_base:
        system_prompt += f"Knowledge base: {json.dumps(knowledge_base, indent=2)}\n\n"
    system_prompt += (
        f"Current machine parameters: {json.dumps(machine_params, indent=2)}\n"
        f"Predicted risk category: {risk_category}\n"
    )

    # Build messages for Ollama
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        # msg['role'] should be 'user' or 'assistant'
        messages.append({"role": msg["role"], "content": msg["content"]})

    # The last user message is always the user's latest input
    # (Handled by Streamlit before calling this function)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        # Ollama returns the latest message in 'message' key
        return data['message']['content'].strip()
    except Exception as e:
        return f"[Local LLM Error] Could not get a response from Ollama. Is it running? Error: {e}"

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
