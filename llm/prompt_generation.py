import pandas as pd
import json

def load_data(csv_path, json_path):
    df = pd.read_csv(csv_path)
    with open(json_path, 'r') as file:
        prompts = json.load(file)
    return df, prompts

def generate_prompts_with_codes(csv_path, json_path, prompt_key):
    df, prompts = load_data(csv_path, json_path)
    
    if prompt_key not in prompts:
        raise ValueError(f"Prompt key '{prompt_key}' not found in JSON file.")
    
    prompt = prompts[prompt_key]
    
    combined_prompts = [f"{prompt}{code}" for code in df['Code']]
    
    return combined_prompts