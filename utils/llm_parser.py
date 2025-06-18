import ast
import json

def clean_and_parse_agent_output(agent_output: str) -> dict:
    # Step 1: Strip markdown-style code block markers
    cleaned = agent_output.strip()
    
    # Remove starting ```json or ``` if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
    
    # Remove ending ``` if present
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    
    # Step 2: Parse the string as a Python dict
    return ast.literal_eval(cleaned)

def clean_and_parse_agent_output_router(agent_output: str) -> dict:
    # Step 1: Strip markdown-style code block markers
    cleaned = agent_output.strip()
    
    # Remove starting ```json or ``` if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
    
    # Remove ending ``` if present
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    new=json.loads(cleaned)
    # Step 2: Parse the string as a Python dict
    return new