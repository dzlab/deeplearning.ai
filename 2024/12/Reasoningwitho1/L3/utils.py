import os
import json
import requests

def o1_tools(messages,model,tools):
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
    data = {
        "model": model,
        "messages": messages,
        "tools": tools
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    return response.json()