import requests

API_URL = "http://nexusraven.nexusflow.ai"

headers = {
        "Content-Type": "application/json"
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def querychat(payload):
    response = requests.post(CHAT_URL, headers=headers, json=payload)
    return response.json()

def query_raven(prompt):
    return query({
        "inputs" : prompt,
        "parameters" : {"do_sample" : True, "temperature" : 0.001, "max_new_tokens" : 400, "stop" : ["<bot_end>", "Thought:"], "return_full_text" : False}
    })[0]["generated_text"].replace("Call:", "").replace("Thought:", "").strip()