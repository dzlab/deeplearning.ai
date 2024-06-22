def raven_post(payload):
    """
    Sends a payload to a TGI endpoint.
    """
    import requests
    # Specify the LLM Endpoint
    # Now, let's prompt Raven!
    API_URL = "http://nexusraven.nexusflow.ai"
    headers = {
            "Content-Type": "application/json"
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def query_raven(prompt):
    """
    This function sends a request to the TGI endpoint to get Raven's function call.
    This will not generate Raven's justification and reasoning for the call, to save on latency.
    """
    import requests
    output = raven_post({
        "inputs": prompt,
        "parameters" : {"temperature" : 0.001, "stop" : ["<bot_end>"], "do_sample" : False, "max_new_tokens" : 2048, "return_full_text" : False}})
    call = output[0]["generated_text"].replace("Call:", "").strip()
    return call
