import os
from dotenv import load_dotenv
import os
from dotenv import load_dotenv, find_dotenv
import warnings
import requests
import json
import time
# Initailize global variables
_ = load_dotenv(find_dotenv())
# warnings.filterwarnings('ignore')
url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/inference"
headers = {
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
        "Content-Type": "application/json"
    }


def llama(prompt, 
          add_inst=True, 
          model="togethercomputer/llama-2-7b-chat", 
          temperature=0.0, 
          max_tokens=1024,
          verbose=False,
          url=url,
          headers=headers,
          base=2, # number of seconds to wait
          max_tries=3):
    
    if add_inst:
        prompt = f"[INST]{prompt}[/INST]"

    if verbose:
        print(f"Prompt:\n{prompt}\n")
        print(f"model: {model}")

    data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

    # Allow multiple attempts to call the API incase of downtime.
    # Return provided response to user after 3 failed attempts.    
    wait_seconds = [base**i for i in range(max_tries)]

    for num_tries in range(max_tries):
        try:
            response = requests.post(url, headers=headers, json=data)
            return response.json()['output']['choices'][0]['text']
        except Exception as e:
            if response.status_code != 500:
                return response.json()

            print(f"error message: {e}")
            print(f"response object: {response}")
            print(f"num_tries {num_tries}")
            print(f"Waiting {wait_seconds[num_tries]} seconds before automatically trying again.")
            time.sleep(wait_seconds[num_tries])
 
    print(f"Tried {max_tries} times to make API call to get a valid response object")
    print("Returning provided response")
    return response


def llama_chat(prompts, 
               responses,
               model="togethercomputer/llama-2-7b-chat", 
               temperature=0.0, 
               max_tokens=1024,
               verbose=False,
               url=url,
               headers=headers,
               base=2,
               max_tries=3
              ):

    prompt = get_prompt_chat(prompts,responses)

    # Allow multiple attempts to call the API incase of downtime.
    # Return provided response to user after 3 failed attempts.    
    wait_seconds = [base**i for i in range(max_tries)]

    for num_tries in range(max_tries):
        try:
            response = llama(prompt=prompt,
                             add_inst=False,
                             model=model, 
                             temperature=temperature, 
                             max_tokens=max_tokens,
                             verbose=verbose,
                             url=url,
                             headers=headers
                            )
            return response
        except Exception as e:
            if response.status_code != 500:
                return response.json()

            print(f"error message: {e}")
            print(f"response object: {response}")
            print(f"num_tries {num_tries}")
            print(f"Waiting {wait_seconds[num_tries]} seconds before automatically trying again.")
            time.sleep(wait_seconds[num_tries])
 
    print(f"Tried {max_tries} times to make API call to get a valid response object")
    print("Returning provided response")
    return response


def get_prompt_chat(prompts, responses):
  prompt_chat = f"<s>[INST] {prompts[0]} [/INST]"
  for n, response in enumerate(responses):
    prompt = prompts[n + 1]
    prompt_chat += f"\n{response}\n </s><s>[INST] \n{ prompt }\n [/INST]"

  return prompt_chat
