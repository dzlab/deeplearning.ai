##Setup for predibase token access, endpoint url and headers

import os
from dotenv import load_dotenv, find_dotenv


# Initailize global variables
_ = load_dotenv(find_dotenv())

predibase_api_token = os.getenv('PREDIBASE_API_TOKEN')

endpoint_url = f"{os.getenv('PREDIBASE_API_BASE', 'https://serving.app.predibase.com/6dcb0c/deployments/v2/llms')}/mistral-7b"

headers = {
    "Authorization": f"Bearer {predibase_api_token}"
}



