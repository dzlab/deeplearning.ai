import os
from dotenv import load_dotenv, find_dotenv


def load_env():
    _ = load_dotenv(find_dotenv())

def get_ai_hub_api_token():
    load_env()
    ai_hub_api_token = os.getenv("AI_HUB_API_KEY")
    return ai_hub_api_token
