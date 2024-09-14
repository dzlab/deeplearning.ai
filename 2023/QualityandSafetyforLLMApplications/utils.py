#!pip install python-dotenv

import os
from dotenv import load_dotenv, find_dotenv

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")


def get_openai_api_base_url():
    return os.getenv("OPENAI_API_BASE")


def get_hf_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("HUGGINGFACE_API_KEY")
