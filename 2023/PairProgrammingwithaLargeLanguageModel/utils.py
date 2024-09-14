
#!pip install python-dotenv
import os
from dotenv import load_dotenv, find_dotenv

def get_api_key():
    
    _ = load_dotenv(find_dotenv()) # read local .env file
    return os.getenv('GOOGLE_API_KEY')
