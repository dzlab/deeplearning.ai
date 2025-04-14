# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv

# these expect to find a .env file at the directory above the lesson.                                                                                                                     # the format for that file is (without the comment)                                                                                                                                       #API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService                                                                                                                                     
def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

def print_mention(processed_mention, mention):
    # Check if we need to respond
    if processed_mention.needs_response:
        # We need to respond
        print(f"Responding to {processed_mention.sentiment} {processed_mention.product} feedback")
        print(f"  User: {mention}")
        print(f"  Response: {processed_mention.response}")
    else:
        print(f"Not responding to {processed_mention.sentiment} {processed_mention.product} post")
        print(f"  User: {mention}")

    if processed_mention.support_ticket_description:
        print(f"  Adding support ticket: {processed_mention.support_ticket_description}")
 