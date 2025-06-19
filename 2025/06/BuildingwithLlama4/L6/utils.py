import os

from dotenv import find_dotenv, load_dotenv


def load_env():
    _ = load_dotenv(find_dotenv())

def get_llama_api_key():
    load_env()
    llama_api_key = os.getenv("LLAMA_API_KEY")
    return llama_api_key

def get_llama_base_url():
    load_env()
    llama_base_url = os.getenv("LLAMA_BASE_URL")
    return llama_base_url

def get_together_api_key():
    load_env()
    together_api_key = os.getenv("TOGETHER_API_KEY")
    return together_api_key

import json
import re
from typing import Any, Dict


def parse_json(input_string: str):
    """
    Attempts to parse the given string as JSON. If direct parsing fails,
    it tries to extract a JSON snippet from code blocks formatted as:
        ```json
        ... JSON content ...
        ```
    or any code block delimited by triple backticks and then parses that content.
    Parameters:
        input_string (str): The input string which may contain JSON.
    Returns:
        The parsed JSON object.
    Raises:
        ValueError: If parsing fails even after attempting to extract a JSON snippet.
    """
    # Try to parse the string directly.
    try:
        return json.loads(input_string)
    except json.JSONDecodeError as err:
        error = err  # Proceed to try extracting a JSON snippet.
    # Define patterns to search for a JSON code block.
    patterns = [
        re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE),  # code block with "json" label
        re.compile(r"```(.*?)```", re.DOTALL)  # any code block delimited by triple backticks
    ]

    # Attempt extraction using each pattern in order.
    for pattern in patterns:
        match = pattern.search(input_string)
        if match:
            json_candidate = match.group(1).strip()
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                # Continue trying if extraction from the code block didn't result in valid JSON.
                continue
    # If all attempts fail, raise an error.
    raise error

def evaluate(ground_truth: Any, predictions: Any, strict_json: bool = True) -> Dict[str, Any]:
    result = {
        "is_valid_json": False,
        "correct_categories": 0.,
        "correct_sentiment": False,
        "correct_urgency": False,
    }
    try:
        ground_truth = ground_truth if isinstance(ground_truth, dict) else (json.loads(ground_truth) if strict_json else parse_json(ground_truth))
        predictions = predictions if isinstance(predictions, dict) else (json.loads(predictions) if strict_json else parse_json(predictions))
    except (json.JSONDecodeError, ValueError):
        pass
    else:
        result["is_valid_json"] = True

        # Handle missing categories in predictions
        correct_categories = 0
        total_categories = len(ground_truth.get("categories", {}))

        if total_categories > 0 and "categories" in predictions:
            for k in ground_truth["categories"].keys():
                # Check if the category exists in predictions before comparing
                if k in predictions["categories"]:
                    if ground_truth["categories"][k] == predictions["categories"][k]:
                        correct_categories += 1
                # Missing category counts as incorrect

            result["correct_categories"] = correct_categories / total_categories

        # Handle missing sentiment and urgency fields
        result["correct_sentiment"] = predictions.get("sentiment") == ground_truth.get("sentiment", None)
        result["correct_urgency"] = predictions.get("urgency") == ground_truth.get("urgency", None)

    # Calculate total score
    correct_fields = [v for k, v in result.items() if k.startswith('correct_')]
    result["total"] = sum(correct_fields) / len(correct_fields) if correct_fields else 0.0

    return result