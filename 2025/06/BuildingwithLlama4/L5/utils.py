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

def get_hf_access_token():
    load_env()
    github_access_token = os.getenv("HF_ACCESS_TOKEN")
    return github_access_token

def get_github_access_token():
    load_env()
    github_access_token = os.getenv("GITHUB_ACCESS_TOKEN")
    return github_access_token


from openai import OpenAI


def llama4(user_prompt, system_prompt="You are a helpful assistant", image_urls=[], model="Llama-4-Scout-17B-16E-Instruct-FP8", debug=False): # Llama-4-Maverick-17B-128E-Instruct-FP8
  image_urls_content = []
  for url in image_urls:
    image_urls_content.append({"type": "image_url", "image_url": {"url": url}})

  content = [{"type": "text", "text": user_prompt}]
  content.extend(image_urls_content)

  client = OpenAI(api_key=get_llama_api_key(), base_url=get_llama_base_url())

  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": content}]

  response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0
  )

  if debug:
    print(response)

  return response.choices[0].message.content


from together import Together


def llama4_together(prompt, image_urls=[], model="meta-llama/Llama-4-Scout-17B-16E-Instruct", debug=False):
  image_urls_content = []
  for url in image_urls:
    image_urls_content.append({"type": "image_url", "image_url": {"url": url}})

  content = [{"type": "text", "text": prompt}]
  content.extend(image_urls_content)

  client = Together(api_key = get_together_api_key())
  response = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": content
    }],
    temperature=0
  )

  if debug:
    print(response)

  return response.choices[0].message.content



from pathlib import Path

from pypdf import PdfReader


def pdf2text(file : str):
  text = ''
  with Path(file).open("rb") as f:
    reader = PdfReader(f)
    text = "\n\n".join([page.extract_text() for page in reader.pages])

  return text




import json

import requests

with open("files/pr.json", "r") as f:
    all_pr_content = json.load(f)

def get_pr_content(repo_owner, repo_name, pr_number, token=None):
    if str(pr_number) not in all_pr_content:
        return None
    return all_pr_content[str(pr_number)]


def get_pull_requests(repo_owner, repo_name, token=None):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    pull_requests = []
    params = {}
    params["state"] = "all"
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = json.loads(response.text)

        pull_requests.extend(data)

        while "next" in response.links:
            next_url = response.links["next"]["url"]

            response = requests.get(next_url, headers=headers)

            if response.status_code == 200:
                data = json.loads(response.text)

                pull_requests.extend(data)
            else:
                print(f"Failed to retrieve next page: {response.text}")
                break
    else:
        print(f"Failed to retrieve pull requests: {response.text}")

    return pull_requests


def get_pr_content_live(repo_owner, repo_name, pr_number, token=None):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_number}"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = json.loads(response.text)
        content = data["title"] + (data["body"] if data["body"] else "")
        return content
    else:
        print(f"Failed to retrieve pull request content: {response.text}")
        return None




import json

import requests


def get_issues(repo_owner, repo_name, token=None):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    issues = []
    params = {
        "state": "all",
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = json.loads(response.text)

        issues.extend([issue for issue in data if "pull_request" not in issue])

        while "next" in response.links:
            next_url = response.links["next"]["url"]

            response = requests.get(next_url, headers=headers)

            if response.status_code == 200:
                data = json.loads(response.text)
                issues.extend([issue for issue in data if "pull_request" not in issue])
            else:
                print(f"Failed to retrieve next page: {response.text}")
                break
    else:
        print(f"Failed to retrieve pull requests: {response.text}")

    return issues

import os
import zipfile
from io import BytesIO

import requests


def download_and_extract_repo(repo_url, extract_dir):
    """Download and extract the GitHub repository ZIP file."""
    if repo_url.endswith('/'):
        repo_url = repo_url[:-1]
    zip_url = f"{repo_url}/archive/refs/heads/main.zip"

    response = requests.get(zip_url)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall(extract_dir)

def get_py_files(repo_dir):
    """Get a list of Python files in the repository."""
    py_files = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            file_path = os.path.join(root, file)
            _, file_extension = os.path.splitext(file_path)
            if file_extension == '.py':
                py_files.append(file_path)
    return py_files

def write_files_to_text(py_files, output_file):
    """Write the paths and contents of non-binary files to a text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path in py_files:
            print(f"Writing {file_path}")
            f.write(f"Path: {file_path}\n")
            f.write("Content:\n")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                f.write(file.read())
            f.write("\n\n")