import os
from dotenv import load_dotenv
import json
import base64
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import functools
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import math
from vertexai.language_models import TextEmbeddingModel
import numpy as np
import scann;

def authenticate():
    return "DLAI-credentials", "DLAI-PROJECT"
    #Load .env
    load_dotenv()
    
    #Decode key and store in .JSON
    SERVICE_ACCOUNT_KEY_STRING_B64 = os.getenv('SERVICE_ACCOUNT_KEY')
    SERVICE_ACCOUNT_KEY_BYTES_B64 = SERVICE_ACCOUNT_KEY_STRING_B64.encode("ascii")
    SERVICE_ACCOUNT_KEY_STRING_BYTES = base64.b64decode(SERVICE_ACCOUNT_KEY_BYTES_B64)
    SERVICE_ACCOUNT_KEY_STRING = SERVICE_ACCOUNT_KEY_STRING_BYTES.decode("ascii")

    SERVICE_ACCOUNT_KEY = json.loads(SERVICE_ACCOUNT_KEY_STRING)


    # Create credentials based on key from service account
    # Make sure your account has the roles listed in the Google Cloud Setup section
    credentials = Credentials.from_service_account_info(
        SERVICE_ACCOUNT_KEY,
        scopes=['https://www.googleapis.com/auth/cloud-platform'])

    if credentials.expired:
        credentials.refresh(Request())
    
    #Set project ID accoridng to environment variable    
    PROJECT_ID = os.getenv('PROJECT_ID')
        
    return credentials, PROJECT_ID
    
    
def generate_batches(sentences, batch_size = 5):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]

def encode_texts_to_embeddings(sentences):
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(sentences))]
        
def encode_text_to_embedding_batched(sentences, api_calls_per_second = 0.33, batch_size = 5):
    # Generates batches and calls embedding API
    
    embeddings_list = []

    # Prepare the batches using a generator
    batches = generate_batches(sentences, batch_size)

    seconds_per_job = 1 / api_calls_per_second

    with ThreadPoolExecutor() as executor:
        futures = []
        for batch in tqdm(
            batches, total = math.ceil(len(sentences) / batch_size), position=0
        ):
            futures.append(
                executor.submit(functools.partial(encode_texts_to_embeddings), batch)
            )
            time.sleep(seconds_per_job)

        for future in futures:
            embeddings_list.extend(future.result())

    is_successful = [
        embedding is not None for sentence, embedding in zip(sentences, embeddings_list)
    ]
    embeddings_list_successful = np.squeeze(
        np.stack([embedding for embedding in embeddings_list if embedding is not None])
    )
    return embeddings_list_successful

# configure ScaNN as a tree - asymmetric hash hybrid with reordering
# anisotropic quantization as described in the paper; see README
def create_index(embedded_dataset, 
                 num_leaves,
                 num_leaves_to_search,
                 training_sample_size):
    
    # normalize data to use cosine sim as explained in the paper
    normalized_dataset = embedded_dataset / np.linalg.norm(embedded_dataset, axis=1)[:, np.newaxis]
    
    searcher = (
        scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product")
        .tree(
            num_leaves = num_leaves,
            num_leaves_to_search = num_leaves_to_search,
            training_sample_size = training_sample_size,
        )
        .score_ah(2, anisotropic_quantization_threshold = 0.2)
        .reorder(100)
        .build()
    )
    return searcher
