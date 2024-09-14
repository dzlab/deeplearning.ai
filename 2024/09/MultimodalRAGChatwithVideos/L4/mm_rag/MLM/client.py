"""Base interface for client making requests/call to visual language model provider API"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union, Iterator
import requests
import json
from utils import isBase64, encode_image, encode_image_from_path_or_url, lvlm_inference

class BaseClient(ABC):
    def __init__(self,
                 hostname: str = "127.0.0.1",
                 port: int = 8090,
                 timeout: int = 60,
                 url: Optional[str] = None):
        self.connection_url = f"http://{hostname}:{port}" if url is None else url
        self.timeout = timeout
        # self.headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        self.headers = {'Content-Type': 'application/json'}

    def root(self):
        """Request for showing welcome message"""
        connection_route = f"{self.connection_url}/"
        return requests.get(connection_route)

    @abstractmethod
    def generate(self, 
                 prompt: str,
                 image: str,
                 **kwargs
        ) -> str:
        """Send request to visual language model API 
        and return generated text that was returned by the visual language model API

        Use this method when you want to call visual language model API to generate text without streaming

        Args:
            prompt: A prompt.
            image: A string that can be either path to image or base64 of an image.
            **kwargs: Arbitrary additional keyword arguments. 
                These are usually passed to the model provider API call as hyperparameter for generation.

        Returns:
            Text returned from visual language model provider API call
        """


    def generate_stream(
            self, 
            prompt: str, 
            image: str, 
            **kwargs
    ) -> Iterator[str]:
        """Send request to visual language model API 
        and return an iterator of streaming text that were returned from the visual language model API call

        Use this method when you want to call visual language model API to stream generated text.

        Args:
            prompt: A prompt.
            image: A string that can be either path to image or base64 of an image.
            **kwargs: Arbitrary additional keyword arguments. 
                These are usually passed to the model provider API call as hyperparameter for generation.

        Returns:
            Iterator of text streamed from visual language model provider API call
        """
        raise NotImplementedError()
    
    def generate_batch(
            self, 
            prompt: List[str], 
            image: List[str], 
            **kwargs
    ) -> List[str]:
        """Send a request to visual language model API for multi-batch generation 
        and return a list of generated text that was returned by the visual language model API

        Use this method when you want to call visual language model API to multi-batch generate text.
        Multi-batch generation does not support streaming.

        Args:
            prompt: List of prompts.
            image: List of strings; each of which can be either path to image or base64 of an image.
            **kwargs: Arbitrary additional keyword arguments. 
                These are usually passed to the model provider API call as hyperparameter for generation.

        Returns:
            List of texts returned from visual language model provider API call
        """
        raise NotImplementedError()
    
class PredictionGuardClient(BaseClient):

    generate_kwargs = ['max_tokens', 
                       'temperature',
                       'top_p', 
                       'top_k']

    def filter_accepted_genkwargs(self, kwargs):
        gen_args = {}
        if "generate_kwargs" in kwargs and isinstance(kwargs["generate_kwargs"], dict):
            gen_args = {k:kwargs["generate_kwargs"][k] 
                        for k in self.generate_kwargs
                        if k in kwargs["generate_kwargs"]}
        return gen_args
            
    def generate(self, 
                 prompt: str,
                 image: str,
                 **kwargs
        ) -> str:
        """Send request to PredictionGuard's API 
        and return generated text that was returned by LLAVA model

        Use this method when you want to call LLAVA model API to generate text without streaming

        Args:
            prompt: A prompt.
            image: A string that can be either path/URL to image or base64 of an image.
            **kwargs: Arbitrary additional keyword arguments. 
                These are usually passed to the model provider API call as hyperparameter for generation.

        Returns:
            Text returned from visual language model provider API call
        """

        assert image is not None and len(image) != "", "the input image cannot be None, it must be either base64-encoded image or path/URL to image"
        if isBase64(image):
            base64_image = image
        else: # this is path to image or URL to image
            base64_image = encode_image_from_path_or_url(image)

        args = self.filter_accepted_genkwargs(kwargs)
        return lvlm_inference(prompt=prompt, image=base64_image, **args)
    