from .client import PredictionGuardClient
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, root_validator
from typing import Any, Optional, List, Dict, Iterator, AsyncIterator
from langchain_core.callbacks import CallbackManagerForLLMRun
from utils import get_from_dict_or_env, MultimodalModelInput

from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.prompt_values import StringPromptValue
# from langchain_core.outputs import GenerationChunk, LLMResult 
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks import (
    # CallbackManager,
    CallbackManagerForLLMRun,
)
# from langchain_core.load import dumpd
from langchain_core.runnables.config import run_in_executor

class LVLM(LLM):
    """This class extends LLM class for implementing a custom request to LVLM provider API"""


    client: Any = None #: :meta private:
    hostname: Optional[str] = None
    port: Optional[int] = None
    url: Optional[str] = None
    max_new_tokens: Optional[int] =  200
    temperature: Optional[float] = 0.6
    top_k: Optional[float] = 0
    stop: Optional[List[str]] = None
    ignore_eos: Optional[bool] = False
    do_sample: Optional[bool] = True
    lazy_mode: Optional[bool] = True
    hpu_graphs: Optional[bool] = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the access token and python package exists in environment if needed"""
        if values['client'] is None:
            # check if url of API is provided
            url = get_from_dict_or_env(values, 'url', "VLM_URL", None)
            if url is None:
                hostname = get_from_dict_or_env(values, 'hostname', 'VLM_HOSTNAME', None)
                port = get_from_dict_or_env(values, 'port', 'VLM_PORT', None)
                if hostname is not None and port is not None:
                    values['client'] = PredictionGuardClient(hostname=hostname, port=port)
                else:
                    # using default hostname and port to create Client
                    values['client'] = PredictionGuardClient()
            else:
                values['client'] = PredictionGuardClient(url=url)
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm"""
        return "Large Vision Language Model"
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling the Prediction Guard API."""
        return {
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "ignore_eos": self.ignore_eos,
            "do_sample": self.do_sample,
            "stop" : self.stop,
        }
    
    def get_params(self, **kwargs):
        params = self._default_params
        params.update(kwargs)
        return params
                          

    def _call(
        self,
        prompt: str,
        image: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the VLM on the given input. 

        Args:
            prompt: The prompt to generate from.
            image: This can be either path to image or base64 encode of the image.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
        Returns:
            The model output as a string. Actual completions DOES NOT include the prompt
        Example: TBD
        """
        params = {}
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        params['generate_kwargs'] = self.get_params(**kwargs)
        response = self.client.generate(prompt=prompt, image=image, **params)
        return response
    
    def _stream(
        self,
        prompt: str,
        image: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream the VLM on the given prompt and image. 

        Args:
            prompt: The prompt to generate from.
            image: This can be either path to image or base64 encode of the image.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
        Returns:
            The model outputs an iterator of string. Actual completions DOES NOT include the prompt
        Example: TBD
        """
        params = {}
        params['generate_kwargs'] = self.get_params(**kwargs)
        for chunk in self.client.generate_stream(prompt=prompt, image=image, **params):
            yield chunk

    async def _astream(
        self,
        prompt: str,
        image: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """An async version of _stream method that stream the VLM on the given prompt and image. 

        Args:
            prompt: The prompt to generate from.
            image: This can be either path to image or base64 encode of the image.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
        Returns:
            The model outputs an async iterator of string. Actual completions DOES NOT include the prompt
        Example: TBD
        """
        iterator = await run_in_executor(
            None,
            self._stream,
            prompt,
            image,
            stop,
            run_manager.get_sync() if run_manager else None,
            **kwargs,
        )
        done = object()
        while True:
            item = await run_in_executor(
                None,
                next,
                iterator,
                done,  # type: ignore[call-arg, arg-type]
            )
            if item is done:
                break
            yield item  # type: ignore[misc]
    
    def invoke(
        self,
        input: MultimodalModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        config = ensure_config(config)
        if isinstance(input, dict) and 'prompt' in input.keys() and 'image' in input.keys():
            return (
                self.generate_prompt(
                    [self._convert_input(StringPromptValue(text=input['prompt']))],
                    stop=stop,
                    callbacks=config.get("callbacks"),
                    tags=config.get("tags"),
                    metadata=config.get("metadata"),
                    run_name=config.get("run_name"),
                    run_id=config.pop("run_id", None),
                    image= input['image'],
                    **kwargs,
                )
                .generations[0][0]
                .text
            )
        return (
            self.generate_prompt(
                [self._convert_input(input)],
                stop=stop,
                callbacks=config.get("callbacks"),
                tags=config.get("tags"),
                metadata=config.get("metadata"),
                run_name=config.get("run_name"),
                run_id=config.pop("run_id", None),
                **kwargs,
            )
            .generations[0][0]
            .text
        )

    async def ainvoke(
        self,
        input: MultimodalModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        config = ensure_config(config)
        if isinstance(input, dict) and 'prompt' in input.keys() and 'image' in input.keys():
            llm_result = await self.agenerate_prompt(
            [self._convert_input(StringPromptValue(text=input['prompt']))],
            stop=stop,
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            image=input['image'],
            **kwargs,
            ) 
        else:
            llm_result = await self.agenerate_prompt(
            [self._convert_input(input)],
            stop=stop,
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            **kwargs,
        )
        return llm_result.generations[0][0].text

    def stream(
        self,
        input: MultimodalModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if type(self)._stream == BaseLLM._stream:
            # model doesn't implement streaming, so use default implementation
            yield self.invoke(input, config=config, stop=stop, **kwargs)
        else:
            if stop is not None:
                raise ValueError("stop kwargs are not permitted.")
            image = None
            prompt = None
            if isinstance(input, dict) and 'prompt' in input.keys():
                prompt = self._convert_input(input['prompt']).to_string()
            else:
                raise ValueError("prompt must be provided")
            if isinstance(input, dict) and 'image' in input.keys():
                image = input['image']
            
            for chunk in self._stream(
                prompt=prompt, image=image, **kwargs
            ):
                yield chunk
    
    async def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if (
            type(self)._astream is BaseLLM._astream
            and type(self)._stream is BaseLLM._stream
        ):
            yield await self.ainvoke(input, config=config, stop=stop, **kwargs)
            return
        else:
            if stop is not None:
                raise ValueError("stop kwargs are not permitted.")
            image = None
            if isinstance(input, dict) and 'prompt' in input.keys() and 'image' in input.keys():
                prompt = self._convert_input(input['prompt']).to_string()
                image = input['image']
            else:
                raise ValueError("missing image is not permitted")
                prompt = self._convert_input(input).to_string()
            
            async for chunk in self._astream(
                prompt=prompt, image=image, **kwargs
            ):
                yield chunk