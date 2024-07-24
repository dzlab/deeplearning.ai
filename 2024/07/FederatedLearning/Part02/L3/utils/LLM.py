import textwrap
import fireworks.client
from typing import Tuple
from fireworks.client.api import CompletionResponse

############################ Fireworks ########################


#######################################
import os
from dotenv import load_dotenv, find_dotenv
def load_env():
    _ = load_dotenv(find_dotenv())

def get_fireworks_api_key():
    load_env()
    fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
    return fireworks_api_key
#############################################################


class LLM_eval:
    def __init__(self, model: str, **kwargs) -> None:
        self.model = model
        self.temp = 0
        if "temp" in kwargs:
            self.temp = kwargs.pop("temp")
        fireworks.client.api_key = get_fireworks_api_key()
        self.response = None

    def eval(self, prompt, log_probs = False, verbose: bool=False, top_k: bool=True) -> None:
        if verbose:
            to_print = textwrap.fill(prompt, 50, subsequent_indent="\t")
            print(f'Prompt:\n\t{to_print}')
        self.response = fireworks.client.Completion.create(
            model=self.model,
            prompt=prompt,
            echo = log_probs,
            max_tokens=0 if log_probs else 80,
            temperature=self.temp,
            logprobs= 0,
            top_k = 5 if top_k else None
        )

    def get_response(self) -> str:
        return self.response.choices[0].text.strip()

    def get_response_reason(self) -> Tuple[str, str]:
        return (
            self.response.choices[0].text.strip(),
            self.response.choices[0].finish_reason.strip(),
        )

    def print_response(self, char_width: int = 50, verbose: bool=True) -> None:

        response = self.response.choices[0].text.strip()
        to_print = textwrap.fill(response, char_width, subsequent_indent="\t")
        if verbose:
            print(f'Response:\n\t{to_print}')
        else:
            print(to_print)
            


    def get_response_raw(self) -> CompletionResponse:
        return self.response


class LLM_pretrained(LLM_eval):
    def __init__(self, model="accounts/flowerai/models/mistral-7b", **kwargs) -> None:
        super().__init__(model, **kwargs)


class LLM_fl(LLM_eval):
    def __init__(self, model="accounts/flowerai/models/mistral-7b-fl", **kwargs) -> None:
        super().__init__(model, **kwargs)


class LLM_cen_partial(LLM_eval):
    def __init__(self, model="accounts/flowerai/models/mistral-7b-cen-10", **kwargs) -> None:
        super().__init__(model, **kwargs)


class LLM_cen(LLM_eval):
    def __init__(self, model="accounts/flowerai/models/mistral-7b-cen-100", **kwargs) -> None:
        super().__init__(model, **kwargs)

