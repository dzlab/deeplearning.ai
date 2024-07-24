import textwrap
import fireworks.client
from typing import Tuple
from fireworks.client.api import CompletionResponse

############################ Fireworks ########################

class LLM_eval:
    def __init__(self, model="accounts/pa511-5b7169/models/cen", **kwargs) -> None:
        self.model = model
        self.temp = 0
        if "temp" in kwargs:
            self.temp = kwargs.pop("temp")
        fireworks.client.api_key = "7b4NY94GL69GVYTuY7Y398e7aR6Mj4BLOX8zdlDX2KH2Y0Si"
        self.response = None

    def eval(self, prompt, log_probs = False, verbose: bool=False, top_k: bool=True) -> None:
        if verbose:
            to_print = textwrap.fill(prompt, 50, subsequent_indent="\t")
            print(f'Prompt:\n\t{to_print}')
        self.response = fireworks.client.Completion.create(
            model=self.model,
            prompt=prompt,
            echo = log_probs,
            max_tokens=0 if log_probs else 40,
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
    def __init__(self, model="accounts/fireworks/models/mistral-7b", **kwargs) -> None:
        super().__init__(model, **kwargs)


class LLM_fl(LLM_eval):
    def __init__(self, model="accounts/pa511-5b7169/models/fl", **kwargs) -> None:
        super().__init__(model, **kwargs)


class LLM_cen_partial(LLM_eval):
    def __init__(self, model="accounts/pa511-5b7169/models/cen-partial", **kwargs) -> None:
        super().__init__(model, **kwargs)


class LLM_cen(LLM_eval):
    def __init__(self, model="accounts/pa511-5b7169/models/cen", **kwargs) -> None:
        super().__init__(model, **kwargs)

