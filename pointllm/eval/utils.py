import os
import time
import random

from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
    InternalServerError,
)


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 40,
    max_delay: int = 30,
    errors: tuple = (
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
    ),
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    print(f"[OPENAI] Encounter error: {e}.")
                    raise Exception(
                        f"[OPENAI] Maximum number of retries ({max_retries}) exceeded."
                    )
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(min(delay, max_delay))
            except Exception as e:
                raise e
    return wrapper


class OpenAIGPT():
    def __init__(self, model="gpt-4o-2024-08-06", temperature=1, top_p=1, max_tokens=2048, **kwargs) -> None:
        setup_openai(model)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.default_chat_parameters = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            **kwargs,
        }

    @retry_with_exponential_backoff
    def safe_chat_complete(self, messages, content_only=True, **kwargs):
        chat_parameters = self.default_chat_parameters.copy()
        if len(kwargs) > 0:
            chat_parameters.update(**kwargs)

        response = self.client.chat.completions.create(
            messages=messages,
            **chat_parameters,
        )

        if content_only:
            return response.choices[0].message.content

        # Convert Pydantic model → plain dict so downstream code that does
        # response['choices'][0]['message']['content'] / response['usage'][...]
        # keeps working unchanged.
        return response.model_dump()


def setup_openai(model_name):
    print("[OPENAI] Setting OpenAI api_key...")
    if not os.getenv("OPENAI_API_KEY"):
        print("[OPENAI] WARNING: OPENAI_API_KEY env var is not set.")
    print(f"[OPENAI] Using MODEL: {model_name}")
