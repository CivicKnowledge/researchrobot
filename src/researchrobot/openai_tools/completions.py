from tenacity import (retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type,
                        retry_if_exception_type)
import os
import openai

from openai import RateLimitError, BadRequestError

#@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20),
#       retry=retry_if_exception_type(RateLimitError))
def openai_one_completion(prompt, system=None,  json_mode = False,
                          return_response = False, return_usage=False, **kwargs):
    """Call the OpenAI completions interface to re-write the extra path for a census variable
    into an English statement that can be used to describe the variable."""

    openai.api_key = os.getenv("OPENAI_API_KEY")

    args = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p":  1,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2,
    }

    args.update(kwargs)

    if json_mode:
        args['response_format']={ "type": "json_object" }

    # The problem can either be split into a system and user prompt, or it can be
    # passed in as a single prompt dict, with multiple roles, since this is a chat interface.
    if isinstance(prompt, str):

        if system:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    client = openai.OpenAI()

    response = client.chat.completions.create(messages=messages, **args)

    if return_response:
        return response
    elif return_usage:
        return response.usage, response.choices[0].message.content.strip()
    else:
        return response.choices[0].message.content.strip()


def openai_run_completions(prompts, system=None, **kwargs):
    """Run multiple completion requests in parallel"""
    pass
