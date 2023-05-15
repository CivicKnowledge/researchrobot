from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20))
def openai_one_completion(prompt, system=None, **kwargs):
    """Call the OpenAI completions interface to re-write the extra path for a census variable
    into an English statement that can be used to describe the variable."""

    import os

    import openai

    openai.api_key = os.getenv("OPENAI_API_KEY")

    args = {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 1,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2,
    }

    args.update(kwargs)

    chat_models = ("3.5", "4", "4.0")

    if any([e in args["model"] for e in chat_models]):

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

        response = openai.ChatCompletion.create(messages=messages, **args)

        return response["choices"][0]["message"]["content"].strip()

    else:

        response = openai.Completion.create(**args, prompt=prompt)

        return response["choices"][0]["text"].strip()


def openai_run_completions(prompts, system=None, **kwargs):
    """Run multiple completion requests in parallel"""
    pass


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus

Milvus.from_documents
