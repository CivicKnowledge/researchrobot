

from .completions import *

# Models list from the OpenAI OpenAPI spec at
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml

def get_model_list():
    """Retrieve the current list of models from the OpenAI OpenAPI spec"""
    import requests
    import yaml
    r = requests.get('https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml')
    spec = yaml.safe_load(r.content.decode('utf-8'))
    models =spec['components']['schemas']['CreateChatCompletionRequest']['properties']['model']['anyOf'][1]['enum']
    return models


models = [
      "gpt-4-1106-preview",
      "gpt-4-vision-preview",
      "gpt-4",
      "gpt-4-0314",
      "gpt-4-0613",
      "gpt-4-32k",
      "gpt-4-32k-0314",
      "gpt-4-32k-0613",
      "gpt-3.5-turbo-1106",
      "gpt-3.5-turbo",
      "gpt-3.5-turbo-16k",
      "gpt-3.5-turbo-0301",
      "gpt-3.5-turbo-0613",
      "gpt-3.5-turbo-16k-0613",
    ]

# Costs for models,
# ( prompt cost, completion cost )
costs = {
    'gpt-4-1106-preview': (0.0100, 0.0300),
    'gpt-4': (0.0300, 0.0600),
    'gpt-4-32k': (0.0600, 0.1200),
    'gpt-3.5-turbo-16k-0613': (0.0010, 0.0020),
    'gpt-3.5-turbo-1106': (0.0010, 0.0020),
}