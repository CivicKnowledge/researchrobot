import openai
import json
from typing import Optional, List
from pydantic import BaseModel, Field
from functools import cached_property
from researchrobot.cmech.util import generate_tools_specification
from researchrobot.openai_tools.completions import openai_one_completion

class SocMatch(BaseModel):
    soc_code: str = Field(..., description="The SOC code of the experience.")
    rationale: str = Field(..., description="Reason for assigning this SOC code")
    quality: float = Field(..., description="The quality of the match, where 1 is perfect and 0 is no match.")


class Experience(BaseModel):
    """Experience hold the title and description for a single job experience,
    and the SOC codes that match it."""

    title: str = Field(..., description="The title of the experience.")
    description: str = Field(..., description="The description of the experience.")

    soc_codes: Optional[List[SocMatch]] = Field(description="The SOC codes of the experience.",
                                                default=None)


class Profile(BaseModel):
    pass


class Functions:

    def __init__(self):
        pass

    def search_job(self, title: str, description: str) -> Experience:
        """Search for a job experience by title and description.

        This function searches for a job experience based on a given title and description.
        It returns an Experience object that matches the search criteria.

        Args:
            title (str): The title of the job experience to search for. This should be a
                         string representing the job title, such as 'Software Engineer' or 'Data Analyst'.
            description (str): The description of the job experience to search for. This should
                               be a detailed string describing the job role, responsibilities, or
                               any specific details relevant to the search.

        Returns:
            Experience: An Experience object that matches the title and description provided.
                        If no matching job experience is found, this may return None or an empty Experience object,
                        depending on the implementation.
        """

        print("!!!!! HERE")

    @classmethod
    def tools(cls):
        return generate_tools_specification(Functions)


class CMechAssistant:

    def __init__(self, tools: List[dict], messages: List[dict] = None):
        self.tools = tools #  Functions to call, and function definitions
        self.func_spec = tools.tools()

        self.messages = messages or []

    def stop(self):
        print("Stopping!!!!")

    def run(self, prompt: str|List[dict], **kwargs):
        """Run a single completion request"""

        client = openai.OpenAI()

        if isinstance(prompt, str):
            self.messages.append([{"role": "user", "content": prompt}])
        else:
            self.messages.extend(prompt)

        while True:

            try:
                r = client.chat.completions.create(messages=self.messages,
                                               tools=self.func_spec,
                                               model='gpt-3.5-turbo-1106')
            except:
                print("!!!!!! ERROR")
                print(json.dumps(json.loads(self.messages), indent=2))
                print(json.dumps(json.loads(r.model_dump_json()), indent=2))
                raise

            print(json.dumps(json.loads(r.model_dump_json()), indent=2))

            self.messages.append(r.choices[0].message)

            match r.choices[0].finish_reason:
                case "stop"|"content_filter":
                    self.stop()
                    return
                case "length":
                    self.stop()
                    return
                case "function_call" |"tool_calls":
                    self.call_function(r)
                case "null":
                    pass # IDK what to do here.

    def call_function(self, response):
        tool_calls = response.choices[0].message.tool_calls

        for tool_call in tool_calls:

            m = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": "I'm going to call the function: " + tool_call.function.name
            }
            print("!!!! Calling function: ", m)
            self.messages.append(m)

import unittest
class MyTestCase(unittest.TestCase):
    def test_something(self):
        import json

        ex = Experience(title="Software Engineer", description="I am a software engineer.")

        messages = [
            {'role': 'system', 'content': 'We are testing the function calling features of the API.'},
            {'role': 'user',
             'content': "I'm going to give you an Experience object, and I want you to call the search_job function with it."},
            {'role': 'user', 'content': json.dumps(ex.model_dump())}
        ]

        cm = CMechAssistant(tools=Functions())

        cm.run(messages)




if __name__ == '__main__':
    unittest.main()
