import os
from datetime import datetime
from itertools import count
from pathlib import Path
from time import sleep, time

import openai
from more_itertools import always_iterable
from openai.error import InvalidRequestError, RateLimitError

from .decompose_questions import comp_keys, key_order


class FineTuner:
    def __init__(self, tag: str, train_file: Path):
        self.tag = tag
        self.train_file = train_file
        self.train_file_id = None
        self.last_model = None
        self.prompt_prefix = ""

        openai.api_key = os.getenv("OPENAI_API_KEY")

    def list_models(self):
        """List the models that have been fine-tuned for this tag"""
        r = openai.FineTune.list()

        return [
            ft_model["fine_tuned_model"]
            for ft_model in r["data"]
            if self.tag in ft_model["fine_tuned_model"]
        ]

    def delete_models(self, model_names):
        model_names = list(always_iterable(model_names))

        for model_name in model_names:
            openai.FineTune.delete(model_name)

    def upload_training_file(self):

        """Upload a training file to the OpenAI API"""

        self.extract_prefix()

        fc_resp = openai.File.create(file=self.train_file.open(), purpose="fine-tune")

        self.train_file_id = fc_resp["id"]

        return fc_resp["status"]

    def create_fine_tune(self):
        ft_resp = openai.FineTune.create(
            training_file=self.train_file_id, model="davinci", suffix=self.tag
        )

        self.fine_tune_id = ft_resp["id"]

        return ft_resp["status"]

    def set_last_model(self):
        self.last_model = self.list_models()[-1]

    def extract_prefix(self):
        """Extract the prompt prefix"""

        import json

        for l in self.train_file.open().readlines():
            r = json.loads(l)
            prefix, *_ = r["prompt"].split(" ")

        if "|" in prefix:
            self.prompt_prefix = prefix
        else:
            self.prompt_prefix = ""

        return self.prompt_prefix

    def check_status(self):
        """Check the status of the fine-tuning job. Report on the status every 15 seconds until"""
        start_time = None
        from IPython.display import clear_output

        for i in count():
            le_resp = openai.FineTune.list_events(id=self.fine_tune_id)
            clear_output(wait=True)
            for d in le_resp["data"]:
                if start_time is None:
                    start_time = d["created_at"]

                clock_time = datetime.fromtimestamp(d["created_at"]).isoformat()
                dt = round((d["created_at"] - start_time) / 60)
                print(i, clock_time, dt, d["level"], d["message"])

            if d["message"] == "Fine-tune succeeded":
                break

            sleep(15)

        self.set_last_model()

    def get_completion(self, question, model_id=None, return_all=False):

        if model_id is None:
            model_id = self.last_model

        prompt = self.prompt_prefix + " " + question + " ->"

        try:
            r = openai.Completion.create(
                model=model_id,
                prompt=prompt,
                max_tokens=1000,
                stop=" ###",
                temperature=0,
            )
        except Exception as e:
            print(f'ERROR for prompt\n"{prompt}"\n\n', e)
            raise
            return None

        if return_all:
            return r
        else:
            return r["choices"][0]["text"]
