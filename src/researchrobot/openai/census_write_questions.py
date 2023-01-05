"""Process  census metadata with OpenAI
"""

import json
from random import shuffle

import pandas as pd

from researchrobot.openai.census_clean_paths import get_census_metadata
from researchrobot.openai.census_demo_conditions import (
    add_rest_str,
    make_restricted_description,
)

__author__ = "eric@civicknowledge.com"


def write_question(d):
    """Call the OpenAI completions interface to re-write the extra path for a census variable
    into an English statement that can be used to describe the variable."""

    import os

    import openai

    openai.api_key = os.getenv("OPENAI_API_KEY")

    jd = json.dumps(d, indent=4)

    prompt = f"""
For each dict entry in the following JSON,  replace the <question> with  a question  that could be answered with a
dataset that is described by the "description" and return the new JSON. The question should include the text in
the "time" and "place". Questions must start start with a variety of phrases such as 'how many', 'which state has',
'in which year', 'what is', 'comparing', 'make a list of', 'show a map of' and 'where'.

{jd}
          """

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response


def build_question_df(db):
    pdf = pd.DataFrame(db.values()).rename(
        columns={
            "unique_id": "table_id",
            "path": "filtered_pathname",
            "name": "path_name",
        }
    )

    tdf, cdf = get_census_metadata()

    mdf = add_rest_str(tdf, cdf)
    mdf = mdf.merge(pdf, on=["table_id", "filtered_pathname"], how="left").copy()

    mdf["rest_description"] = mdf.apply(make_restricted_description, axis=1)

    mdf["col_desc"] = mdf.stripped_title + " for " + mdf.rest_description

    mdf = mdf[~mdf.table_id.str.endswith("PR")]

    return mdf.copy()


def questions_tasks(mdf):
    """Write data records for generating questions from columns"""

    import uuid

    from researchrobot.openai.census_demo_conditions import random_place, random_time

    vds = []

    for idx, r in mdf.sample(frac=1).iterrows():
        vds.append(
            {
                "column_id": r.uid,
                "uuid": str(uuid.uuid4()),
                "description": r.col_desc,
                "time": random_time(),
                "place": random_place(),
                "question": "<question>",
            }
        )

    shuffle(vds)
    return vds


def store_question_responses(response, db):
    import json

    response_objs = json.loads(response["choices"][0]["text"])

    for i, o in enumerate(response_objs):
        db[o["uuid"]] = o
