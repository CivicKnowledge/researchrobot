""" Generate Questions
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from random import choice, choices, randint
from typing import List, Union

import yaml
from geoid.censusnames import geo_names


# class syntax
class Subject(Enum):
    PEOPLE = "people"
    HOMES = "homes"
    HOUSEHOLDS = "households"
    FAMILIES = "families"
    COUNTRIES = "countries"
    COMPANIES = "companies"
    # CITIES = 'cities'
    # STATES = 'states'
    # COUNTIES = 'counties'
    # SCHOOL_DISTRICTS = 'sdistricts'
    # SCHOOLS = 'schools'

    def from_value(value: str):
        for s in Subject:
            if s.value == value:
                return s
        else:
            raise ValueError(f"Subject {value} not found")


@dataclass
class Measure:
    """A property that is being measured"""

    property: str
    predicates: List[str]

    def as_dict(self):
        return {"property": self.property, "predicates": self.predicates}


def oxford_comma(items: List[str]) -> str:
    """Join a list of strings with commas and an Oxford comma"""

    if len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return f"{', '.join(items[:-1])}, and {items[-1]}"


@dataclass()
class Question:
    subject: str
    measures: List[Measure]
    dimensions: List[str]

    question: str = None  # Re-written question

    def _repr_html_(self):
        from IPython.display import HTML

        o = f"""<p>For {self.subject.value}, measure: </p><ul>"""

        for m in self.measures:
            o += f"<li>{m.property}  where {oxford_comma(m.predicates)}</li>"

        o += f"</ul><p> And group by {oxford_comma(self.dimensions)} </p>"

        return o

    def rewrite(self):
        """Have GPT rewrite the question to be more grammatical"""
        from researchrobot.openai.completions import openai_one_completion

        prompt_tmpl = """Rewrite the following question to be more grammatical and easier to read.

        {question} """

        self.question = openai_one_completion(prompt_tmpl.format(question=str(self)))

        return self

    def __str__(self):

        sort_phrase = choice(
            [
                "",
                " sorted",
                " grouped",
                " arranged",
                " ordered",
                " ranked",
                " listed",
                " tabulated",
                " categorized",
                " classified",
            ]
        )

        if len(self.measures) == 1:
            m = self.measures[0]
            return (
                f"For {self.subject.value}, what is the  {m.property}  for the conditions of {oxford_comma(m.predicates)}"
                + f",{sort_phrase} by {oxford_comma(self.dimensions)} ?"
            )
        else:
            m1 = self.measures[0]
            m2 = self.measures[1]
            return (
                f"For {self.subject.value}, what are the  {m1.property}  for the conditions of {oxford_comma(m1.predicates)}"
                + f" compared to {m2.property}  for the conditions of {oxford_comma(m2.predicates)}"
                + f",{sort_phrase} by {oxford_comma(self.dimensions)} ?"
            )

    def dump(self):
        """Dump to a dict"""

        return {
            "topic": self.subject.value,
            "measures": [m.as_dict() for m in self.measures],
            "dimensions": self.dimensions,
        }

    @property
    def json(self):
        import json

        return json.dumps(self.dump(), indent=2)

    @property
    def as_list(self):
        from itertools import chain

        q = self.dump()
        d = [q["topic"]]
        d.extend(m["property"] for m in q["measures"])
        d.extend(chain(*[list(m["predicates"]) for m in q["measures"]]))
        d.extend(q["dimensions"])
        return d


class QuestionGenerator:
    def __init__(self, patterns_path: str = None):
        import researchrobot.openai as rro

        if patterns_path is None:
            patterns_path = Path(rro.__file__).parent / "patterns.yaml"

        self.pp = Path(patterns_path)
        self.load_patterns()

        self.context = {}

        for k in (
            "age",
            "sex",
            "race",
            "maritalstatus",
            "place",
            "time",
            "opinion",
            "thing",
            "city",
            "date",
        ):
            self.context[k] = choice(self.patterns[k])

    def load_patterns(self):

        counties = []
        states = []

        with self.pp.open() as f:
            patterns = yaml.safe_load(f)

        for (staten, countyn), name in geo_names.items():

            if countyn == 0:
                states.append(name)
            else:
                counties.append(name)

        patterns["states"] = states
        patterns["counties"] = counties

        self.patterns = patterns
        return patterns

    def substall(self, v, **kwargs):

        for i in range(10):
            if not "{" in str(v):
                return v
            v = v.format(**self.context, **kwargs)

        return v

    def get_sub_val(self, sub: Union[Subject, str], category: str):

        if not isinstance(sub, Subject):
            sub = Subject.from_value(sub)

        try:
            return self.patterns["subjects"][sub.value][category]
        except KeyError:
            raise KeyError(f"Subject {sub.value} has no {category}")

    @property
    def rand_subject(self):
        """Return a random subject, but one with valid measures in the patterns file"""

        for _ in range(10):
            try:
                sub = choice(list(Subject))
                measures = self.get_sub_val(
                    sub, "measures"
                )  # Just to check that the sub is valid
                break
            except ValueError as e:
                print(e)
                continue
        else:
            raise ValueError("No valid subjects found")

        return sub

    def rand_measure(self, sub: Subject, extant_m: Measure = None):

        if extant_m is None:
            extant_m = []

        measures = list(
            filter(lambda e: e not in extant_m, self.get_sub_val(sub, "measures"))
        )

        measure = self.substall(choice(measures), subject=sub.value)

        n_predicates = choices([1, 2, 3], k=1, weights=[1, 0.5, 0.25])[0]
        all_pred = self.get_sub_val(sub, "predicates")
        predicates = [
            self.substall(e, subject=sub.value)
            for e in choices(all_pred, k=n_predicates)
        ]

        return Measure(measure, predicates)

    def rand_dims(self, sub: Subject, n_dims=None):

        dims = []

        if n_dims is None:
            n_dims = choices([1, 2, 3], k=1, weights=[1, 1, 0.3])[0]

        for i in range(n_dims):
            all_dims = list(
                filter(lambda e: e not in dims, self.get_sub_val(sub, "dimensions"))
            )
            d = choice(all_dims)
            dims.append(d)

        return dims

    def generate(self):
        sub = self.rand_subject

        n_measures = choices([1, 2], k=1, weights=[1, 0.5])[0]

        self.context["minage"] = randint(18, 65)
        self.context["maxage"] = randint(self.context["minage"], 100)

        measures = []
        for i in range(n_measures):
            measures.append(self.rand_measure(sub, measures))

        dims = self.rand_dims(sub)

        return Question(sub, measures, dims)
