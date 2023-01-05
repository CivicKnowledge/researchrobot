""" Convert the components of the census column metadata file into a string that
better describes the column
"""
from random import choice

import pandas as pd
from geoid.censusnames import geo_names
from more_itertools import always_iterable

states = []
counties = []
for k, v in geo_names.items():
    if k[1] == 0:
        states.append(v)
    else:
        counties.append(v)


def random_place():
    return choice(
        [
            "by " + choice(["counties", "tracts", "cities"]),
            choice(["for", "by"])
            + " "
            + choice(["counties", "tracts", "cities"])
            + " in "
            + choice(counties),
            "in " + choice([choice(states), choice(counties)]),
            choice(
                [
                    "comparing urban to rural areas",
                    "comparing states",
                    "comparing counties",
                ]
            ),
            *(([""] * 3)),  # Even out the amount of nothing
        ]
    )


def random_time():
    return choice(
        [
            "in the past year",
            "in the past " + str(choice([3, 6, 12])) + " months",
            "in the past " + str(choice([5, 7, 10])) + " years",
            "in " + choice(["2016", "2017", "2018", "2019", "2020"]),
            "from "
            + choice(
                [
                    "2010 to 2015",
                    " 2011 to 2017",
                    "2012 to 2018",
                    "2013 to 2019",
                    "2014 to 2020",
                ]
            ),
            *(([""] * 3)),  # Even out the amount of nothing
        ]
    )


subject_choices = {
    "Age-Sex": [
        "sex and age distribution",
        "number of people by age",
        "count of people by sex and age",
    ],
    "Race": ["portion of people by race", "count of races", "largest racial group"],
    "Ancestry": ["ancestry", "nationality", "ethnicity"],
    "Foreign Birth": [
        "country of birth",
        "where people were born",
        "number of people born in other countries",
    ],
    "Place of Birth - Native": [
        "country of birth",
        "where people were born",
        "number of people born in other countries",
    ],
    "Residence Last Year - Migration": [
        "place that people moved from",
        "place of origin",
        "where people moved from",
    ],
    "Journey to Work": ["commute", "travel time", "distance to work"],
    "Children - Relationship": [
        "relationship to children",
        "household structure",
        "number of children",
    ],
    "Households - Families": [
        "number of families in a household",
        "household structure",
        "number of households",
    ],
    "Marital Status": [
        "marital status",
        "relationship status",
        "number of married people",
    ],
    "Fertility": ["number of children", "number of births", "number of pregnancies"],
    "School Enrollment": [
        "school enrollment",
        "number of students",
        "number of children in school",
    ],
    "Educational Attainment": [
        "education level",
        "number of people with a degree",
        "number of people with a high school diploma",
    ],
    "Language": [
        "language spoken",
        "number of people who speak a language",
        "number of people who speak English",
    ],
    "Poverty": [
        "poverty level",
        "number of people in poverty",
        "number of people who are poor",
    ],
    "Disability": [
        "disability",
        "number of people with a disability",
        "number of people who are disabled",
    ],
    "Income": [
        "income",
        "number of people by income",
        "median income",
        "average income",
    ],
    "Earnings": [
        "after tax earnings",
        "number of people by earnings",
        "median earnings",
        "average earnings",
    ],
    "Veteran Status": [
        "veteran status",
        "number of veterans",
        "number of people who served in the military",
    ],
    "Transfer Programs": [
        "SNAP",
        "food stamps",
        "welfare",
        "number of people on welfare",
        "number of people on food stamps",
    ],
    "Employment Status": [
        "number of people who are employed",
        "number of people who are unemployed",
        "number of people who are looking for work",
    ],
    "Industry-Occupation-Class of Worker": ["", ""],
    "Housing": ["", ""],
    "Group Quarters": ["", ""],
    "Health Insurance": ["", ""],
    "Computer and Internet Usage": ["", ""],
}

race_iterations = [
    ("A", "white", ["White", "caucasian"]),
    ("B", "black", ["Black", "African American"]),
    ("C", "aian", ["American Indian", "Alaska Native"]),
    ("D", "asian", "Asian"),
    ("E", "nhopi", ["Native Hawaiian", "Pacific Islander"]),
    ("F", "other", "Some Other Race"),
    ("G", "many", "multiple race"),
    ("H", "nhwhite", "non-hispanic White"),
    ("N", "nhisp", "Not Hispanic"),
    ("I", "hisp", ["Hispanic", "Latino", "chicano", "mexican"]),
    ("C", "aian", ["American Indian", "Indian", "Native American"]),
    (None, "all", "All races"),
]


def raceeth_str(v):
    ri_map = {k: v for _, k, v in race_iterations}

    if v == "all":
        return None
    else:

        return choice(list(always_iterable(ri_map.get(v))))


age_phrases = {
    "018-120": ["adults"],
    "065-120": ["seniors"],
    "000-018": ["children"],
    "000-019": ["children"],
    "000-025": ["children and young adults"],
    "025-064": ["adults", "working age adults"],
    "000-003": ["infants"],
    "000-005": ["infants and todlers"],
    "000-006": ["infants and todlers"],
    "000-010": ["young children"],
    "000-015": ["young children"],
    "012-014": ["pre-teens"],
    "012-017": ["teens"],
    "015-019": ["teens"],
}


def age_str(v):
    if v == "all":
        return None
    else:

        min_age, max_age = map(int, v.split("-"))

        phrases = age_phrases.get(v, [])
        phrases.append(f"ages {min_age} to {max_age}")
        v = choice(phrases)

        repl = ["and up", "and over", "and older"]

        return v.replace("to 120", choice(repl))


pov_phrases = {
    "all": None,
    "lt100": ["in poverty", "below poverty line", "poor"],
    "100-200": [
        "near poverty",
        "poor but not in poverty",
        "from 100% to 200% of poverty level",
    ],
    "100-150": [
        "near poverty",
        "poor but not in poverty",
        "from 100% to 150% of poverty level",
    ],
    "gt150": ["near poverty", "poor but not in poverty", "above 150% of poverty line"],
    "gt100": ["not in poverty", "above poverty line"],
    "100-137": ["not in poverty", "from 100% to 137% of poverty line"],
    "200-300": [
        "not in poverty",
        "twice poverty line",
        "middle class",
        "from 200% to 300% of poverty line",
    ],
}


def poverty_str(v):
    if v == "all":
        return None
    else:
        return choice(pov_phrases.get(v))


def sex_str(v, age):
    if age == "all":
        age = "000-120"

    min_age, max_age = map(int, age.split("-"))

    if v == "all":
        if max_age <= 18:
            return choice([None, "both sexes", "children", "boys and girls"])
        else:
            return choice([None, "both sexes", "adults", "men and women"])
    else:

        if max_age <= 18:
            sex_str = {
                "male": ["males", "boys", "male children"],
                "female": ["females", "girls", "female children"],
            }
        else:
            sex_str = {"male": ["males", "men"], "female": ["females", "women"]}

        return choice(sex_str.get(v))


def restriction_str(r: pd.Series):
    rest_parts = [
        sex_str(r.sex, r.age),
        raceeth_str(r.raceeth) or raceeth_str(r.race),
        age_str(r.age),
        poverty_str(r.poverty_status),
    ]

    v = filter(None, rest_parts)
    restr = ", ".join(v)

    if not restr:
        restr = choice(["all population", "total population"])

    return restr


def add_rest_str(tdf, cdf):
    """Merge the table and column metadata and add the restriction string
    :param tdf: table metadata
    :type tdf:  pandas.Dataframe
    :param cdf: column metadata
    :type cdf:  pandas.Dataframe
    :return:
    :rtype:
    """

    t = tdf.rename(
        columns={"sex": "table_sex", "age": "table_age", "age_range": "table_age_range"}
    ).merge(cdf, on="table_id")

    t["restriction_str"] = t.apply(restriction_str, axis=1)

    return t


def make_restricted_description(r):
    try:
        d = r.description
        d = d.replace("Total", "")
        d = d.replace("groups tallied ", "")
        d = d.strip()
        d = d.replace("living", "who are living")
        d = d.replace("People who", "who")
        d = d.replace("people who", "who")

        return r.restriction_str + " " + d
    except AttributeError:
        return r.restriction_str
