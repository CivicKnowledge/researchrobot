""" Convert the components of the census column metadata file into a string that
better describes the column
"""
from random import choice

import pandas as pd
from geoid.censusnames import geo_names
from more_itertools import always_iterable

# Maybe Deprecated! All of this code has been moved
# to the civicknowledge.com-census_meta-2020e5 data package


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
            *(([""] * 2)),  # Even out the amount of nothing
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
            *(([""] * 2)),  # Even out the amount of nothing
        ]
    )


subject_choices = {
    "Age-Sex": [
        "sex and age distribution",
        "age group",
        "gender",
        "people by age",
        "sex and age",
    ],
    "Race": ["count of races", "racial group"],
    "Ancestry": ["ancestry", "nationality", "ethnicity"],
    "Foreign Birth": [
        "country of birth",
        "where people were born",
        "people born in other countries",
        "immigrant",
        "alien",
        "foreign-born",
        "migrant",
    ],
    "Place of Birth - Native": [
        "country of birth",
        "where people were born",
        "people born in other countries",
        "immigrant",
        "alien",
        "foreign-born",
        "migrant",
    ],
    "Residence Last Year - Migration": [
        "place that people moved from",
        "place of origin",
        "where people moved from",
    ],
    "Journey to Work": ["commute", "travel time", "distance to work", "transporation"],
    "Children - Relationship": [
        "relationship to children",
        "household structure",
        "children",
    ],
    "Households - Families": [
        "families in a household",
        "household structure",
        "households",
    ],
    "Marital Status": [
        "marital status",
        "married",
        "divorced",
        "seperated",
        "widowed",
        "single",
        "never married",
        "couple",
        "relationship",
        "partner",
        "spouse",
        "husband",
        "wife",
        "married couple",
        "married spouse",
        "married partner",
        "relationship status",
        "married people",
    ],
    "Fertility": [
        "children",
        "births",
        "pregnancies",
        "children",
        "births",
        "pregnancies",
    ],
    "School Enrollment": [
        "school enrollment",
        "school",
        "education",
        "enrollment",
        "students",
        "children in school",
    ],
    "Educational Attainment": [
        "education level",
        "degree",
        "diploma",
        "certificate",
        "education",
        "educational attainment",
        "people with a degree",
        "people with a high school diploma",
    ],
    "Language": [
        "language spoken",
        "people who speak a language",
        "people who speak English",
    ],
    "Poverty": [
        "poverty level",
        "people in poverty",
        "people who are poor",
    ],
    "Disability": [
        "disability",
        "disabled",
        "people with a disability",
        "people who are disabled",
    ],
    "Income": [
        "income",
        "earnings",
        "wages",
        "salary",
        "people by income",
        "median income",
        "average income",
    ],
    "Earnings": [
        "after tax earnings",
        "people by earnings",
        "median earnings",
        "average earnings",
        "income",
        "earnings",
        "wages",
        "salary",
    ],
    "Veteran Status": [
        "veteran status",
        "military",
        "armed forces",
        "services",
        "veterans",
        "people who served in the military",
    ],
    "Transfer Programs": [
        "SNAP",
        "food stamps",
        "welfare",
        "people on welfare",
        "people on food stamps",
    ],
    "Employment Status": [
        "employed",
        "unemployed",
        "looking for work",
    ],
    "Industry-Occupation-Class of Worker": [
        "jobs",
        "workers",
        "laborers",
        "people who work",
        "employees",
        "job field",
        "field of work",
    ],
    "Housing": ["", ""],
    "Group Quarters": ["living at college", "in prison", "in a nursing home"],
    "Health Insurance": ["", ""],
    "Computer and Internet Usage": [
        "computers",
        "smart phones",
        "internet access",
        "social media",
    ],
}


def inv_subject():
    """Invert the subject choices dictionary"""

    inv_map = {}
    for code, terms in subject_choices.items():
        for term in always_iterable(terms):
            inv_map[term.lower()] = code.lower()

    return inv_map


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


def inv_race_iterations():
    """Invert the race-iteration map, mapping plain english terms to codes"""

    from more_itertools import always_iterable

    race_inv_map = {}
    for _, code, terms in race_iterations:
        for term in always_iterable(terms):
            race_inv_map[term.lower()] = code.lower()

    return race_inv_map


age_phrases = {
    "018-120": ["adults", "age 18 and over", "over 18"],
    "065-120": ["seniors", "aged 65 and older", "over 65"],
    "000-018": ["children", "age 18 and under", "under 18"],
    "000-019": ["children", "age 19 and under", "under 19"],
    "000-025": ["children and young adults", "age 25 and under", "under 25"],
    "025-064": [
        "adults",
        "working age adults",
    ],
    "000-003": [
        "infants",
        "babies",
        "newborns",
        "carpet rats",
        "ankle biters",
        "pre-schoolers",
        "sprog",
    ],
    "000-005": ["infants and todlers", "toddlers", "young children", "kindergarteners"],
    "000-006": ["infants and todlers", "toddlers", "young children"],
    "000-010": ["young children", "children", "kids", "snot-nosed brats"],
    "000-015": ["young children", "children", "kids"],
    "012-014": ["pre-teens", "tweens", "young teens"],
    "012-017": ["teens", "young adults", "young people", "teenagers"],
    "015-019": ["teens", "young adults", "young people", "teenagers"],
}


def inv_age_phrases():
    """Invert the age phrase dictionary"""

    inv_map = {}
    for code, terms in age_phrases.items():
        for term in always_iterable(terms):
            inv_map[term.lower()] = code.lower()

    return inv_map


def age_range_terms(df):
    from itertools import product

    def _age_range_terms(mn, mx):

        if mn == 0:
            return [
                f"{mx} and under",
                f"{mx - 1} and under",
                f"younger than {mx}",
                f"younger than {mx + 1}",
            ]
        elif mx == 120:
            return [
                f"{mn} and older",
                f"{mn + 1} and older",
                f"older than {mn}",
                f"older than {mn + 1}",
            ]
        else:
            v = [f"{mn} to {mx}"]
            for mni, mxi in product((-1, 1), (-1, 1)):
                v.append(f"{mn + mni} to {mx + mxi}")
                v.append(f"{mn + mni} to {mx + mxi} years old")

            return v

    age_terms = {}
    for age in df.age.unique():
        if age != "all":
            mn, mx = [int(e) for e in age.split("-")]
            terms = _age_range_terms(mn, mx)
            for term in terms:
                age_terms[term] = age

    return age_terms


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


def inv_pov_phrases():
    """Invert the poverty phrase dictionary"""

    inv_map = {}
    for code, terms in pov_phrases.items():
        for term in always_iterable(terms):
            inv_map[term.lower()] = code.lower()

    return inv_map


def poverty_str(v):
    if v == "all":
        return None
    else:
        return choice(pov_phrases.get(v))


sex_phrases = {
    "all": [
        None,
        "both sexes",
        "children",
        "boys and girls",
        "adults",
        "men and women",
    ],
    "male": ["males", "men", "boys", "male children", "male adult"],
    "female": ["females", "women", "girls", "female children", "female adult"],
}


def inv_sex_phrases():
    inv_map = {}
    for code, terms in sex_phrases.items():
        for term in always_iterable(terms):
            if term is not None:
                inv_map[term.lower() if term else term] = code.lower()

    return inv_map


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
    except TypeError:
        return r.restriction_str
