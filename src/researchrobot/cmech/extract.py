import json
from dataclasses import dataclass, field, fields
from operator import itemgetter as ig
from typing import Dict, List

# Name of countries in the 'location_country' field that are considered Anglo
anglo_countries = [
    "australia",
    "american samoa",
    "canada",
    "united kingdom",
    "united states",
    "new zealand",
]


def iter_source_cache(rc, source_key, pb=None):
    """Iterate over profiles in a source file stored in the cache"""

    s = rc[source_key]

    for source_idx, l in enumerate(s.splitlines()):
        o = json.loads(l)
        o["_source_key"] = source_key
        o["_source_line"] = source_idx

        yield o


def iter_source_path(path, pb=None):
    """Iterate over profiles in a source file stored in the cache"""

    import gzip

    with gzip.open(path, "rt") as f:
        for source_idx, l in enumerate(f):
            o = json.loads(l)
            o["_source_key"] = path.stem
            o["_source_line"] = source_idx

            yield o


@dataclass
class ProfileKeys:
    """Keep track of all keys in each section, because nulls will get deleted"""

    profile: set[str] = field(default_factory=set)
    exp: set[str] = field(default_factory=set)
    edu: set[str] = field(default_factory=set)
    cert: set[str] = field(default_factory=set)

    # All of the keys for each section, because nulls will get deleted
    keys = {
        "certifications": ["date", "org", "name", "pid"],
        "education": [
            "id",
            "summary",
            "name",
            "pid",
            "majors",
            "degrees",
            "web",
            "minors",
            "date",
        ],
        "experience": [
            "id",
            "summary",
            "title",
            "name",
            "pid",
            "role",
            "company_size",
            "web",
            "date",
        ],
        "profile": [
            "job_summary",
            "job_title",
            "mobile_phone",
            "github_url",
            "github_username",
            "id",
            "facebook_username",
            "job_company_industry",
            "twitter_username",
            "inferred_years_experience",
            "job_company_id",
            "skills",
            "facebook_id",
            "summary",
            "inferred_salary",
            "linkedin_url",
            "facebook_url",
            "job_title_role",
            "twitter_url",
            "job_start_date",
            "linkedin_username",
            "location_country",
            "linkedin_id",
            "gender",
            "linkedin_connections",
            "work_email",
            "industry",
        ],
    }


def delete_none(d, keys):
    """Delete keys with None values from a dict, and add the keys to a set"""
    for k, v in list(d.items()):
        keys.add(k)
        if v is None:
            del d[k]
    return d


def make_ig(a):
    if isinstance(a, (list, tuple)):
        names = a
    else:
        names = a.split()

    _ig = ig(*names)

    def _f(d):
        return dict(zip(names, _ig(d)))

    return _f


pig = make_ig(
    "id location_country industry skills gender job_title job_title_role job_start_date job_company_id job_summary summary "
    + "job_company_industry linkedin_connections inferred_salary inferred_years_experience"
)


def expig(d):
    """Extract Experience"""

    def _e(pid, d):

        return {
            "pid": pid,
            "id": d["company"]["id"],
            "name": d["company"]["name"],
            "web": d["company"]["website"],
            "company_size": d["company"]["size"],
            "title": d["title"]["name"],
            "role": d["title"]["role"],
            "date": d["start_date"],
            "summary": d["summary"],
        }

    for e in d["experience"]:
        try:
            yield _e(d["id"], e)
        except Exception as exc:
            pass


def edig(d):
    """Extract Education"""

    def _e(pid, d):

        return {
            "pid": pid,
            "id": d["school"]["id"],
            "name": d["school"]["name"],
            "web": d["school"]["website"],
            "degrees": d["degrees"],
            "minors": d["minors"],
            "majors": d["majors"],
            "date": d["end_date"],
            "summary": d["summary"],
        }

    for e in d["education"]:
        try:
            yield _e(d["id"], e)
        except:
            pass


def certig(d):
    """Extract certifications"""

    def _e(pid, d):
        return {
            "pid": pid,
            "org": d["organization"],
            "name": d["name"],
            "date": d["start_date"],
        }

    for e in d["certifications"]:
        yield _e(d["id"], e)


contactig = make_ig(
    [
        "linkedin_url",
        "linkedin_username",
        "linkedin_id",
        "facebook_url",
        "facebook_username",
        "facebook_id",
        "twitter_url",
        "twitter_username",
        "github_url",
        "github_username",
        "work_email",
        "mobile_phone",
    ]
)


def extract_to_dict(e: Dict, pkeys):
    import numpy as np

    p = pig(e)
    p.update(contactig(e))
    p = delete_none(p, pkeys.profile)
    p["experience"] = [delete_none(e, pkeys.exp) for e in expig(e)]
    p["education"] = [delete_none(e, pkeys.edu) for e in edig(e)]
    p["certs"] = [delete_none(e, pkeys.cert) for e in certig(e)]

    exp_sizes = []
    for e in p["experience"]:
        exp_sizes.append(len(e.get("summary", "")))

    p["_n_exp"] = len(exp_sizes)
    p["_min_exp_len"] = min(exp_sizes) if exp_sizes else 0
    p["_mean_exp_len"] = np.mean(exp_sizes) if exp_sizes else 0

    return p
