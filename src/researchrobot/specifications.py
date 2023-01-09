"""
Pares and process question specifications
"""


def parse_specification(e, s=None):
    """Parse a specification into a dictionary"""

    if s is None:
        q, s = e["question"], e["specification"]
    else:
        q = e

    s = s.replace("]", "]---")
    # d = {k:None for k in 'measure of by in during'.split() }
    d = {"question": q}
    for p in [e.strip() for e in s.split("---") if e.strip()]:
        if p.startswith("["):
            d["measure"] = p.strip("[]").strip()
        else:
            try:
                pre, term = [e.strip() for e in p.split(" ", 1)]
                d[pre] = term.strip("[]").strip()
            except Exception as e:
                print("Error", s)
                if "errors" not in d:
                    d["errors"] = []
                d["errors"].append((e, p))

    return d


def unparse_specification(r):
    """Unparse a specification dictionary into a string"""

    parts = []
    parts.append(f"[{r.measure}]")

    d = r.fillna("").to_dict()

    for part_name in "of for by in during".split():
        if d[part_name]:
            parts.append(f"{part_name} [{d[part_name]}]")

    return {"question": r.question, "specification": " ".join(parts)}
