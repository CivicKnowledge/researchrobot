# Open a file and read text lines with the follwing format, parsing the blocks into
# a list of dictionaries. The format is:
# ---
# QUESTION: What is the median before tax income of families by year?
# rewrite: What is the [median] [before tax income] [of families] [by year]?
# measure: before tax income
# object: family
# aggregation: median
# dimension: year
# ---
# QUESTION: How much debt do married households have, for households headed by males, by state, by year?
# rewrite: How much [debt] do [married] [households] have, for [households] [headed by males], [by state], [by year]?
# measure: debt
# object: household
# aggregation: mean
# condition: married AND male householder
# dimension: state, year


def parse_blocks(s: str):
    """Parse a string into a list of dictionaries, where each dictionary is a block of text
    delimited by a line of ---"""
    blocks = []
    block = {}
    for line in s.splitlines():
        if line == "---":
            if block:
                blocks.append(block)
                block = {}
        else:
            k, v = line.split(":", 1)
            block[k.strip()] = v.strip()
    if block:
        blocks.append(block)
    return blocks
