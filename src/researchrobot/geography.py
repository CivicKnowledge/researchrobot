"""

"""
from collections import Counter
from functools import cached_property

import metapack as mp
import numpy as np
import pandas as pd
import rapidfuzz as rfz
import spacy

DEFAULT_PACKAGE = (
    "https://library.metatab.org/civicknowledge.com-geonames-2022-1.2.3.csv"
)


class GeographySearch:
    def __init__(self, df: pd.DataFrame = None):
        """Create a new GeographySearch object from a dataframe of region names

        :param df: A dataframe with a column named "name" containing the names of regions
        :type df: pd.DataFrame

        """

        if df is None:
            pkg = mp.open_package(DEFAULT_PACKAGE)
            df = pkg.resource("names").dataframe().fillna("")

        self.df = df
        self.nlp = spacy.load("en_core_web_sm")

        self._add_tokens()

    def _add_tokens(self):
        self.df["tok"] = [
            " ".join(self.tokenize(r["name"])) for idx, r in self.df.iterrows()
        ]

    def tokenize(self, v: str):
        """Tokenize a geoname"""
        import re

        return [e.lower().replace("'", "") for e in re.split(r"\s|,", v) if e]

    @cached_property
    def index(self):
        """Return an inverted index of tokens in the names of regions"""
        rows = []
        for idx, r in self.df.iterrows():

            minsize = 2 if r.category == "state" else 3

            for p in self.tokenize(r["name"]):
                if len(p) >= minsize:
                    rows.append({"word": p, "idx": idx, "sz": len(p)})

        return pd.DataFrame(rows)

    def filter_words(self, query, inv_df):

        words = []

        for w in self.tokenize(query):
            s = inv_df[inv_df.sz.between(len(w) - 1, len(w) + 1)]
            match, score, widx = rfz.process.extractOne(w, s.word)
            if score > 90:
                words.append(match)

        return words

    def match_region(self, query):

        query = " ".join(self.tokenize(query))
        query = " ".join(
            [str(e) for e in self.nlp(query) if e.pos_ not in ("ADJ", "ADV", "ADP")]
        )

        score1 = pd.Series([rfz.fuzz.token_sort_ratio(query, t) for t in self.df.tok])
        score2 = pd.Series([rfz.fuzz.partial_ratio(query, t) for t in self.df.tok])
        score = score1 + score2

        score_sort = score.argsort()[::-1]

        df = self.df.iloc[score_sort].head(20).drop_duplicates(subset="ref")
        return df.sort_values(["weight"], ascending=False)
