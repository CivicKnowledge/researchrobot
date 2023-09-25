""" Classify text using Standard Occupation Codes
"""
from typing import List, Dict, Any

from metapack import open_package
from pathlib import Path
from whoosh import index
from whoosh.index import open_dir, create_in, EmptyIndexError
from whoosh.fields import *
from whoosh.qparser import QueryParser, FuzzyTermPlugin, PhrasePlugin, SequencePlugin
from whoosh.query import FuzzyTerm

import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

from functools import cached_property

soc_titles_pkg_default = 'index:civicknowledge.com-onet_soc_db'
soc_embed_pkd_default = 'index:civicknowledge.com-onet_soc-embed'

class SOCClassifier:
    def __init__(self, titles_pkg_url=None,
                 embed_pkg_url=None,
                 whoosh_dir=None):

        # use the default package if the user doesn't specify one
        titles_pkg_url = titles_pkg_url or soc_titles_pkg_default
        embed_pkg_url = embed_pkg_url or soc_embed_pkd_default

        # load the packages
        self.titles_pkg = open_package(titles_pkg_url)
        self.embed_pkg = open_package(embed_pkg_url)

        self.whoosh_dir = Path(whoosh_dir) if whoosh_dir else Path('./.whoosh')

        if not self.whoosh_dir.exists():
            self.whoosh_dir.mkdir(parents=True)

    @cached_property
    def occ_df(self):

        occ = self.titles_pkg.reference('onet_occ').dataframe()
        occ.columns = ['soc', 'title', 'desc']

        return occ

    @cached_property
    def embed_df(self):
        mean_ebd = self.embed_pkg.resource('mean_ebd').dataframe()
        mean_ebd['embeddings'] = mean_ebd.embeddings.apply(json.loads)

        return mean_ebd

    @cached_property
    def titles_df(self):
        titles = self.titles_pkg.resource('titles').dataframe()

        return titles

    @cached_property
    def title_map(self):

        title_map = { r.soc:r.title for idx, r in self.occ_df.iterrows() }
        return title_map

    @cached_property
    def whoosh_index(self):
        """Create the whoosh index"""

        try:
            ix = open_dir(self.whoosh_dir)
            if ix.doc_count() > 0:
                return ix

        except EmptyIndexError:
            pass

        # Nope, we need to build the index
        schema = Schema(id=ID(unique=True, stored=True),
                        soc=TEXT(stored=True), title=TEXT(stored=True))
        ix = index.create_in(self.whoosh_dir, schema)

        writer = ix.writer()

        try:

            for idx, r in self.titles_df.reset_index().iterrows():
                writer.update_document(id=str(idx), soc=r.soc, title=r.title)
        finally:
            writer.commit()

        return ix

    @classmethod
    def generate_whoosh_query(cls, s):
        """Generate a whoosh query from a string. Breaks the string into words
        and composes sub-sets of the words. """

        words = s.split()

        words = [w for w in words if w.strip().lower() not in ('and', 'or')]

        n = len(words)

        # Generate all possible subsets of two or more words
        subsets = []
        for i in range(n):
            for j in range(i + 2, n + 1):
                subsets.append(words[i:j])

        # Join the words in each subset with AND and wrap them in parentheses
        and_queries = ["(" + " AND ".join(subset) + ")" for subset in subsets]

        # Join the subsets with OR
        query = " OR ".join(and_queries)

        if len(words) == 2:
            query += f" OR {words[0]} OR {words[1]} "

        query += f" OR \"{s}\" "

        return query

    def search_title(self, title, print_query=False):

        ix = self.whoosh_index

        with ix.searcher() as searcher:
            parser = QueryParser("title", ix.schema, termclass=FuzzyTerm)
            # parser.add_plugin(FuzzyTermPlugin())
            # parser.remove_plugin_class(PhrasePlugin)
            # parser.add_plugin(SequencePlugin())

            qs = self.generate_whoosh_query(title)

            if print_query:
                print(qs)

            query = parser.parse(qs)
            results = searcher.search(query)

            rows = []
            for r in results:
                rows.append({'soc': r['soc'], 'score': r.score, 'alt_title': r['title'],
                             'soc_title': self.title_map.get(r['soc'])})

            t = pd.DataFrame(rows)

            if len(t) > 0:
                t['score'] = t.score / t.score.max()

                return t[t.score > .8].iloc[:5]
            else:
                return t


    def search_embed(self, q_df, nhits=30) -> list[dict[str, Any]]:
        """Find the top SOC codes for a query dataframe, using embeddings"""

        m_df = self.embed_df

        if not isinstance(q_df, pd.DataFrame):
            q_df = pd.DataFrame({'embeddings': [q_df]})

        similarity_matrix = cosine_similarity(q_df["embeddings"].tolist(), m_df["embeddings"].tolist())

        sorted_indices = np.argsort(similarity_matrix, axis=1)

        results: list[dict[str, Any]] = []

        for (idx, r), indexes, scores in zip(q_df.iterrows(), sorted_indices[:, -nhits:], similarity_matrix):

            hits = []
            for (m_idx, m_row), score in zip(m_df.iloc[indexes].iterrows(), scores[indexes]):
                m_row['score'] = score
                del m_row['embeddings']

                hits.append(m_row.to_dict())

            hits_df = pd.DataFrame(hits).sort_values('score', ascending=False).reset_index().drop_duplicates(subset=['soc'])
            mhits_df = pd.DataFrame(hits) \
                .groupby('soc') \
                .agg({'score': ['mean', 'count']}) \
                .set_axis(['score', 'score_count'], axis=1) \
                .sort_values('score', ascending=False) \
                .reset_index()
            d = {
                'q_idx': idx,
                'q_row': r,
                'hits': hits_df,
                'mhits': mhits_df
            }

            results.append(d)

        return results

    def search(self, title, embed):
        """Search for titles with Whoosh, and descriptions using embeddings,
        combining the scores. The top scoring title will have a value of 1,
        and a perfect embedding match is also 1. the scores for all of the
        returned are added together per SOC, so if the set of hits has more than
        one occurance of an SOC, the total score from this function can be larger than 2. """

        from collections import defaultdict

        h = self.search_title(title)

        d = defaultdict(float)

        for idx, r in h.iterrows():
            d[r.soc] += r.score

        z = self.search_embed(embed)[0]

        for idx, r in z['hits'].iterrows():
            d[r.soc] += r.score

        return dict(reversed(sorted(d.items(), key=lambda x: x[1])))
