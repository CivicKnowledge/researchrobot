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
        self.titles_pkg_url = titles_pkg_url or soc_titles_pkg_default
        self.embed_pkg_url = embed_pkg_url or soc_embed_pkd_default



        self.whoosh_dir = Path(whoosh_dir) if whoosh_dir else Path('./.whoosh')

        if not self.whoosh_dir.exists():
            self.whoosh_dir.mkdir(parents=True)

    @cached_property
    def embed_pkg(self):
        return open_package(self.embed_pkg_url)

    @cached_property
    def titles_pkg(self):
        return open_package(self.titles_pkg_url)


    @cached_property
    def occ_df(self):

        occ = self.titles_pkg.reference('onet_occ').dataframe()
        occ.columns = ['soc', 'title', 'desc']

        return occ

    @cached_property
    def embed_df(self):

        def rebuild_embd(v):
            return np.array(json.loads(v))

        rwe = self.embed_pkg.resource('onet_occupations').dataframe()
        rwe['embeddings'] = rwe.embeddings.apply(rebuild_embd)
        return rwe

        oce = self.embed_pkg.resource('onet_rewrites').dataframe()
        oce['embeddings'] = oce.embeddings.apply(rebuild_embd)

        return oce

        def mean_embed(g):
            return np.mean(g.embeddings.values)

        a = rwe.groupby('soc').apply(mean_embed).to_frame('embeddings')
        b = oce.groupby('soc').apply(mean_embed).to_frame('embeddings')

        mean_embd = pd.concat([a,b]).groupby('soc').apply(mean_embed).to_frame('embeddings').reset_index()

        return mean_embd

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
                rows.append({'soc': r['soc'],
                             'score': r.score,
                             'alt_title': r['title'],
                             'soc_title': self.title_map.get(r['soc'])})

            t = pd.DataFrame(rows)

            if len(t) > 0:

                return t[t.score > .8].iloc[:10]
            else:
                return pd.DataFrame([], columns=['soc','score','alt_title','soc_title'])

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

            hits_df = pd.DataFrame(hits)\
                .sort_values('score', ascending=False)\
                .drop_duplicates(subset=['soc'])

            mhits_df = pd.DataFrame(hits) \
                .groupby('soc') \
                .agg({'score': ['mean', 'count']}) \
                .set_axis(['score', 'score_count'], axis=1) \
                .sort_values('score', ascending=False) \

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

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(.40, 1))

        # title hits
        th = self.search_title(title).rename(columns={'score': 'title_score'})

        alt_title_map = {r.soc: r.alt_title for idx, r in th.iterrows()}

        th = th[['soc','title_score']].groupby('soc').agg({'title_score': 'sum'}).reset_index()

        # body hits
        z = self.search_embed(embed)[0]
        bh = z['hits'].rename(columns={'score': 'body_score'})
        bs_max = bh.body_score.max()

        bh = bh[['soc','body_score']].groupby('soc').agg({'body_score': 'sum'}).reset_index()

        df = th[['soc','title_score']].merge(bh, on='soc', how='outer')

        # The second fillna() handles the case where all of the series is nan,
        # in which case the first fillna() will not fill anything because of the nan
        # in .min()
        df['body_score'] = df['body_score'].fillna(df['body_score'].min() / 2).fillna(.1)
        df['title_score'] = df['title_score'].fillna(df['title_score'].min() / 2).fillna(.1)

        df['alt_title'] = df.soc.apply(lambda v: alt_title_map.get(v))
        df['soc_title'] = df.soc.apply(lambda v: self.title_map.get(v))

        df['body_score_scaled'] = scaler.fit_transform(df[['body_score']])
        df['title_score_scaled'] = scaler.fit_transform(df[['title_score']])

        df['score'] = ((df.body_score_scaled+.25) * df.title_score_scaled)/1.25

        return df.sort_values('score', ascending=False)
