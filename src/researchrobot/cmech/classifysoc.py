""" Classify text using Standard Occupation Codes, using Whoosh for
full text search and local cosine distances
"""
import json
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import Union
import logging
from tqdm.auto import tqdm
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from metapack import open_package
from sklearn.metrics.pairwise import cosine_similarity
from whoosh import index
from whoosh.fields import *
from whoosh.index import open_dir, EmptyIndexError
from whoosh.qparser import QueryParser
from whoosh.query import FuzzyTerm

from researchrobot import ObjectStore
from researchrobot.cmech.cache import get_classification_queues

soc_titles_pkg_default = 's3://linkedin.civicknowledge.com/packages/civicknowledge.com-onet_soc_db-1.2.1.csv'
soc_embed_pkd_default = 's3://linkedin.civicknowledge.com/packages/civicknowledge.com-onet_soc-embed-1.2.4.csv'

logger = logging.getLogger(__name__)


class SOCClassifier:

    @classmethod
    def get_data(cls, titles_pkg_url=None, embed_pkg_url=None ):
        # use the default package if the user doesn't specify one
        titles_pkg_url = titles_pkg_url or soc_titles_pkg_default
        titles_okg = open_package(titles_pkg_url)
        occ = titles_pkg.reference('onet_occ').dataframe()
        occ.columns = ['soc', 'title', 'desc']

        titles = self.titles_pkg.resource('titles').dataframe()

        bob = fooarber
        

        embed_pkg_url = embed_pkg_url or soc_embed_pkd_default
        embed_pkg = open_package(embed_pkg_url)
        def rebuild_embd(v):
            return np.array(json.loads(v))

        rwe = embed_pkg.resource('onet_occupations').dataframe()
        rwe['embeddings'] = rwe.embeddings.apply(rebuild_embd)

        title_map = {r.soc: r.title for idx, r in self.embd_df.iterrows()}

        return {
            'occ': occ,
            'titles': titles,
            'embed': rwe,
            'title_map': title_map
        }

    def __init__(self,
                 whoosh_dir=None):

        self.whoosh_dir = Path(whoosh_dir) if whoosh_dir else Path('./.whoosh')

        if not self.whoosh_dir.exists():
            self.whoosh_dir.mkdir(parents=True)


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
                return pd.DataFrame([], columns=['soc', 'score', 'alt_title', 'soc_title'])

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

            hits_df = pd.DataFrame(hits) \
                .sort_values('score', ascending=False) \
                .drop_duplicates(subset=['soc'])

            mhits_df = pd.DataFrame(hits) \
                .groupby('soc') \
                .agg({'score': ['mean', 'count']}) \
                .set_axis(['score', 'score_count'], axis=1) \
                .sort_values('score', ascending=False)

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

        th = th[['soc', 'title_score']].groupby('soc').agg({'title_score': 'sum'}).reset_index()

        # body hits
        z = self.search_embed(embed)[0]
        bh = z['hits'].rename(columns={'score': 'body_score'})
        bs_max = bh.body_score.max()

        bh = bh[['soc', 'body_score']].groupby('soc').agg({'body_score': 'sum'}).reset_index()

        df = th[['soc', 'title_score']].merge(bh, on='soc', how='outer')

        # The second fillna() handles the case where all of the series is nan,
        # in which case the first fillna() will not fill anything because of the nan
        # in .min()
        df['body_score'] = df['body_score'].fillna(df['body_score'].min() / 2).fillna(.1)
        df['title_score'] = df['title_score'].fillna(df['title_score'].min() / 2).fillna(.1)

        df['alt_title'] = df.soc.apply(lambda v: alt_title_map.get(v))
        df['soc_title'] = df.soc.apply(lambda v: self.title_map.get(v))

        df['body_score_scaled'] = scaler.fit_transform(df[['body_score']])
        df['title_score_scaled'] = scaler.fit_transform(df[['title_score']])

        df['score'] = ((df.body_score_scaled + .25) * df.title_score_scaled) / 1.25

        return df.sort_values('score', ascending=False)


class SocPartProcessor:

    def __init__(self,
                 df,
                 titles_pkg_url=None,
                 embed_pkg_url=None,
                 whoosh_dir='./whoosh_dir',
                 progress=False):
        """
        :param rc_config: Cache configuration
        :param part_name: name of the part we are processing
        :param titles_pkg_url: Url to metapack package for SOC titles
        :param embed_pkg_url:  Url to metapack package for SOC embeddings
        :param whoosh_dir: Directory name for storing Whoosh index
        :param progress: If true, show progress bar
        :param df: If provided, use this as the dataframe to process, rather than get it from the cache
        """

        self.df = df
        self.progress = progress
        self.whoosh_dir = whoosh_dir
        self.titles_pkg_url = titles_pkg_url
        self.embed_pkg_url = embed_pkg_url

        if self.progress:
            self.pb = lambda x, size: tqdm(x, leave=False, total=size)
        else:
            self.pb = lambda x, size: x

        self.sc = SOCClassifier(titles_pkg_url=self.titles_pkg_url,
                                embed_pkg_url=self.embed_pkg_url,
                                whoosh_dir='./whoosh_dir')

        self.result_df = None

    def match_embeddings(self):
        """Worker function to Run embeddings given the keys for the text segments"""

        frames = []

        for idx, r in self.pb(self.df.iterrows(), len(self.df)):
            t = self.sc.search(r.title, r.embeddings)
            t = t.iloc[:10]
            t['source'] = r.source
            t['exp_id'] = r.exp_id
            t['pid'] = r.pid

            frames.append(t)

        self.result_df = pd.concat(frames)
        return self.result_df


class SocProcessor:

    def __init__(self, rc_config: Union[dict, ObjectStore], n_jobs=8,
                 titles_pkg_url=None,
                 embed_pkg_url=None,
                 ):

        self.n_jobs = n_jobs
        self.rc_config = rc_config

        if isinstance(rc_config, dict):
            rc = ObjectStore(**rc_config)
        else:
            rc = rc_config

        self.q = get_classification_queues(rc)

    @cached_property
    def embd_keys(self):
        """Get the keys for the experience embeddings dataframes"""
        return list(self.q.exp_embeddings.list())

    @cached_property
    def tem_keys(self):
        """Get the keys for the already processed matched"""
        return list(self.q.parts.sub('te_matches').list())

    @cached_property
    def remain_match_keys(self):
        """Get the keys for the experience embeddings dataframes that have not been matched"""
        return list(sorted(set(self.embd_keys) - set(self.tem_keys)))

    def chunk_embed_df(self, df, chunk_size=1000):
        """Chunk the embeddings dataframe into smaller dataframes"""

        for i in range(0, df.shape[0], chunk_size):
            yield df[i:i + chunk_size]

    @staticmethod
    def match_worker(chunk):
        sp = SocPartProcessor(chunk)
        sp.match_embeddings()
        return sp.result_df

    def run(self):

        for key_n, key in tqdm(enumerate(self.remain_match_keys),
                               total=len(self.remain_match_keys),
                               desc='Match parts'):
            chunks = list(self.chunk_embed_df(self.q.exp_embeddings[key]))

            frames = Parallel(n_jobs=self.n_jobs)(
                delayed(self.match_worker)(chunk)
                for chunk in tqdm(
                    chunks,
                    desc=f'Match chunks for {key}'))

            df = pd.concat(frames)

            self.q.parts.sub('te_matches')[key] = df

        return results
