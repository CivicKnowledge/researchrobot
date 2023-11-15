""" Classify experiences using  TypeSense
"""
import typesense
from typesense.exceptions import ObjectAlreadyExists
import json
from more_itertools import chunked
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from functools import cached_property
from researchrobot.cmech.soc import parse_soc, make_soc
import metapack as mp
import logging
from researchrobot import  ObjectStore, cache_dl
import textdistance

logger = logging.getLogger(__name__)

def cosine_similarity(A,B):
    from numpy.linalg import norm
    return np.dot(A,B)/(norm(A)*norm(B))

def mk_exp_decay(range):
    """Make a function that exponentially decays from about max_val to about min val over
    the values (0,range) """

    max_val = 1
    min_val = 0.01
    b = (min_val / max_val) ** (1 / (range - 1))  # Calculating b

    def _f(v):
        return max_val * b ** v

    return _f

def resolve_cache(cache_config):

    if cache_config is None:
        return None

    if isinstance(cache_config, dict):
        return ObjectStore.new(**cache_config)
    else:
        return cache_config


class SocEdData:

    soc_titles_pkg_url = 's3://linkedin.civicknowledge.com/packages/civicknowledge.com-onet_soc_titles-1.2.2.csv'
    embd_pkg_url = 's3://linkedin.civicknowledge.com/packages/civicknowledge.com-onet_soc-embed-3.1.1.csv'
    ce_pkg_url = 's3://linkedin.civicknowledge.com/packages/credentialengine.org-registry-1.1.2.csv'
    nces_pkg_url = 's3://linkedin.civicknowledge.com/packages/nces.ed.gov-cip-2020.csv'

    def __init__(self, data_cache=None, embed_cache=None):

        self.data_cache = resolve_cache(data_cache)
        self.embed_cache = resolve_cache(embed_cache)

    def clean_cache(self):
        if self.data_cache is not None:
            del self.data_cache['occ_df']

    def embed_text(self, text):
        from researchrobot.embeddings import get_embeddings
        from hashlib import sha256

        text = text.lower()
        key = sha256(text.encode('utf8')).hexdigest()
        if self.embed_cache is not None and key in self.embed_cache:
            embeddings = self.embed_cache[key]
            logger.debug(f'Cache hit for {key},  {text[:20]}')

        else:
            embeddings = [e[1] for e in get_embeddings([text])][0]
            logger.debug(f'Embedding  {text[:20]}')
            if self.embed_cache is not None:
                self.embed_cache[key] = embeddings
                logger.debug(f'Cache save for {key},  {text[:20]}')

        return embeddings

    @cached_property
    def embeddings_length(self):
        return len(self.embd_df['embeddings'].iloc[0])

    jobs_titles_schema = {
        'name': 'job_titles',
        'fields': [
            {'name': 'idx', 'type': 'int32'},
            {'name': 'soc', 'type': 'string'},
            {'name': 'title', 'type': 'string'},
            {'name': 'source', 'type': 'string'},
            {'name': 'soc_title', 'type': 'string'}
        ]
    }

    edu_schema = {
        'name': 'edu',
        'fields': [
            {'name': 'idx', 'type': 'int32'},
            {'name': 'code', 'type': 'string'},
            {'name': 'title', 'type': 'string'},
            {'name': 'source', 'type': 'string'},
        ]
    }

    def embd_schema(self, name):
        return {
            'name': name,
            'fields': [
                {'name': 'eid', 'type': 'string'},
                {'name': 'text', 'type': 'string'},
                {'name': 'kind', 'type': 'string'},
                {"name": "embeddings", "type": "float[]", "num_dim": self.embeddings_length},
            ]
        }

    @cached_property
    def vec_dims(self):
        """Return the size of the output cip + soc vector"""
        return max(self.sparse_map.values()) + 1

    @cached_property
    def nces_df(self):
        """CIP Codes fom the NCES"""
        pkg = mp.open_package(self.nces_pkg_url)
        return pkg.resource('cip').dataframe().assign(source='nces')

    @cached_property
    def ce_df(self):
        """CIP Codes from the Credential Engine"""
        pkg = mp.open_package(self.nces_pkg_url)
        return pkg.resource('cip').dataframe().assign(source='ce')

    @cached_property
    def embd_df(self):
        """SOC Codes and embeddings, from O*NET"""

        pkg = mp.open_package(self.embd_pkg_url)
        df =  pkg.resource('onet_embd').dataframe()
        df['text'] = df.text.str.lower()
        df['embeddings'] = df.embeddings.apply(json.loads)
        return df

    @property
    def text_embd_df(self):
        return self.embd_df[self.embd_df.kind == 'text']

    @property
    def title_embd_df(self):
        return self.embd_df[self.embd_df.kind == 'title']

    @cached_property
    def ent_df(self):
        pkg = mp.open_package(self.embd_pkg_url)
        df =  pkg.resource('onet_ent').dataframe()
        df['detailed'] = df['soc'].apply(lambda v: parse_soc(v)['detailed'])
        df['idx'] = df['detailed'].apply(lambda v: self.sparse_map.get(v))

        return df

    @cached_property
    def _soc_df(self):
        """Returns first part of SOC dataframe, for breaking a cycle with title_map"""

        pkg = mp.open_package(self.soc_titles_pkg_url)
        soc_df = pkg.resource('titles').dataframe()

        soc_df['idx'] = list(soc_df.index)
        soc_df['title'] = soc_df['title'].str.lower()
        soc_df['detailed'] = soc_df['soc'].apply(lambda v: parse_soc(v)['detailed'])
        return soc_df

    @cached_property
    def soc_df(self):
        soc_df = self._soc_df
        soc_df['soc_title'] = soc_df['soc'].apply(lambda v: self.title_map.get(v, v))
        soc_df['idx'] = soc_df['detailed'].apply(lambda v: self.sparse_map.get(v))
        return soc_df

    @cached_property
    def _cip_df(self):

        df = pd.concat([self.ce_df, self.nces_df]).sort_values(['code'])

        df['title'] = df['title'].str.lower()

        return df

    @cached_property
    def cip_df(self):
        df = self._cip_df
        df['idx'] = df['discipline'].apply(lambda v: self.sparse_map.get(v))
        return df


    @cached_property
    def code_list(self):

        # For CIP, using the discipline codes
        cip_codes = list(sorted(self._cip_df.discipline.unique().tolist()))
        cip_codes.append('99.99')

        # For SOC, using the detailed codes, which don't have the '.00' suffix
        soc_codes = list(sorted(self._soc_df.detailed.unique().tolist()))
        soc_codes.append('99-9999')

        return cip_codes + soc_codes

    @cached_property
    def sparse_map(self):
        from itertools import chain
        from researchrobot.cmech.soc import parse_soc, make_soc

        sparse_map = {v: int(k) for k, v in enumerate(self.code_list)}

        return sparse_map

    @staticmethod
    def dtob26(n):
        n += 1
        result = ""
        while n > 0:
            n -= 1
            result = chr(ord('a') + n % 26) + result
            n //= 26
        return result

    @staticmethod
    def b26tod(s):
        result = 0
        for c in s:
            result = result * 26 + (ord(c) - ord("a")) + 1
        return result -1

    @cached_property
    def title_map(self):

        t = self.cip_df
        t = t[(t['cat'] != 'program') & (t.source == 'nces')].drop_duplicates(subset=['discipline'])
        cip_titles = {r.discipline: r.title for idx, r in t.iterrows()}

        # Prefer the higher level ( '.00' ) over the lower level ( '.01' )
        soc = self._soc_df[self._soc_df.source == 'occ'].sort_values('soc').drop_duplicates(subset=['detailed'])


        soc_titles = { r.detailed: r.title for idx, r in soc.iterrows()}

        d = {**cip_titles, **soc_titles}
        o = {}

        # create the final map, which guarantees to have all of the codes in the sparse map
        for k,v in self.sparse_map.items():
            if k in d:
                o[k] = d[k] # The code
                o[v] = d[k] # The index number
                o[SocEdData.dtob26(v)] = d[k] # The index number in base 26
            else:
                o[k] = k
                o[v] = k

        return o

    @cached_property
    def idx_title_map(self):
        """Map indexed to titles"""
        return { v: self.title_map[k] for k,v in self.sparse_map.items()}

    @cached_property
    def text_map(self):
        """Map of soc and idx to jobs text"""

        #[['detailed', 'idx', 'soc', 'text']]
        m = {}
        for idx, r in self.ent_df[self.ent_df.kind == 'text'].iterrows():
            m[r['detailed']] = r['text']
            m[r['soc']] = r['text']
            m[r['idx']] = r['text']

        return m



    @cached_property
    def related_df(self ):
        """"Related occupations"""

        relat = pd.read_excel('https://www.onetcenter.org/dl_files/database/db_27_2_excel/Related%20Occupations.xlsx')
        relat = relat.iloc[:, [0, 2, 4]]
        relat.columns = ['soc1', 'soc2', 'tier']

        return relat

    @cached_property
    def soc_cip_df(self):
        """Links between CIP and SOC codes"""
        soc_cip_url = 'https://nces.ed.gov/ipeds/cipcode/Files/CIP2020_SOC2018_Crosswalk.xlsx'

        fn = cache_dl(soc_cip_url)

        cip_soc = pd.concat([pd.read_excel(fn, 1)[['CIP2020Code', 'SOC2018Code']],
                             pd.read_excel(fn, 2)[['CIP2020Code', 'SOC2018Code']]])\
            .drop_duplicates()

        cip_soc.columns = ['cip', 'soc']
        cip_soc['cip'] = cip_soc.cip.apply(lambda v: "{:07.4f}".format(v))
        cip_soc['cip_discipline'] = cip_soc.cip.apply(lambda v: v[:5])
        cip_soc['soc_detailed'] = cip_soc.soc.apply(lambda v: parse_soc(v)['detailed'])

        cip_soc['cip_idx'] = cip_soc.cip_discipline.apply(lambda v : self.sparse_map[v]).astype(int)

        # Get b/c there are still missing code, mostly "all other" codes, like
        # https://www.onetcodeconnector.org/ccreport/25-1069.00
        cip_soc['soc_idx'] = cip_soc.soc_detailed.apply(lambda v : self.sparse_map.get(v))

        cip_soc = cip_soc.dropna()  # There are a lot of military and "other" SOCs that are missing

        cip_soc['soc_idx'] = cip_soc['soc_idx'].astype(int)

        return cip_soc

class SocEdClassifier:

    def __init__(self, data: SocEdData, typesense_client: typesense.Client):

        self.data = data
        self.client = typesense_client

        pass

    def create_collection(self, df, sch, delete=False,progress=False):

        logger.debug(f"Loading {sch['name']} schema")

        try:
            self.client.collections.create(sch)
        except ObjectAlreadyExists:
            if delete:
                logger.debug(f"Collection {sch['name']} already exists, deleting and recreating")
                self.client.collections[sch['name']].delete()
                self.client.collections.create(sch)
            else:
                logger.debug(f"Collection {sch['name']} already exists, skipping")
                return

        chunks = list(chunked(df.to_dict(orient='records'), 100))
        pb = tqdm(chunks) if progress else chunks;

        for c in pb:
            r = self.client.collections[sch['name']].documents.import_(c)

            for e in r:
                if not e['success']:
                    print("ERROR for ", sch, e['error'])
                    return

    def create_collections(self, delete=False, progress=False):

        for df, sch in ((self.data.cip_df, self.data.edu_schema),
                        (self.data.soc_df, self.data.jobs_titles_schema),
                        (self.data.title_embd_df, self.data.embd_schema('embd_title')),
                        (self.data.text_embd_df, self.data.embd_schema('embd_text'))):

            self.create_collection(df, sch, delete=delete,  progress=progress)

    def embd_search(self, e_or_text, collection='embd_title'):

        if isinstance(e_or_text, str):
            # It's text, not an embedding, so embed it.

            embeddings = self.data.embed_text(e_or_text)

        else:
            embeddings = e_or_text

        cols = ['soc', 'eid', 'text', 'kind']
        null_df = pd.DataFrame(columns=cols)

        if embeddings is None:
            return null_df

        n_hits = 30

        search_parameters = {
            'searches': [{
                'vector_query': f' embeddings:([{",".join(str(e) for e in embeddings)}], k:{n_hits})',
                'q': '*',
                'collection': collection,
                'per_page': n_hits
            }]}

        r = self.client.multi_search.perform(search_parameters, {})

        if len(r['results']) == 0:
            return null_df



        hits = r['results'][0]['hits']

        rows = []

        for e in hits:
            doc = e['document']
            doc['score'] = cosine_similarity(doc['embeddings'], embeddings).round(5)
            del doc['embeddings']
            # doc['detailed'] = parse_soc(doc['soc'])['detailed']
            # detailed = parse_soc(doc['soc'])['detailed']
            # doc['title'] = self.data.title_map.get(detailed, detailed)

            rows.append(doc)

        t = pd.DataFrame(rows).sort_values('score', ascending=False)

        # decay_f = mk_exp_decay(n_hits)
        # t['f'] = decay_f(t.index)

        t = t.merge(self.data.ent_df[['eid', 'soc', 'detailed','idx']], on='eid')

        return t

    def job_body_search_text(self, text):

        cols = ['soc', 'detailed', 'idx', 'score', 'title', 'source']
        null_df = pd.DataFrame(columns=cols)

        if text is None:
            return null_df

        t = self.embd_search(text, collection='embd_text')
        t['title'] = t.detailed.apply(lambda v:self.data.title_map.get(v, v))

        t = t[['detailed', 'idx', 'kind', 'title', 'text', 'score']].rename(columns={'score':'body_score'})

        return t

    def job_title_search_text(self, title):

        cols = ['detailed', 'idx', 'title', 'occ_title', 'jaccard', 'lcsstr', 'source']
        null_df = pd.DataFrame(columns=cols)

        if not title:
            return null_df

        search_parameters = {
            'q': title,
            'query_by': 'title',
            #'prioritize_token_position': True,
            'prioritize_exact_match': True,
            'limit': 50

        }

        rows = []
        hits = self.client.collections['job_titles'].documents.search(search_parameters)['hits']

        for e in hits:
            doc = e['document']
            del doc['soc_title']

            doc['jaccard'] = textdistance.jaccard.normalized_similarity(doc['title'], title)
            doc['lcsstr'] = textdistance.lcsstr.similarity(title, doc['title'])/ len(title)

            doc['occ_title'] = self.data.title_map.get(doc['detailed'], doc['detailed'])

            rows.append(doc)

        try:
            t = pd.DataFrame(rows).sort_values('jaccard', ascending=False).drop_duplicates(subset=['detailed'])
            return t[cols]
        except Exception as e:
            return null_df

    def job_title_search_embd(self, title):
        t = self.embd_search(title)

        if isinstance(title, str):
            pass
            #t['lcsstr']  = t.text.apply(lambda v:textdistance.lcsstr.similarity(title.lower(), v) / len(title))
            #t['jaccard'] = t.text.apply(lambda v:textdistance.jaccard.normalized_similarity(v, title))

        return t

    def job_title_search(self, title):

        t = self.job_title_search_embd(title).rename(columns={'score': 'title_embd_score'})
        b = self.job_title_search_text(title).rename(columns={'jaccard': 'title_text_score'})[
            ['detailed', 'title_text_score']]
        b = b.sort_values('title_text_score', ascending=False).drop_duplicates(subset='detailed')
        t = t.merge(b, on='detailed', how='left').fillna(0)
        t['title_score'] = t.title_embd_score + t.title_text_score
        t = t.sort_values('title_score', ascending=False).rename(columns={'text': 'title'})

        return t[['soc', 'detailed', 'idx',  'kind', 'title', 'title_embd_score', 'title_text_score', 'title_score']]

    def job_search(self, title=None, summary=None):
        """Search for jobs, with a title or embed"""

        t = self.job_title_search(title)
        b = self.job_body_search_text(summary)

        i = pd.concat([t[['idx', 'detailed', 'title']], b[['idx', 'detailed', 'title']]]).drop_duplicates(
            subset=['idx'])
        df = i.merge(t[['idx', 'title_embd_score', 'title_text_score', 'title_score']], on='idx', how='left') \
            .merge(b[['idx', 'text', 'body_score']], on='idx', how='left')
        df = df[
            ['idx', 'detailed', 'title', 'text', 'title_embd_score', 'title_text_score', 'title_score', 'body_score']]

        df['body_score'] = df.body_score.fillna(0)
        df.loc[df.text.isnull(),'text'] = df.loc[df.text.isnull(),'detailed'].apply(self.data.text_map.get)

        df = df.fillna(0)
        df['score'] = (df.title_score + df.body_score)
        score_cols = ['title_embd_score', 'title_text_score', 'title_score', 'body_score', 'score']
        df.loc[:, score_cols] = df.loc[:, score_cols].round(4)
        df = df.sort_values('score', ascending=False).drop_duplicates(subset=['idx'])

        return df

    def edu_title_search(self, major):
        search_parameters = {
            'q': major.lower(),
            'query_by': 'title',
            'prioritize_token_position': True,
            'limit': 40
        }

        hits = self.client.collections['edu'].documents.search(search_parameters)['hits']

        rows = []
        for e in hits:
            doc = e['document']
            doc['score'] = textdistance.jaccard.normalized_similarity(doc['title'], major.lower())
            doc['lcsstr'] = textdistance.lcsstr.similarity(title, doc['title']) / len(title)
            rows.append(doc)

        try:
            return pd.DataFrame(rows).sort_values('score', ascending=False).drop_duplicates(subset=['code'])
        except Exception as e:
            return pd.DataFrame()

    def make_vector(self, df, v: np.ndarray = None):

        def isnan(v):
            if v is None:
                return True
            else:
                return np.isnan(v)

        if len(df) == 0:
            return np.zeros(self.data.vec_dims)

        if v is None:
            v = np.zeros(self.data.vec_dims)

        if False:
            v[df.idx.values] += 1  # Max of 1 per dimension
        else:
            for i in df.idx:
                try:
                    if not isnan(i):
                        v[int(i)] += 1  # Max per dimension is # of times code appears in results
                except Exception as e:
                    print("ERROR", i, df.idx.values, e)
        return v



    def experience_vector(self, title=None, embed=None):
        """Return an experience vector for a title or embed"""
        if title is not None:
            v1 = self.make_vector(self.job_title_search(title))
        else:
            v1 = np.zeros(self.data.vec_dims)

        if embed is not None:
            v2 = self.make_vector(self.job_text_search(embed))
        else:
            v2 = np.zeros(self.data.vec_dims)

        v = v1 + v2

        return v

    def education_vector(self, major):

        v = self.make_vector(self.edu_title_search(major))

        return v
