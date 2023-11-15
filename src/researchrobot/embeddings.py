import logging
from typing import List, Union


import numpy as np
import openai
import pandas as pd
from more_itertools import always_iterable, chunked
import asyncio
import openai_async
from openai import OpenAI

from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)
from tqdm.auto import tqdm

# No idea what is going on here.
try:
    from openai import RateLimitError
except ImportError:
    from openai.error import RateLimitError

logger = logging.getLogger(__name__)

from researchrobot.datadecomp.census_demo_conditions import (
    age_range_terms,
    inv_age_phrases,
    inv_pov_phrases,
    inv_race_iterations,
    inv_sex_phrases,
    inv_subject,
)

mdf_cols = [
    "table_id",
    "uid",
    "title",
    "bare_title",
    "stripped_title",
    "universe",
    "subject",
    "race",
    "raceeth",
    "table_age",
    "age",
    "table_sex",
    "sex",
    "poverty_status",
    "filtered_path",
]


def build_table_terms_map(df):
    groups = (("subject", inv_subject()),)

    terms = []

    for gn, d in groups:
        for k, v in d.items():
            if bool(k) and bool(v):
                terms.append({"group": gn, "key": k, "value": v})

    return pd.DataFrame(terms)


def build_columns_terms_map(df):
    groups = (
        ("sex", inv_sex_phrases()),
        ("pov", inv_pov_phrases()),
        ("age", inv_age_phrases()),
        ("age_range", age_range_terms(df)),
        ("raceeth", inv_race_iterations()),
        ("subject", inv_subject()),
    )

    terms = []

    for gn, d in groups:
        for k, v in d.items():
            if bool(k) and bool(v):
                terms.append({"group": gn, "key": k, "value": v})

    for v in df.name.unique():
        terms.append({"group": "name", "key": v, "value": v})

    return pd.DataFrame(terms)

retry_config = dict(
    wait=wait_exponential(multiplier=1, min=1, max=32),
    retry=retry_if_exception_type(RateLimitError),
    stop=(stop_after_delay(60 * 3) | stop_after_attempt(10)) )

@retry(**retry_config)
def get_embeddings(texts, add_spaces=True, normalize=True):

    client = OpenAI()

    texts = list(always_iterable(texts))

    if add_spaces:
        texts = [" " + text.strip() + " " for text in texts]

    response = client.embeddings.create(input=texts, model="text-embedding-ada-002")

    if normalize:
        normalize = lambda x: x / np.linalg.norm(x, ord=2)
    else:
        normalize = lambda x: x

    ebd = [(t, normalize(e.embedding)) for t, e in zip(texts, response.data)]

    return ebd

@retry(**retry_config)
def run_embeddings(
        terms: pd.DataFrame,
        text_col: str = "text",
        embeddings_col: str = "embeddings",
        extant_terms=None,
        normalize=True,
        progress=False,
        n_jobs=2,
):
    from itertools import chain

    from joblib import Parallel, delayed

    batched_terms = [
        pd.DataFrame(e) for e in chunked(terms.to_dict(orient="records"), 200)
    ]

    def _f(batch):
        texts = [r[text_col] for idx, r in batch.iterrows()]
        return get_embeddings(texts, normalize=normalize)

    if progress:
        pb = tqdm(batched_terms, desc=f"embedding {terms.index[0]}", leave=False)
    else:
        pb = batched_terms

    results = Parallel(n_jobs=n_jobs)(delayed(_f)(batch) for batch in pb)

    keys, embeds = list(zip(*list(chain(*results))))
    terms = terms.copy()
    terms[embeddings_col] = [np.array(e) for e in embeds]

    return terms


class EmbeddingsAsync:
    """Fetch embeddings for a dataframe of text using OpenAI's API, making
    mutiple asyncio calls in parallel. Works basically the same as run_embeddings,
    but with asyncio"""

    def __init__(self, df, text_col: str = "text", embeddings_col: str = "embeddings",
                 concurrency=2, chunk_size=200, normalize=True, progress=False,
                 cache = None):

        self.df = df
        self.text_col = text_col
        self.embeddings_col = embeddings_col
        self.normalize = normalize
        self.concurrency = concurrency
        self.chunk_size = chunk_size
        self.progress = progress
        self.semaphore = asyncio.Semaphore(self.concurrency)
        self.index_chunks = list(chunked(list(self.df.index), self.chunk_size))
        self.cache = cache

        self.completed_chunks = {}
        self.errors = {}

        self.no_op = False

        logger.debug(f"EA Init")

    async def get_embeddings(self, chunk_n, df):

        import os

        df = df.copy()
        chunk_key = f'embd-{chunk_n}-{min(df.index)}-{max(df.index)}'

        if chunk_key in self.cache:
            edf = self.cache[chunk_key]
            self.completed_chunks[chunk_n] = edf
            logger.debug(f"Using cached embeddings for chunk {chunk_key}")
            return edf

        model_name = 'text-embedding-ada-002'

        inp = df[self.text_col].to_list()

        async for attempt in AsyncRetrying(**retry_config):
            with attempt:
                try:
                    resp = await openai_async.embeddings(
                        api_key=os.environ["OPENAI_API_KEY"],
                        timeout=2,
                        payload={"model": model_name, "input": inp},
                    )

                    response = resp.json()

                    if 'error' in response:
                        if 'Rate' in response['error']['message']:
                            raise RateLimitError(response['error']['message'])

                    if not 'data' in response:
                        raise Exception(f'No data in response')

                except RateLimitError as e:
                    raise
                except Exception as e:
                    self.errors[chunk_n] =  (e, None, chunk_n, df)
                    logger.debug(f"Error getting embeddings for chunk {chunk_key}: {e}")
                    return None

                if self.normalize:
                    normalize = lambda x: x / np.linalg.norm(x, ord=2)
                else:
                    normalize = lambda x: x

                try:
                    df[self.embeddings_col] = [normalize(e["embedding"]) for e in response["data"]]
                except KeyError as e:
                    self.errors[chunk_n] = (e, response, chunk_n, df)
                    logger.debug(f"Error normalizing embeddings for chunk {chunk_key}: {e}")
                    return None

        self.completed_chunks[chunk_n] = df

        if self.cache is not None:
            self.cache[chunk_key] = df

        if chunk_n in self.errors:
            del self.errors[chunk_n]

        logger.debug(f"Completed {chunk_key}")

        return df

    async def worker(self, chunk_n, text_chunk):

        if chunk_n in self.completed_chunks:
            return

        async with self.semaphore:
            return await self.get_embeddings(chunk_n, text_chunk)



    async def run(self):
        from tqdm.asyncio import tqdm

        tasks = [self.worker(chunk_n, self.df.loc[chunk])
                 for chunk_n, chunk in enumerate(self.index_chunks)]

        if self.progress:
            await tqdm.gather(*tasks)
        else:
            await asyncio.gather(*tasks)

        return len(self.errors) == 0

    @property
    def results_df(self):
        return pd.concat(self.completed_chunks.values()).sort_values(['pid', 'exp_id'])


class EmbedDb:

    def __init__(
            self, collection_name: str, host=None, port=19530, alias="default", **kwargs
    ):
        from pymilvus import connections, utility

        self.name = collection_name
        self.embeddings_col = None
        self.text_col = "text"

        self.schema_map = {
            str: DataType.VARCHAR,
            int: DataType.INT64,
            float: DataType.FLOAT,
        }

        connections.connect(alias=alias, host=host, post=port)

        self._table_intition = None

        if utility.has_collection(self.name):
            self.schema = self.collection.schema

            for f in self.schema.fields:
                if f.dtype == 101:
                    self.embeddings_col = f.name

        else:
            self.schema = None

        # Parameters from langchain
        self.index_params = {
            "IVF_FLAT": {"params": {"nprobe": 10}},
            "IVF_SQ8": {"params": {"nprobe": 10}},
            "IVF_PQ": {"params": {"nprobe": 10}},
            "HNSW": {"params": {"ef": 10}},
            "RHNSW_FLAT": {"params": {"ef": 10}},
            "RHNSW_SQ": {"params": {"ef": 10}},
            "RHNSW_PQ": {"params": {"ef": 10}},
            "IVF_HNSW": {"params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"params": {"search_k": 10}},
        }

    def load_collection(
            self,
            df: Union[pd.DataFrame, List[str]],
            embeddings_col="embeddings",
            text_col="text",
            drop=False,
    ):

        df = self._maybe_run_embeddings(
            df, embeddings_col=embeddings_col, text_col=text_col
        )

        dq_coll = self.make_collection(df, drop=drop)

        dq_coll.release()

        cols = df.columns.tolist()

        batched_terms = [
            pd.DataFrame(e) for e in chunked(df.to_dict(orient="records"), 500)
        ]

        for batch in tqdm(batched_terms):
            try:
                dq_coll.insert(batch[cols])
            except:
                print(batch[cols])
                raise

        self.build_index()

        return dq_coll

    def _maybe_run_embeddings(
            self,
            df: Union[pd.DataFrame, List[str]],
            embeddings_col="embeddings",
            text_col="text",
    ):

        ec = self._get_embedding_col(df)

        if ec is None:

            if isinstance(df, list):
                df = pd.DataFrame(df, columns=[text_col])

            df = run_embeddings(df, text_col=text_col, embeddings_col=embeddings_col)

        return df

    def _get_embedding_col(
            self,
            df: Union[pd.DataFrame, List[str]],
    ) -> str:

        from tableintuit import intuit_df

        if isinstance(df, list):
            return None

        t = intuit_df(df.sample(10 if len(df) > 10 else len(df)))

        try:
            embeddings_col = [
                col.header
                for col in t.columns.values()
                if col.resolved_type == np.ndarray
            ][0]
            return embeddings_col
        except IndexError:
            return None

    def make_schema(self, df: pd.DataFrame):
        from pymilvus import CollectionSchema, DataType, FieldSchema

        self.embeddings_col = self._get_embedding_col(df)

        if self.embeddings_col is None:
            raise ValueError("No embeddings column found in dataframe")

        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True)
        ]

        t = intuit_df(df.sample(1000 if len(df) > 1000 else len(df)))

        for col in t.columns.values():
            if col.header == self.embeddings_col:
                fields.append(
                    FieldSchema(
                        name=col.header,
                        dtype=DataType.FLOAT_VECTOR,
                        dim=df[self.embeddings_col].iloc[0].shape[0],
                    )
                )
            elif col.header == "pk":
                pass  # Already set first, and it  usually the index anyway.

            elif col.resolved_type == str:

                max_len = int(df[col.header].str.len().max() + 1)

                fields.append(
                    FieldSchema(
                        name=col.header, dtype=self.schema_map[str], max_length=max_len
                    )
                )

            else:
                fields.append(
                    FieldSchema(
                        name=col.header,
                        dtype=self.schema_map.get(
                            col.resolved_type, self.schema_map[str]
                        ),
                    )
                )

        schema = CollectionSchema(fields, "data question schema")

        return schema

    def make_collection(self, df: pd.DataFrame, drop=False):
        from pymilvus import Collection, utility

        if drop:
            self.drop_collection()

        if not utility.has_collection(self.name):
            logger.info(f"Creating collection {self.name}")
            self.schema = self.make_schema(df)

            dq_coll = Collection(self.name, self.schema, consistency_level="Strong")

        else:
            logger.info(f"Collection {self.name} already exists")
            dq_coll = Collection(self.name)

        return dq_coll

    @property
    def collection(self):
        from pymilvus import Collection

        return Collection(self.name)

    def drop_collection(self):
        from pymilvus import utility

        logger.info(f"Dropping collection {self.name}")
        utility.drop_collection(self.name)

    def build_index(self):
        from pymilvus import Collection

        assert self.embeddings_col is not None, "No embeddings column specified"

        dq_coll = Collection(self.name)

        dq_coll.flush()

        dq_coll.drop_index()

        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 100},
        }

        dq_coll.create_index(self.embeddings_col, index)

    def drop_index(self):
        from pymilvus import Collection

        dq_coll = Collection(self.name)
        dq_coll.release()
        dq_coll.drop_index()

    def _vector_query(self, vectors, expr=None, limit=20):

        ec = self.embeddings_col

        output_fields = [f.name for f in self.schema.fields if f.name not in [ec, "pk"]]

        self.collection.load()

        search_params = {
            "metric_type": "IP",
            "params": {
                "nprobe": 10,
            },
        }

        result = self.collection.search(
            vectors,
            ec,
            search_params,
            limit=limit,
            expr=expr,
            output_fields=output_fields,
        )

        return result

    def vector_query(self, qdf, expr=None, limit=20, progress=False):

        n = 10000  # chunk row size

        if len(qdf) > n:
            # If the query frame is tool large, break it into chunks
            # and run the query on each chunk.
            chunks = [qdf[i: i + n] for i in range(0, qdf.shape[0], n)]

            frames = [
                self.vector_query(c, expr=expr, limit=limit, progress=progress)
                for c in chunks
            ]

            return pd.concat(frames)

        else:

            if not isinstance(qdf, pd.DataFrame):
                qdf = pd.DataFrame({"text": qdf})

            if "embeddings" in list(qdf.columns):
                edf = qdf
            else:
                edf = run_embeddings(
                    qdf,
                    text_col=self.text_col,
                    embeddings_col=self.embeddings_col,
                    progress=progress,
                )

            result = self._vector_query(
                np.stack(edf[self.embeddings_col]), expr=expr, limit=limit
            )

            rows = []
            for hits in result:

                for hit in hits:
                    d = {
                        "score": hit.score,
                    }
                    d.update({e: hit.entity.get(e) for e in hit.entity.fields})

                    rows.append(d)

            return pd.DataFrame(rows)
