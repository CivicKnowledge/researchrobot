import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import openai
import pandas as pd
from more_itertools import always_iterable, chunked
from pymilvus import DataType
from tableintuit import intuit_df
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

from researchrobot.openai.census_demo_conditions import (
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


from openai.embeddings_utils import cosine_similarity


def get_embeddings(texts, add_spaces=True, normalize=True):
    texts = list(always_iterable(texts))

    if add_spaces:
        texts = [" " + text.strip() + " " for text in texts]

    embedding = openai.Embedding.create(input=texts, engine="text-embedding-ada-002")

    if normalize:
        normalize = lambda x: x / np.linalg.norm(x, ord=2)
    else:
        normalize = lambda x: x

    ebd = [(t, normalize(e["embedding"])) for t, e in zip(texts, embedding["data"])]

    return ebd


def run_embeddings(
    terms: pd.DataFrame,
    text_col: str,
    embeddings_col="embeddings",
    extant_terms=None,
    normalize=True,
):
    from itertools import chain

    from joblib import Parallel, delayed

    batched_terms = [
        pd.DataFrame(e) for e in chunked(terms.to_dict(orient="records"), 200)
    ]

    # batched_terms = [terms[i:i + 100] for i in range(0, len(terms), 100)]

    def _f(batch):
        texts = [r[text_col] for idx, r in batch.iterrows()]
        return get_embeddings(texts, normalize=normalize)

    results = Parallel(n_jobs=4)(delayed(_f)(batch) for batch in tqdm(batched_terms))

    keys, embeds = list(zip(*list(chain(*results))))
    terms = terms.copy()
    terms[embeddings_col] = [np.array(e) for e in embeds]

    return terms


from langchain.vectorstores import Milvus


class EmbedDb:
    schema_map = {
        str: DataType.VARCHAR,
        int: DataType.INT64,
        float: DataType.FLOAT,
    }

    def __init__(self, collection_name: str):
        from pymilvus import connections, utility

        self.name = collection_name
        self.embeddings_col = None

        connections.connect("default", host="localhost", port="19530")

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
            dq_coll.insert(batch[cols])

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

                max_len = df[col.header].str.len().max() + 1

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

    def _vector_query(self, terms, add_spaces=True, expr=None, limit=20):

        if isinstance(terms, str):
            terms = terms.split(",")

        if add_spaces:
            terms = [" " + term.strip().lower() + " " for term in terms]

        vectors = [e[1] for e in get_embeddings(terms)]

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

    def vector_query(self, terms, add_spaces=True, expr=None, limit=20):

        result = self._vector_query(terms, add_spaces, expr=expr, limit=limit)

        hits = []
        for term, term_hits in zip(terms, result):

            for hit in term_hits:
                d = {
                    "query": term,
                    "score": hit.score,
                }
                d.update({e: hit.entity.get(e) for e in hit.entity.fields})

                hits.append(d)

        hits_df = pd.DataFrame(hits)

        return hits_df
