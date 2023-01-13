import openai
from more_itertools import always_iterable, chunked
from tqdm.auto import tqdm

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


def build_terms_map(df):
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
                terms.append([gn, k, v, None])

    # The tables datadrame has these values, but not the columns
    try:

        for v in df.bare_title.unique():
            terms.append(["title", v, v, None])

        for v in df.universe.unique():
            terms.append(["universe", v, v, None])
    except AttributeError:
        pass

    return terms


def get_embeddings(texts, add_spaces=True):
    texts = list(always_iterable(texts))

    if add_spaces:
        texts = [" " + text.strip() + " " for text in texts]

    embedding = openai.Embedding.create(input=texts, engine="text-embedding-ada-002")

    ebd = [(t, e["embedding"]) for t, e in zip(texts, embedding["data"])]

    return ebd


# Re-integrate the embeddings with the terms, and combine the
# batches back into a single set.
def embed_terms(results):
    terms_embd = []

    n = 0
    for r in results:
        assert len(r[0]) == len(r[2])

        for term, embed in zip(r[0], r[2]):
            term[-1] = embed[1]
            terms_embd.append([n] + term)
            n += 1

    return terms_embd


def run_embeddings(terms):
    batched_terms = list(chunked(terms, 200))

    results = []

    for batch in tqdm(batched_terms):
        texts = [e[1] for e in batch]
        ebd = get_embeddings(texts)

        results.append((batch, texts, ebd))

    et = embed_terms(results)

    return et


def make_vdb_collection(terms_embd):
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

    collection_name = "dataquest"

    connections.connect("default", host="localhost", port="19530")

    utility.drop_collection(collection_name)

    if not utility.has_collection(collection_name):

        entities = list(zip(*terms_embd))

        group_len = max([len(e) for e in entities[1]]) + 1
        key_len = max([len(e) for e in entities[2]]) + 1
        value_len = max([len(e) for e in entities[3]]) + 1
        embedding_dim = max([len(e) for e in entities[4]])

        print("Lengths", group_len, key_len, value_len, embedding_dim)

        fields = [
            FieldSchema(
                name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False
            ),
            FieldSchema(name="group", dtype=DataType.VARCHAR, max_length=group_len),
            FieldSchema(name="key", dtype=DataType.VARCHAR, max_length=key_len),
            FieldSchema(name="value", dtype=DataType.VARCHAR, max_length=value_len),
            FieldSchema(
                name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim
            ),
        ]

        schema = CollectionSchema(fields, "data question schema")

        dq_coll = Collection(collection_name, schema, consistency_level="Strong")
        print("Created")

    else:
        dq_coll = Collection(collection_name)
        print("Loaded")

    return dq_coll


def load_vdb_collection(dq_coll, terms_embd):
    for batch in tqdm(list(chunked(terms_embd, 500))):
        entities = list(zip(*batch))
        dq_coll.insert(entities)
        # print(insert_result)


def make_vdb_index(dq_coll):
    dq_coll.flush()

    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }

    dq_coll.create_index("embeddings", index)
