""" Process profiles, and sets of profiles, into dataframes of experiences"""
import logging
from typing import Union
import asyncio
import pandas as pd
from tqdm.auto import tqdm

from researchrobot import ObjectStore
from researchrobot.cmech.cache import get_classification_queues
from researchrobot.embeddings import EmbeddingsAsync

logger = logging.getLogger(__name__)

class Experience:
    """A class for extracting experiences from profiles, embedding them, and creating dataframes for
    later processing"""

    @classmethod
    def main(cls, rc_config: Union[dict, ObjectStore], limit=None, progress=True):
        """Run the top loop of the process"""
        from random import choices

        if isinstance(rc_config, dict):
            rc = ObjectStore(**rc_config)
        else:
            rc = rc_config

        q = get_classification_queues(rc, version=3)

        prof_keys = list(q.profiles.list())
        embd_keys = list(q.parts.sub('exp_embeddings').list())

        logger.info(f'Found {len(prof_keys)} profiles and {len(embd_keys)} existing embeddings')

        remain_keys = list(set(prof_keys) - set(embd_keys))

        logger.info(f'Found {len(remain_keys)} unprocessed profiles')

        if limit:
            remain_keys = choices(remain_keys, k=min(len(remain_keys), limit))
            logger.info(f'Limiting to {len(remain_keys)} profiles')

        for pk in tqdm(remain_keys, desc='Embedding'):
            exp = Experience(rc, pk, n_processes=2, progress=progress)
            if exp.embed_exists:
                continue
            df = asyncio.run(exp.run())
            exp.write()




    def __init__(self, rc_config: Union[dict, ObjectStore], part_name: str,
                 n_processes: int = 2, progress=False):

        self.processes = n_processes
        self.part_name = part_name
        self.progress = progress

        if isinstance(rc_config, dict):
            rc = ObjectStore(**rc_config)
        else:
            rc = rc_config

        self.q = get_classification_queues(rc, version=3)

        self.exp_key = f'exp_embeddings/{self.part_name}'

        self.emb_df = None

        self.errors = None  # Last round of errors from the embeddings
        self.completed_chunks = None  # Last round of completed chunks from the embeddings

    @property
    def embed_exists(self):
        """Check if the embeddings already exist"""

        return self.q.parts.exists(self.exp_key)

    @property
    def profile(self):
        """Get the profile"""

        return self.q.profiles[self.part_name]

    def make_text(self, exp):
        """Add the text for embedding from an experience"""

        title = exp['title'] if exp['title'] else ''
        role = exp['role'] if exp.get('role') else ''

        if title and role:
            title = f'{title}, {role}'
        elif role:
            title = role

        summary = exp['summary'] if exp['summary'] else ''

        return title + '\n' + summary

    def get_profiles(self):
        return self.q.profiles[self.part_name]

    @property
    def exp_df(self):

        rows = []
        for prof in self.get_profiles():
            for exp_id, exp in enumerate(reversed(prof.get('experience', []))):
                exp['text'] = self.make_text(exp)
                exp['exp_id'] = exp_id
                exp['source'] = self.part_name
                rows.append(exp)

        t = pd.DataFrame(rows)

        return t[['source', 'pid', 'exp_id', 'id', 'date',
                  'title', 'role', 'name', 'web', 'company_size', 'summary']]

    @property
    def expt_df(self):
        """Get the experience dataframe, with the text column for embedding"""
        df = self.exp_df.fillna('')
        df['text'] = [self.make_text(r) for idx, r in df.iterrows()]

        return df

    async def run(self):
        """Run the embedding for a profile's experiences"""

        df = self.expt_df

        ea = EmbeddingsAsync(df, chunk_size=200, text_col='text',
                             concurrency=self.processes, progress=self.progress)

        # If there were errors from the prior run, run it again. The successful
        # chunks will no-op
        for i in range(4):
            await ea.run()
            self.errors = ea.errors
            self.completed_chunks = ea.completed_chunks
            if len(ea.errors) == 0:
                break
        else:
            raise Exception(f'Failed to embed experiences for {self.part_name}; {len(ea.errors)} errors')

        t = pd.concat(ea.completed_chunks.values()).sort_index()

        self.emb_df = t[['source', 'exp_id', 'pid', 'id', 'name', 'web', 'company_size', 'title',
                         'role', 'date', 'summary', 'text', 'embeddings']]

        # return the source, with the column order of the original round of embeddings
        return self.emb_df

    def write(self):
        """Write the experience dataframe to the queue"""
        assert self.emb_df is not None, 'Must run first'

        self.q.parts[self.exp_key] = self.emb_df

        return self.exp_key
