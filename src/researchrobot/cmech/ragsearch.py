import json
import pandas as pd
from functools import cache
from tabulate import tabulate
from researchrobot.cmech.classify import resolve_cache
import logging

from researchrobot.openai.completions import openai_one_completion

logger = logging.getLogger(__name__)


def make_example():
    """Generate an example for the system prompt"""

    ex_pid = 'PzaMlWLnVEV-wsTR2Tsllg_0000'

    ex = exp_df[exp_df.pid == ex_pid]

    rows = []
    for idx, r in ex.iterrows():
        t = sec.job_search(r.title, r.summary)

        soc = t.head(3).sample(3).detailed.to_list()
        d = {
            "reasoning": "Financial advisor is very close to financial representative",
            "soc": json.dumps(soc),
            "quality": "fair"
        }
        rows.append(d)

    out = pd.DataFrame(rows, index=ex.index)
    t = pd.concat([ex, out], axis=1).drop(columns=['id', 'name', 'web', 'company_size', 'text', 'role', 'embeddings'])
    t.to_csv('example.csv')


class RAGJobSearch:

    # noinspection PyShadowingNames
    def __init__(self, cache, sec, data, rag_style='plain',
                 example_file='example.csv'):
        self.cache = resolve_cache(cache)
        self.data = data
        self.sec = sec
        self.rag_style = rag_style

        self.example_file = example_file

    @staticmethod
    def request_key(self, pid):
        return f"request/{pid}"

    @staticmethod
    def response_key(self, pid):
        return f"response/{pid}"

    @property
    def complete_pids(self):
        """PIDs that have been requested and have a response"""
        return list(self.cache.sub('response').list())

    @property
    def incomplete_pids(self):
        """PIDS that have a request and no response"""
        return list(set(self.cache.sub('request').list()) - set(self.complete_pids))

    def search_experience(self, exp):

        r = self.sec.job_search(exp.title, exp.summary)

        return r

    def search_profile(self, profile):

        r = self.sec.job_search(profile.title, profile.summary)

        return r

    @staticmethod
    def decode_response(r):
        records = []

        for e in r.splitlines():
            try:
                j = json.loads(e)
                records.append(j)
            except json.JSONDecodeError:
                continue

        if len(records) == 1 and isinstance(records[0], list):
            # We got a single-line JSON list, instead of json lines
            return records[0]
        else:
            return records

    def unpack_response(self, r):

        d = {}

        recs = RAGJobSearch.decode_response(r)

        for j in recs:
            j['soc'] = [(k, self.data.title_map.get(k)) for k in j['soc']]
            exp_id = int(j['job'].replace('jc_', ''))
            del j['job']
            d[exp_id] = j

        return d

    @staticmethod
    def response_blocks(r):

        blocks = {}

        for exp_id, e in r.items():
            o = f"Job Title Matches\n-----------------\n\nMatch quality: {e['quality']}; {e['reasoning']}\n"
            for soc, title in e['soc']:
                o += f"  {soc} {title}\n"

            blocks[exp_id] = o

        return blocks

    def format_search_results(self, results_df, head=7):

        t = results_df.head(head)

        if self.rag_style == 'table':
            table = tabulate(t, headers=t.columns,
                             maxcolwidths=[10, 10, 10, 20, 40, 10, 10, 10, 10, 10],
                             tablefmt="grid")
        elif self.rag_style == 'plain':
            t = t[['detailed', 'title', 'text']].rename(columns={'detailed': 'soc'})
            table = tabulate(t, headers=t.columns, maxcolwidths=[10, 10, 40, 100], tablefmt="plain")
        elif self.rag_style == 'list':
            table = 'Candidate SOC codes:\n\n' + \
                    ''.join([f"  * {r.detailed} {r.title}: {r.text}\n" for idx, r in t.iterrows()])
        else:
            table = t.to_json(orient='records')

        return table

    def format_exp(self, exp, search_results, head=7):
        """Format a single job experience and its search results"""
        return f"Job code: jc_{exp.exp_id}\nTitle: {exp.title}\nDescription:\n{exp.summary}\n\n" + \
            "Candidate job matches:\n\n" + \
            self.format_search_results(search_results, head=head) + '\n'

    @cache
    def compose_examples(self):

        ex = pd.read_csv(self.example_file)

        resp = []

        out = []

        for idx, r in ex.iterrows():
            t = self.sec.job_search(r.title, r.summary).head(5)

            out.append(self.format_exp(r, t))

            d = {
                'job': "jc_" + str(r.exp_id),
                'reasoning': r.reasoning,
                'soc': json.loads(r.soc),
                'quality': r.quality
            }

            resp.append(json.dumps(d))

        return out, resp

    def system_prompt(self):

        ex, resp = self.compose_examples()
        nl = "\n"

        jsonl_inst = "Your output will be in JSON Lines format, with one complete JSON document " \
                     "per line. Each JSON document will include these keys and associated values:"

        return f"""
 You are a Human Resources Specialist whose job is to classify the
 experiences on candidates resumes according to their formal O*NET SOC
 codes. You will be given multiple job experiences and a set of
 possible SOC codes. You will select the 1 to 3 codes that are the best
 matches for the job description. 

 For each job experience you will get the job id, title, and
 description and a table of matches, in JSON format, that includes
 the possible SOC code, title, and a search engine score for the
 title and description. THe table also includes scores that indicate
 how good the match is for the job title and body. 

    {jsonl_inst} 

    * 'job': the job code linking this answer to the input
    * 'reasoning': A text string with your thought process about selecting the SOC
    * 'soc': A list of up to 3 SOC codes that are a good match for the job.
    * 'quality': A quality assessment of how well the matches SOC codes represent 
       the job. The quality assessment is string, one of "good", "fair" or "poor".

    If there are no good matches, return a JSON document with the job code and an empty list of SOC codes

    Example
    =======

    Input
    -----

    {nl.join(ex[:2])}

    Output
    ------

    {'[' + ','.join(resp[:2]) + ']'}

    Guidelines
    ==========

    1. Prefer SOC codes with higher score values to ones with lower score values. 
    2. Use your best judgment
    3. Be sure your JSON is properly formatted
    4. Return your response in JSON lines format, with one JSON dict per line 
       for each job in the input
    5. All of the jobs are for a single person's career, in chronological order,
       so when choosing a job title, consider the
    continuity with the job titles before it. 
    6. Weight the title higher than the body text
    7. Prefer the higher level title. If the choices are "vice president" and 
       "human resources consultant", prefer to select a higher level executive role
    8. Prefer more common titles over less common titles. Rare jobs like 
       "Brownfield manager" and Biofuels jobs are almost never the correct choice.

    """

    # noinspection PyMethodMayBeStatic
    def prompt(self, candidates):

        nl = '\n'

        return f"""Please determine the SOC codes for the following jobs  and return you classification 
        in JSON lines format, one JSON document per line.

    {(nl + nl).join(candidates)}

    JSON Lines Output
    =================

    """

    def prepare_completion(self, education, experiences, fast_return=True, force=False):

        pids = experiences.pid.unique()

        assert len(pids) == 1, "Only one person's experiences can be processed at a time"

        pid = pids[0]

        key = self.request_key(pid)

        if key in self.cache and force is False:
            if fast_return:
                return key, None
            else:
                return key, self.cache[key]

        experiences = [exp for idx, exp in experiences.iterrows()]

        searches = [self.sec.job_search(exp.title, exp.summary) for exp in experiences]

        candidates = [self.format_exp(exp, search) for exp, search in zip(experiences, searches)]

        prompt = self.prompt(candidates)

        d = {
            "pid": pid,
            "education": education,
            "experiences": experiences,
            "searches": searches,
            "candidates": candidates,
            'prompt': prompt,
        }

        self.cache[key] = d

        return key, d

    def run_prepared_completion(self, request_key):

        _, pid = request_key.split('/')

        key = f"response/{pid}"

        if key in self.cache:
            return key

        d = self.cache[request_key]

        self.cache[key] = openai_one_completion(d['prompt'], system=self.system_prompt(),
                                                model='gpt-3.5-turbo-16k', json_mode=False,
                                                return_response=True, max_tokens=600)

        return key

    # noinspection PyUnusedLocal
    def run_completion(self, education, experiences, force=False):

        request_key, d = self.prepare_completion(education, experiences, fast_return=False, force=force)

        return self.run_prepared_completion(request_key)


    def run_experiences(self, experiences):
        """Run multiple experiences"""
        from tqdm.auto import tqdm

        for gn, g in tqdm(experiences.groupby('pid')):
            self.run_completion(None, g)




