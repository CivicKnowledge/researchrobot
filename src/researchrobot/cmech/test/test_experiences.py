import unittest
import logging
import asyncio

from researchrobot.cmech.experiences import Experience, logger
logging.basicConfig(level=logging.FATAL)
logger.setLevel(level=logging.INFO)

class TestExperiences(unittest.TestCase):

    def test_run_experiences(self):

        rc_config = dict(name='barker_minio', bucket='linkedin')

        Experience.main(rc_config, limit=10)

if __name__ == '__main__':
    unittest.main()
