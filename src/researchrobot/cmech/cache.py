from dataclasses import dataclass

from researchrobot.objectstore import RedisObjectStore, S3ObjectStore


# A dataclass for the return values from get_classification_queues
@dataclass
class ClassificationQueues:

    redis: RedisObjectStore  # The redis object store

    sources: S3ObjectStore  # All sources
    profiles: S3ObjectStore  # Selected profiles

    rce: S3ObjectStore
    chunks: S3ObjectStore
    parts: S3ObjectStore

    tasks: RedisObjectStore
    inprocess: RedisObjectStore
    complete: RedisObjectStore

    stats: RedisObjectStore
    log: RedisObjectStore


# Collect references to all of the queues we use for classification tasks
def get_classification_queues(rc, version=2):

    sources = rc.sub("sources")
    profiles = rc.sub("profiles")

    rce = rc.sub(f"exp_v{version}")

    chunks = rce.sub("chunks")
    parts = rce.sub("parts")

    # Redis Queues
    redis_os = rce.sub(name="barker_redis")
    tasks = redis_os.set("tasks")
    complete = redis_os.set("complete")
    inprocess = redis_os.set("in_process")

    stats = redis_os.sub("stats")

    log = redis_os.queue("log", max_length=1000)

    return ClassificationQueues(
        redis_os, sources, profiles, rce, chunks, parts, tasks, inprocess, complete, stats, log
    )
