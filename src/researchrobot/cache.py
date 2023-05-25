"""Functions for caching data, particularly responses from the OpenAI API"""

import io
import json
import os
import pickle
from pathlib import Path, PosixPath

import redis
from minio import Minio
from minio.error import S3Error

from .config import config

minio = Minio(
    config["MINIO_URL"],
    access_key=config["MINIO_ACCESS_KEY"],
    secret_key=config["MINIO_SECRET_KEY"],
    secure=False,
)

pool = redis.ConnectionPool(
    host=config["REDIS_HOST"], port=config["REDIS_PORT"], db=config["REDIS_DB"]
)
redis = redis.StrictRedis(decode_responses=True, connection_pool=pool)


def _put_iobytes(o, bucket, path, size, content_type="application/octet-stream"):

    try:
        return minio.put_object(bucket, path, o, size, content_type=content_type)
    except S3Error as e:
        if not minio.bucket_exists(bucket):
            minio.make_bucket(bucket)
            return minio.put_object(bucket, path, o, size, content_type=content_type)
        else:
            raise e


def _put_bytes(b, bucket, path, size, content_type="application/octet-stream"):

    return _put_iobytes(io.BytesIO(b), bucket, path, size, content_type=content_type)


def put_object(o, bucket, path=None):

    if path is None:
        bucket, path = bucket.split("/", 1)

    if isinstance(o, PosixPath):
        # Put data from a file
        b = o.read_bytes()
        return _put_bytes(
            b, bucket, path, len(b), content_type="application/octet-stream"
        )

    elif isinstance(o, str):
        # A normal string, so encode it
        size = len(o)
        b = o.encode("utf8")
        return _put_bytes(
            b, bucket, path, size, content_type="text/plain; charset=utf-8"
        )

    elif isinstance(o, bytes):
        return _put_bytes(
            o, bucket, path, len(o), content_type="application/octet-stream"
        )

    elif hasattr(o, "read"):
        size = o.getbuffer().nbytes
        return _put_iobytes(
            o, bucket, path, size, content_type="application/octet-stream"
        )

    elif isinstance(o, object):
        try:
            o = json.dumps(o).encode("utf8")
            size = len(o)
            return _put_bytes(o, bucket, path, size, content_type="application/json")

        except TypeError as e:  # Probably can't be serialized with JSON

            o = pickle.dumps(o)
            size = len(o)
            o = io.BytesIO(o)
            return _put_iobytes(
                o,
                bucket,
                path + ".pickle",
                size,
                content_type="application/octet-stream",
            )

    else:
        raise IOError("Can't understand how to use object")


def get_object(bucket, path=None):
    if path is None:
        bucket, path = bucket.split("/", 1)

    try:
        r = minio.get_object(bucket, path)

        if r.getheader("Content-Type") == "application/octet-stream":
            return r.read()
        elif r.getheader("Content-Type") == "text/plain; charset=utf-8":
            return r.read().decode("utf8")
        elif r.getheader("Content-Type") == "application/json":
            import json

            return json.loads(r.read().decode("utf8"))
        else:
            raise IOError(f"Can't understand response for get of {bucket}/{path}")
    except S3Error as e:
        if e.code == "NoSuchKey":

            # Check if the object is a pickle
            try:
                stat_object(bucket, path + ".pickle")
                o = get_object(bucket, path + ".pickle")
                return pickle.loads(o)
            except S3Error:
                pass

            raise FileNotFoundError(f"No such key bucket ={bucket} path={path}")
        else:
            raise e


def stat_object(bucket, path=None):
    if path is None:
        bucket, path = bucket.split("/", 1)

    return minio.stat_object(bucket, path)


def list_objects(bucket, prefix=None, recursive=True):
    if "/" in bucket:
        bucket, prefix = bucket.split("/", 1)

    r = minio.list_objects(bucket, prefix=prefix, recursive=recursive)

    for o in r:
        fn = f"{o.bucket_name}/{o.object_name}"

        yield (fn, o)


def join_path(p, n=""):
    _p = p.strip("/").split("/") if p else []
    if n:
        if isinstance(n, str):
            _p.extend(n.strip("/").split("/"))
        else:
            _p.extend([e.strip("/") for e in n if e and e.strip("/")])

    return "/".join(_p).strip("/")


class RobotCache:
    def __init__(self, bucket, prefix=None):

        if "/" in bucket and prefix is None:
            self.bucket, prefix = bucket.split("/", 1)
        else:
            self.bucket = bucket

        self._prefix = join_path(prefix)

        if not minio.bucket_exists(bucket):
            minio.make_bucket(bucket)

    def __getitem__(self, key):
        return get_object(self.bucket, self.nbprefix(key))

    def __setitem__(self, key, value):
        return put_object(value, self.bucket, self.nbprefix(key))

    def __iter__(self):
        """Iterate over all keys in the cache, recursively"""
        yield from self.list(recursive=True)

    def delete(self, *v):
        return minio.remove_object(self.bucket, self.nbprefix(*v))

    def list(self, prefix=None, recursive=False):
        """Iterate over all keys in the cache, default is non-recursive"""

        for fn, o in list_objects(
            self.bucket, prefix=self.nbprefix(prefix) + "/", recursive=recursive
        ):
            yield fn.replace(self.prefix(""), "").strip("/"), o

    def sub(self, prefix):
        """Create a new robot with a sub-prefix"""
        return RobotCache(self.bucket, self.nbprefix(prefix))

    def prefix(self, *v):
        return self.bucket + "/" + join_path(self._prefix, v)

    def nbprefix(self, *v):
        return join_path(self._prefix, v)

    def exists(self, *v):
        try:
            self.stat(*v)
            return True
        except S3Error:
            return False

    def stat(self, *v):

        return stat_object(self.bucket, self.nbprefix(*v))

    @property
    def redis(self):
        return redis

    def set_key(self, key, value):
        return self.redis.set(self.nbprefix(key), value)

    def get_key(self, key):
        return self.redis.get(self.prefix(key))

    def keys(self):
        for e in self.redis.scan_iter(self.prefix("*")):
            yield e.decode("utf8").replace(self.prefix(""), "")
