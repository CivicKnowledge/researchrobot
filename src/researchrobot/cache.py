"""A Robotic Multi-Cache. THe main interface is to minio, but
also has a redis cache for fast access to metadata, and links to Mongodb
for documents. """

import io
import json
import os
import pickle
from pathlib import Path, PosixPath

import redis
from langchain.document_loaders import PyPDFLoader
from minio import Minio
from minio.error import S3Error

from .config import config as base_config

# Type codes for various datatypes and content types
type_codes = {
    str: b"s",
    bytes: b"b",
    int: b"i",
    float: b"f",
    bool: b"B",
    "json": b"J",
    "pickle": b"P",
    PosixPath: b"p",
    "io": b"I",
}


def _to_bytes(o):
    """Convert an object to bytes, and return:
    - the type code
    - the bytes
    - the size
    - the content type
    - the extension


    This function will convert a range of input objects, including
    strings, bytes, integers, floats, booleans, lists, tuples, dicts,
    sets and objects. Scalars are converted to strings. Objects are serialized
    to JSON, if possible, or pickled if not. If the object is a Path or has a read()
    method, the contents of the file are returned.
    """

    if isinstance(o, PosixPath):
        # Put data from a file
        b = o.read_bytes()
        return type_codes[PosixPath], b, len(b), "application/octet-stream", ""

    elif isinstance(o, str):
        # A normal string, so encode it
        size = len(o)
        b = o.encode("utf8")
        return type_codes[str], b, size, "text/plain; charset=utf-8", ""

    elif isinstance(o, bytes):
        return type_codes[bytes], o, len(o), "application/octet-stream", ""

    elif hasattr(o, "read"):
        size = o.getbuffer().nbytes
        return type_codes["io"], o, size, "application/octet-stream", ""

    elif isinstance(o, object):
        try:
            o = json.dumps(o).encode("utf8")
            size = len(o)
            return type_codes["json"], o, size, "application/json", ""

        except TypeError as e:  # Probably can't be serialized with JSON

            o = pickle.dumps(o)
            size = len(o)
            return type_codes["pickle"], o, size, "application/x-pickle", ""

    else:
        raise IOError("Can't understand how to use object")


class RedisCache:
    """Access Redis using a root key prefix from a Robot Cache object. The Redis
    key will include both the bucket and the prefix from the RobotCache
    """

    def __init__(self, cache: "RobotCache"):
        self.cache = cache
        self.redis = self.cache.redis
        self.bucket = self.cache.bucket

    def prefix(self, *v):
        return self.bucket + "/" + self.cache.prefix(*v)

    def keys(self):
        for e in self.redis.scan_iter(self.prefix("*")):
            yield e.decode("utf8").replace(self.prefix(""), "")

    type_codes = type_codes

    def to_bytes(self, o):
        """Convert an object to bytes, and return:
        - the type code
        - the bytes

        This function will convert a range of input objects, including
        strings, bytes, integers, floats, booleans, lists, tuples, dicts,
        sets and objects. Scalars are converted to strings. Objects are serialized
        to JSON, if possible, or pickled if not.

        """

        if isinstance(o, str):
            return self.type_codes[str], o.encode("utf8")

        elif isinstance(o, bytes):
            return self.type_codes[bytes], o

        elif isinstance(o, bool):
            return self.type_codes[bool], str(int(o)).encode("utf8")

        elif isinstance(o, int):
            return self.type_codes[int], str(o).encode("utf8")

        elif isinstance(o, float):
            return self.type_codes[float], str(o).encode("utf8")

        elif isinstance(o, (list, tuple, dict, set, object)):
            try:
                return self.type_codes["json"], json.dumps(o).encode("utf8")
            except TypeError as e:  # Probably can't be serialized with JSON
                return self.type_codes["pickle"], pickle.dumps(o)
        else:
            raise IOError(f"Can't understand how to use object {o}")

    def from_bytes(self, o):
        """Convert bytes to an object, using the type code to determine
        the type of the object. The type code is the first byte of the
        input bytes."""

        tc = o[0:1]
        o = o[1:]

        if tc == self.type_codes[str]:
            return o.decode("utf8")
        elif tc == self.type_codes[bytes]:
            return o
        elif tc == self.type_codes[int]:
            return int(o.decode("utf8"))
        elif tc == self.type_codes[float]:
            return float(o.decode("utf8"))
        elif tc == self.type_codes[bool]:
            return bool(int(o.decode("utf8")))
        elif tc == self.type_codes["json"]:
            return json.loads(o.decode("utf8"))
        elif tc == self.type_codes["pickle"]:
            return pickle.loads(o)
        else:
            raise IOError(f"Can't understand how to use object pf type code {tc}")


class RedisKeyValue(RedisCache):
    """Redis key/value accessor. The KV interface uses the Redis
    set() and get() functions."""

    def __getitem__(self, key):
        o = self.redis.get(self.prefix(key))

        if o is None:
            raise KeyError(key)

        return self.from_bytes(o)

    def __setitem__(self, key, value):
        tc, b = self.to_bytes(value)
        return self.redis.set(self.prefix(key), tc + b)

    def exists(self, key):
        return self.redis.exists(self.prefix(key))

    def __contains__(self, key):
        return self.exists(key)

    def __iter__(self):
        yield from self.keys()

    def __delitem__(self, key):
        return self.redis.delete(self.prefix(key))

    def get(self, key):
        """Get the value, without translating encoding"""
        return self.redis.get(self.prefix(key))

    def getint(self, key):
        """Get the value as an int, with no encoding"""
        return int(self.redis.get(self.prefix(key)))

    def setint(self, key, value):
        """set the value as an int, with no encoding"""
        self.redis.set(self.prefix(key), value)

    def incr(self, key, amount=1):
        return self.redis.incr(self.prefix(key), amount)

    def decr(self, key, amount=1):
        return self.redis.decr(self.prefix(key), amount)


class RedisSet(RedisCache):
    """Redis set accessor, which uses the sadd(), srem() and sismember() methods"""

    def add(self, value):
        tc, b = self.to_bytes(value)
        return self.redis.sadd(self.prefix(), tc + b)

    def remove(self, value):
        tc, b = self.to_bytes(value)
        return self.redis.srem(self.prefix(), tc + b)

    def is_member(self, value):
        tc, b = self.to_bytes(value)
        return self.redis.sismember(self.prefix(), tc + b)

    def __len__(self):
        return self.redis.scard(self.prefix())

    def __contains__(self, item):
        return self.is_member(item)

    def __iter__(self):
        for e in self.redis.smembers(self.prefix()):
            yield self.from_bytes(e)

    def delete(self):
        return self.redis.delete(self.prefix())


class RobotCache:
    def __init__(self, bucket: str, prefix: str = None, config: dict = None):

        if "/" in bucket and prefix is None:
            self.bucket, prefix = bucket.split("/", 1)
        else:
            self.bucket = bucket

        self._prefix = RobotCache.join_path(prefix)

        self.config = dict(base_config.items())

        if config is not None:
            self.config.update(config)

        self.minio = Minio(
            self.config["MINIO_URL"],
            access_key=self.config["MINIO_ACCESS_KEY"],
            secret_key=self.config["MINIO_SECRET_KEY"],
            secure=False,
        )

        self._redis = None
        self._mongo_client = None
        self._mongo_db = None

        if not self.minio.bucket_exists(bucket):
            self.minio.make_bucket(bucket)

    def sub(self, *prefix):
        """Create a new robot with a sub-prefix"""
        return RobotCache(self.bucket, self.nbprefix(*prefix))

    @property
    def parent(self):
        """Create a new cace on a parent path"""

        parent_path = "/".join(self.nbprefix().split("/")[:-1])

        return RobotCache(self.bucket, parent_path)

    def prefix(self, *v):
        """Create a key based on the bucket, the current prefix, and any arguments"""
        return self.bucket + "/" + RobotCache.join_path(self._prefix, v)

    def nbprefix(self, *v):
        """Create a key based on the current prefix, and any arguments. Like prefix(),
        but without the bucket name
        """
        return RobotCache.join_path(self._prefix, v)

    def __getitem__(self, key):
        return RobotCache.get_object(self.bucket, self.nbprefix(key))

    def __setitem__(self, key, value):
        return RobotCache.put_object(value, self.bucket, self.nbprefix(key))

    def __iter__(self):
        """Iterate over all keys in the cache, recursively"""
        yield from self.list(recursive=True)

    @staticmethod
    def join_path(p, n=""):
        """Combine the arguments into a valid Minio key"""
        _p = p.strip("/").split("/") if p else []
        if n:
            if isinstance(n, str):
                _p.extend(n.strip("/").split("/"))
            else:
                _p.extend([str(e).strip("/") for e in n if e and str(e).strip("/")])

        return "/".join(_p).strip("/")

    @staticmethod
    def get_object(bucket, path=None):
        """Get an object from the cache for a key

        This function will automatically unserialize JSON and Pickled objects,
        and decode UTF8 strings.
        """

        if path is None:
            bucket, path = bucket.split("/", 1)

        try:
            r = self.minio.get_object(bucket, path)

            if (
                r.getheader("Content-Type") == "application/x-gzip"
                or r.getheader("Content-Encoding") == "gzip"
            ):
                import gzip

                return gzip.decompress(r.read())
            if r.getheader("Content-Type") == "application/octet-stream":
                if path.endswith(".gz"):
                    import gzip

                    return gzip.decompress(r.read())
                else:
                    return r.read()
            elif r.getheader("Content-Type") == "text/plain; charset=utf-8":
                return r.read().decode("utf8")
            elif r.getheader("Content-Type") == "application/json":
                return json.loads(r.read().decode("utf8"))
            elif r.getheader("Content-Type") == "application/x-pickle":
                return pickle.loads(r.read())
            else:
                raise IOError(
                    f"Can't understand response for get of {bucket}/{path}: content-type={r.getheader('Content-Type')}"
                )
        except S3Error as e:
            if e.code == "NoSuchKey":
                raise FileNotFoundError(f"No such key bucket={bucket},  path={path}")
            else:
                raise e

    @staticmethod
    def _put_iobytes(o, bucket, path, size, content_type="application/octet-stream"):
        """Store an object in minio for a bucket and a path. If the bucket does not
        exist, it will be created.
        """

        try:
            return self.minio.put_object(
                bucket, path, o, size, content_type=content_type
            )
        except S3Error as e:
            if not self.minio.bucket_exists(bucket):
                self.minio.make_bucket(bucket)
                return minio.put_object(
                    bucket, path, o, size, content_type=content_type
                )
            else:
                raise e

    @staticmethod
    def _put_bytes(b, bucket, path, size, content_type="application/octet-stream"):
        return RobotCache._put_iobytes(
            io.BytesIO(b), bucket, path, size, content_type=content_type
        )

    @staticmethod
    def put_object(o, bucket, path=None):
        """Store an object in minio for a bucket and a path. If the bucket does not
        exist, it will be created.
        """

        if path is None:
            bucket, path = bucket.split("/", 1)

        if isinstance(o, io.BytesIO):
            size = o.getbuffer().nbytes
            return RobotCache._put_iobytes(
                o, bucket, path, size, content_type="application/octet-stream"
            )
        else:
            tc, b, size, content_type, ext = _to_bytes(o)
            return RobotCache._put_bytes(
                b, bucket, path + ext, size, content_type=content_type
            )

    def delete(self, *v):
        return minio.remove_object(self.bucket, self.nbprefix(*v))

    @staticmethod
    def _list_objects(bucket, prefix=None, recursive=True):
        if "/" in bucket:
            bucket, prefix = bucket.split("/", 1)

        r = minio.list_objects(bucket, prefix=prefix, recursive=recursive)

        for o in r:
            fn = f"{o.bucket_name}/{o.object_name}"

            yield (fn, o)

    def list(self, prefix=None, recursive=True):
        """Iterate over all keys in the cache, default is non-recursive"""

        for fn, o in RobotCache._list_objects(
            self.bucket, prefix=self.nbprefix(prefix) + "/", recursive=recursive
        ):
            key = fn.replace(self.prefix(""), "").strip("/")
            if any(e.startswith("_") for e in key.strip("/").split("/")):
                continue

            yield key, o

    def keys(self, recursive=True):
        """Return a list of all keys in the cache, default is non-recursive,
        like list(), but return only the first item of the tuple"""

        for k, o in self.list(recursive=recursive):
            yield k

    def exists(self, *v):
        """Check if a key exists in the cache"""
        try:
            self.stat(*v)
            return True
        except S3Error:
            return False

    def __contains__(self, item):
        """Allow using the python in operator to check if a key exists"""
        return self.exists(item)

    def stat(self, *v):
        path = self.nbprefix(*v)

        if path is None:
            bucket, path = self.bucket.split("/", 1)
        else:
            bucket = self.bucket

        return minio.stat_object(bucket, path)

    @property
    def redis(self):
        """Return the redis connection for this cache. Use this for general access to the
        Redis API. For Key/Value access use, .kv, and for set access, use .set"""

        import redis

        if not self._redis:

            pool = redis.ConnectionPool(
                host=self.config["REDIS_HOST"],
                port=self.config["REDIS_PORT"],
                db=self.config["REDIS_DB"],
            )
            self._redis = redis.StrictRedis(decode_responses=True, connection_pool=pool)

        return self._redis

    @property
    def mongo(self):
        """Return the MongoDB connection for this cache. Use this for general access to the
        MongoDB API."""

        if not self._mongo_db:
            from pymongo import MongoClient

            self._mongo_client = MongoClient(self.config["MONGO_URL"])
            self._mongo_db = self._mongo_client[self.bucket]

        return self._mongo_db

    def mdb(self, collection: str):
        """Return a MongoDB collection object"""
        return self.mongo[collection]

    @property
    def kv(self):
        """KeyValue interface for this cache"""
        return RedisKeyValue(self)

    @property
    def set(self):
        """Set interface for this cache"""
        return RedisSet(self)

    # I think these are deprecated ..
    # def set_key(self, key, value):
    #     """Deprecated?"""
    #     return self.redis.set(self.nbprefix(key), value)
    #
    # def get_key(self, key):
    #     return self.redis.get(self.prefix(key))
    #
    # def keys(self):
    #     for e in self.redis.scan_iter(self.prefix("*")):
    #         yield e.decode("utf8").replace(self.prefix(""), "")

    def __str__(self):
        return f"RobotCache({self.bucket}, {self._prefix})"

    def __repr__(self):
        return str(self)
