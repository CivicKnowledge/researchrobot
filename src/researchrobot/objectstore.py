""" An abstraction layer for the object store, so we
can access S3 via Boto, but also use a local filesystem
"""

import json
import pickle
import shelve
import sys
from pathlib import Path, PosixPath

from .config import get_config


def oscache(storage):
    """Cache values from a function in `storage`. Not that this does not properly
    handle default argument, which will not be included in the key"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            key = (func.__name__, args, frozenset(kwargs.items()))
            key = "oscache-" + str(key)
            if key not in storage:
                storage[key] = func(*args, **kwargs)
            return storage[key]

        return wrapper

    return decorator


def get_content_type(o):
    import magic

    try:
        return magic.from_file(o, mime=True)
    except ImportError:
        raise ImportError(
            "You probably need to install lib magic: "
            + "https://github.com/ahupp/python-magic#installation"
        )


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
        content_type = get_content_type(o)
        b = o.read_bytes()
        return b, len(b), content_type, ""  # "application/octet-stream", ""

    elif isinstance(o, str):
        # A normal string, so encode it
        size = len(o)
        b = o.encode("utf8")
        return b, size, "text/plain; charset=utf-8", ""

    elif isinstance(o, bytes):
        return o, len(o), "application/octet-stream", ""

    elif hasattr(o, "read"):
        try:
            size = o.getbuffer().nbytes
            return o, size, "application/octet-stream", ""
        except AttributeError:
            # Nope, not a buffer
            return _to_bytes(o.read())

    elif isinstance(o, object):
        try:
            o = json.dumps(o).encode("utf8")
            size = len(o)
            return o, size, "application/json", ""

        except TypeError:  # Probably can't be serialized with JSON

            o = pickle.dumps(o)
            size = len(o)
            return o, size, "application/x-pickle", ""

    else:
        raise IOError("Can't understand how to use object")


def new_object_store(**kwargs):
    """Create a new object store, based on the configuration

    Patterns:

        new_object_store(name='name')

            Get the default configuration and use the configuration for the
            cache named 'name'

        new_object_store(name='name', config=config)

            Load the configuration and look up the configuration for the
            cached named 'name'

        new_object_store(bucket='bucket', prefix='prefix', **config['name'])

            If config is the base configuration, create a new cache using
            that configuration, with a specific bucket and prefix

    """

    if "config_path" in kwargs:
        # Config is provided with a reference to a file
        config = get_config(kwargs["config_path"])
        kwargs.update(config)
    elif "config" in kwargs:
        config = kwargs["config"]
    elif "class_" in kwargs:
        # The user interpolated in a named section of the config,
        # ( ie, ** config['name'] )
        config = None
    else:
        config = get_config()

    if config is not None:

        if "class_" in config:
            # This is one of the named configs, from
            # the named section of the config

            cache_config = config
        else:

            # This should be a top-level config, which has
            # a 'caches' section
            assert "caches" in config, "No cache configuration found"

            if "name" not in kwargs:
                # Specified config, but no name, so use the default
                name = "default"

                assert (
                    name is not None
                ), "No default object store specified and no name provided"

            else:
                name = kwargs["name"]

            try:
                if name == "default":
                    name = config["default"]

                cache_config = config["caches"][name]
            except KeyError:

                raise KeyError(
                    f"No configuration for object store named '{name}'. "
                    + f"keys are: {', '.join(config.get('caches', {}).keys())}"
                )
    else:
        cache_config = dict(**kwargs)

    for n in ["name", "caches", "default", "_class"]:
        if n in kwargs:
            del kwargs[n]

    if "class_" not in cache_config:
        print(cache_config)
        raise KeyError("No `class_` specified for object store")

    clz = getattr(sys.modules[__name__], cache_config["class_"])

    args = {**cache_config, **kwargs}

    return clz(**args)


class ObjectStore(object):
    bucket: str = None
    prefix: str = None
    config: dict = None

    def __init__(self, bucket: str = None, prefix: str = None, **kwargs):
        self.bucket = bucket
        self.prefix = prefix or ""
        self.config = kwargs

        if "_" in bucket:
            raise ValueError(
                f"Bucket name '{bucket}' contains an underscore, which is not allowed"
            )

        assert self.bucket is not None, "No bucket specified"

    def sub(self, *args, extra_kw_args=None, **kwargs):
        """
        Return a new ObjectStore with a sub-prefix

        :param args: Prefix to append the current prefix

        :param kwargs: If provided, create a new object store with the
        same bucket and base prefix, but with the new configuration

        :rtype:
        """

        if kwargs:

            config = kwargs
            if not "class_" in config and not "name" in config:
                config["class_"] = self.__class__.__name__
        else:
            config = self.config

        config["bucket"] = self.bucket
        config["prefix"] = self.join_path(*args)

        if extra_kw_args:
            config.update(extra_kw_args)

        return new_object_store(**config)

    @classmethod
    def new(self, **kwargs):

        return new_object_store(**kwargs)

    def join_path(self, *args):
        args = [self.prefix] + list(args)
        args = [e.strip("/") for e in args if e]
        args = [e for e in args if e]

        return "/".join(args)

    def join_pathb(self, *args):
        return self.bucket + "/" + self.join_path(*args)

    def put(self, key: str, data: bytes):
        raise NotImplementedError

    def __setitem__(self, key, value):
        return self.put(key, value)

    def get(self, key: str) -> bytes:
        raise NotImplementedError

    def __getitem__(self, key):
        return self.get(key)

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def __contains__(self, item):
        """Allow using the python in operator to check if a key exists"""
        return self.exists(item)

    def delete(self, key: str):
        raise NotImplementedError

    def __delitem__(self, key):
        return self.delete(key)

    def list(self, prefix: str, recursive=True) -> list:
        raise NotImplementedError

    def __iter__(self):
        """Iterate over all keys in the cache, recursively"""
        yield from self.list(recursive=True)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.__class__.__name__}({self.bucket}, {self.prefix})"

    def set(self, key: str = "set"):
        return ObjectSet(self, key)


def first_env_var(config, args):
    """Return the first key"""
    for arg in args:
        if arg in config:
            return config[arg]

    return None


class _ObjectStore(ObjectStore):
    """Re-declares new() as an instance method"""

    def new(self, **kwargs):
        """Create a new instance of an object store, re-using the
        bucket and prefix from this one."""

        args = {"bucket": self.bucket, "prefix": self.prefix, **kwargs}
        return new_object_store(**args)


class S3ObjectStore(_ObjectStore):
    def __init__(
        self,
        bucket: str = None,
        prefix: str = None,
        access_key: str = None,
        secret_key: str = None,
        endpoint: str = None,
        region: str = None,
        client=None,
        **kwargs,
    ):

        import boto3

        super().__init__(bucket=bucket, prefix=prefix, **kwargs)

        self.client = None

        self.region = region
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key

        if "/" in bucket:
            bucket, _prefix = bucket.split("/", 1)
            if prefix is None:
                self.prefix = _prefix
            else:
                self.prefix = _prefix + "/" + self.prefix

        config = {}

        if endpoint:
            config["endpoint_url"] = self.endpoint
        if region:
            config["region_name"] = self.region

        if client is None:
            if "profile" in kwargs:
                self.session = boto3.session.Session(profile_name=kwargs["profile"])
            else:
                self.session = boto3.session.Session()

            self.client = self.session.client(
                "s3",
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                **config,
            )
        else:
            self.client = client

        # create_bucket is idempotent, and not more expensive than head_bucket
        # so we can just call it here
        self.create_bucket()

    def sub(self, *args, **kwargs):

        if "name" in kwargs or "class_" in kwargs:
            # We are changing the type, so don't keep the client
            return super().sub(*args, **kwargs)
        else:
            return S3ObjectStore(
                bucket=self.bucket,
                prefix=self.prefix if not args else self.join_path(*args),
                region=self.region,
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                client=self.client,
            )

    def create_bucket(self):

        try:
            self.client.create_bucket(Bucket=self.bucket)

            print("Created bucket", self.bucket)
        except (
            self.client.exceptions.NoSuchBucket,
            self.client.exceptions.ClientError,
        ) as e:
            pass

    def _put_bytes(
        self, key: str, data: bytes, content_type: str = None, metadata: dict = None
    ):

        return self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type or "application/octet-stream",
            # ACL="private",
            # Metadata=metadata or {}
        )

    def put(self, key: str, data, metadata: dict = None):
        if dict is None:
            metadata = {}

        b, size, content_type, ext = _to_bytes(data)

        key = self.join_path(key)

        return self._put_bytes(
            key, data=b, content_type=content_type, metadata=metadata
        )

    def _get_bytes(self, key: str) -> bytes:

        try:
            r = self.client.get_object(Bucket=self.bucket, Key=key)

            return r
        except Exception:
            raise

    def get(self, key: str):

        key = self.join_path(key)

        try:
            r = self._get_bytes(key)

            body = r.get("Body")

            if (
                r.get("ContentType") == "application/x-gzip"
                or r.get("ContentEncoding") == "gzip"
            ):
                import gzip

                return gzip.decompress(r.read())
            if r.get("ContentType") == "application/octet-stream":
                if key.endswith(".gz"):
                    import gzip

                    return gzip.decompress(body.read())
                else:
                    return body.read()
            elif r.get("ContentType") == "text/plain; charset=utf-8":
                return body.read().decode("utf8")
            elif r.get("ContentType") == "text/plain":
                return body.read().decode("ascii")
            elif r.get("ContentType") == "application/json":
                return json.loads(body.read().decode("utf8"))
            elif r.get("ContentType") == "application/x-pickle":
                return pickle.loads(body.read())
            else:
                raise IOError(
                    f"Can't understand response for get of {self.bucket}/{key}: content-type={r.get('ContentType')}"
                )
        except self.client.exceptions.NoSuchKey:
            raise KeyError(f"No such key {key} in bucket {self.bucket}")
        except Exception:
            raise

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self.join_path(key))
            return True
        except Exception:
            return False

    def delete(self, key: str):
        self.client.delete_object(Bucket=self.bucket, Key=self.join_path(key))

    def list(self, prefix: str = "", recursive=True):

        # Create a reusable Paginator
        paginator = self.client.get_paginator("list_objects")

        # Create a PageIterator from the Paginator
        itr = paginator.paginate(Bucket=self.bucket, Prefix=self.join_path(prefix))

        try:
            for page in itr:
                for e in page.get("Contents", []):
                    yield e["Key"].removeprefix(self.prefix).lstrip("/")
        except Exception:
            print("Error listing", self.join_pathb(prefix))
            raise

    def presigned_url(self, key, expiration=60 * 60 * 24 * 7):

        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": self.join_path(key)},
            ExpiresIn=expiration,
        )

    def public_url(self, key):
        from urllib.parse import urlparse

        host = urlparse(self.client.meta.endpoint_url).netloc

        return f"http://{self.bucket}.{host}/{self.join_path(key)}"

    def __str__(self):
        return f"{self.__class__.__name__}({self.bucket}, {self.prefix})"


class LocalObjectStore(_ObjectStore):
    def __init__(
        self, bucket: str = None, prefix: str = None, path: str = None, **kwargs
    ):

        self.bucket = bucket
        self.prefix = prefix or ""

        super().__init__(bucket=bucket, prefix=prefix, path=path, **kwargs)

        path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)

        self.path = str(path / self.bucket)

    def put(self, key: str, data: bytes):
        with shelve.open(self.path) as db:
            db[self.join_path(key)] = data

    def get(self, key: str) -> bytes:
        with shelve.open(self.path) as db:
            return db[self.join_path(key)]

    def exists(self, key: str) -> bool:
        with shelve.open(self.path) as db:
            return self.join_path(key) in db

    def delete(self, key: str):
        with shelve.open(self.path) as db:
            del db[self.join_path(key)]

    def _list(self):
        with shelve.open(self.path) as db:
            for key in db.keys():
                yield key

    def list(self, prefix: str = "", recursive=True) -> list:

        prefix = self.join_path(prefix) + "/"

        for key in self._list():
            if key.startswith(prefix):
                yield key.removeprefix(prefix)

    def __str__(self):
        return f"{self.__class__.__name__}({self.path}; {self.bucket}; {self.prefix})"


redis_pools = dict()


def connect_redis(url):
    import redis

    if url not in redis_pools:
        redis_pools[url] = redis.ConnectionPool.from_url(url)

    return redis.StrictRedis(decode_responses=True, connection_pool=redis_pools[url])


class RedisObjectStore(_ObjectStore):
    bucket: str = None
    prefix: str = None

    def __init__(
        self,
        bucket: str = None,
        prefix: str = None,
        url: str = None,
        client=None,
        **kwargs,
    ):

        self.url = url
        self.bucket = bucket
        self.prefix = prefix

        if client is None:
            self.client = connect_redis(url)
        else:
            self.client = client

    def sub(self, *args, **kwargs):

        if "name" not in kwargs and "class_" not in kwargs:
            extra_kwargs = {"client": self.client}
        else:
            extra_kwargs = {}

        return super().sub(*args, extra_kwargs=extra_kwargs, **kwargs)

    def put(self, key: str, data: bytes):
        return self.client.set(self.join_pathb(key), pickle.dumps(data))

    def get(self, key: str) -> bytes:

        d = self.client.get(self.join_pathb(key))
        if d is None:
            raise KeyError(f"No such key {key} in bucket {self.bucket}")
        return pickle.loads(d)

    def exists(self, key: str) -> bool:
        return self.client.exists(self.join_pathb(key))

    def delete(self, key: str):
        self.client.delete(self.join_pathb(key))

    def list(self, prefix: str = "", recursive=True) -> list:
        for e in self.client.scan_iter(self.join_pathb("*")):
            yield e.decode("utf8").replace(self.join_pathb(""), "").strip("/")

    def set(self, key: str = "set"):
        return RedisSet(self, key)

    def queue(self, key: str = "queue", max_length=None):
        return RedisQueue(self, key, max_length=max_length)

    def __str__(self):
        return f"{self.__class__.__name__}({self.bucket}, {self.prefix})"


class ObjectSet:
    """Interface for a set of objects, which operates on a single key"""

    def __init__(self, os: ObjectStore, key: str):
        self.os = os
        self.key = key

    def add(self, value):

        o = self.get()

        o.add(value)

        self.os.put(self.key, o)

    def remove(self, value):
        try:
            o = self.os.get(self.key)

            if not isinstance(o, set):
                raise TypeError(f"Object at {self.key} is not a set")

        except KeyError:
            o = set()

        o.remove(value)

        self.os.put(self.key, o)

    def __delitem__(self, key):
        return self.remove(key)

    def is_member(self, value):
        return value in self.get()

    def rand_member(self):
        import random

        return random.choice(list(self.os.get(self.key)))

    def get(self):
        try:
            o = self.os.get(self.key)

            if not isinstance(o, set):
                raise TypeError(f"Object at {self.key} is not a set")

        except KeyError:
            o = set()

        return o

    def __iadd__(self, other):

        # convert other to iterable if it isn't already
        if not hasattr(other, "__iter__"):
            other = [other]

        for e in other:
            self.add(e)
        return self

    def __isub__(self, other):

        # convert other to iterable if it isn't already
        if not hasattr(other, "__iter__"):
            other = [other]

        for e in other:
            self.remove(e)
        return self

    def __len__(self):
        return len(self.get())

    def __contains__(self, item):
        return self.is_member(item)

    def __iter__(self):
        return iter(self.get())

    def clear(self):
        self.put(self.key, set())


class RedisSet(ObjectSet):
    def __init__(self, os: ObjectStore, key: str):
        super().__init__(os, key)
        self.redis = os.client
        self.prefix = os.join_pathb(key)

    def add(self, value):

        # If value is iterable, call .add() to add each of the items
        if isinstance(value, (list, tuple, set)):
            for v in value:
                self.add(v)
            return

        return self.redis.sadd(self.prefix, pickle.dumps(value))

    def remove(self, value):
        return self.redis.srem(self.prefix, pickle.dumps(value))

    def __delitem__(self, key):
        return self.remove(key)

    def is_member(self, value):
        return self.redis.sismember(self.prefix, pickle.dumps(value))

    def rand_member(self):
        return pickle.loads(self.redis.srandmember(self.prefix))

    def get(self):
        return [pickle.loads(e) for e in self.redis.smembers(self.prefix)]

    def pop(self):
        """Select a random element in the set, remove it, and return it"""
        r = self.redis.spop(self.prefix)
        if r is None:
            return None
        else:
            return pickle.loads(r)

    def ipop(self):
        """A generate that pops from the set"""
        while True:
            v = self.pop()
            if v is None:
                break
            yield v

    def move(self, other, value):
        """Move a value from this set to another set"""
        return self.redis.smove(self.prefix, other.prefix, pickle.dumps(value))

    def __len__(self):
        return self.redis.scard(self.prefix)

    def __contains__(self, item):
        return self.is_member(item)

    def __iter__(self):
        for e in self.redis.smembers(self.prefix):
            yield pickle.loads(e)

    def clear(self):
        return self.redis.delete(self.prefix)


class RedisQueue(ObjectSet):
    def __init__(self, os: ObjectStore, key: str, max_length=None):
        super().__init__(os, key)
        self.redis = os.client
        self.prefix = os.join_pathb(key)
        self.max_length = max_length

    def _return(self, r):
        try:
            if r is not None:
                return pickle.loads(r)
            else:
                return r
        except TypeError:
            print("!!!!", r[:100])
            raise

    def push(self, value):
        r = self.redis.lpush(self.prefix, pickle.dumps(value))
        if self.max_length is not None:
            self.redis.ltrim(self.prefix, 0, self.max_length)
        return r

    def unpush(self):
        """Remove the last pushed item (pop from the tail)"""
        return self._return(self.redis.lpop(self.prefix))

    def pop(self):
        """Pop from the head"""
        return self._return(self.redis.rpop(self.prefix))

    def ipop(self):
        """Return an interator that pops from the head"""
        while True:
            v = self.pop()
            if v is None:
                break
            yield v

    def bpop(self, timeout=0):
        """Blocking pop from the head"""
        r = self.redis.brpop(self.prefix, timeout=timeout)
        if r is None:
            return None
        else:
            return self._return(r[1])

    def ibpop(self, timeout=0):
        """Blocking pop from the head"""
        while True:
            v = self.bpop(timeout=timeout)

            if v is None:
                break
            yield v

    def peek(self):
        """Peek at the head"""
        return self._return(self.redis.lrange(self.prefix, -1, -1))

    def is_member(self, value):
        return self.redis.sismember(self.prefix, pickle.dumps(value))

    def __len__(self):
        return self.redis.llen(self.prefix)

    def __contains__(self, item):
        return bool(self.lpos(item))

    def __iter__(self):
        for i in range(len(self)):
            e = self.redis.lindex(self.prefix, i)
            if e is not None:
                yield pickle.loads(e)

    def tail(self, n=10):
        for i in range(-1, -n - 1, -1):
            e = self.redis.lindex(self.prefix, i)
            if e is not None:
                yield pickle.loads(e)

    def head(self, n=10):
        for i in range(0, n, 1):
            e = self.redis.lindex(self.prefix, i)
            if e is not None:
                yield pickle.loads(e)

    def clear(self):
        return self.redis.delete(self.prefix)
