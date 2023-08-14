""" An abstraction layer for the object store, so we
can access S3 via Boto, but also use a local filesystem
"""

import json
import pickle
from pathlib import PosixPath

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
        try:
            size = o.getbuffer().nbytes
            return type_codes["io"], o, size, "application/octet-stream", ""
        except AttributeError:
            # Nope, not a buffer
            return _to_bytes(o.read())

    elif isinstance(o, object):
        try:
            o = json.dumps(o).encode("utf8")
            size = len(o)
            return type_codes["json"], o, size, "application/json", ""

        except TypeError:  # Probably can't be serialized with JSON

            o = pickle.dumps(o)
            size = len(o)
            return type_codes["pickle"], o, size, "application/x-pickle", ""

    else:
        raise IOError("Can't understand how to use object")


class ObjectStore:
    bucket: str = None
    prefix: str = None

    def __init__(self, bucket: str = None, prefix: str = None):
        self.bucket = bucket
        self.prefix = prefix

    def join_path(self, *args):

        args = [self.prefix] + list(args)
        args = [e.strip("/") for e in args]
        args = [e for e in args if e]

        return "/".join(args)

    def put(self, key: str, data: bytes):
        raise NotImplementedError

    def get(self, key: str) -> bytes:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def delete(self, key: str):
        raise NotImplementedError

    def list(self, prefix: str) -> list:
        raise NotImplementedError


class S3ObjectStore(ObjectStore):
    def __init__(
        self,
        bucket: str = None,
        prefix: str = None,
        access_key: str = None,
        secret_key: str = None,
        endpoint: str = None,
        region: str = None,
        client=None,
    ):

        import boto3

        if "/" in bucket:
            bucket, _prefix = bucket.split("/", 1)
            if prefix is None:
                prefix = _prefix
            else:
                prefix = _prefix + "/" + prefix

        self.bucket = bucket
        self.prefix = prefix
        self.client = None

        config = {}

        if endpoint:
            config["endpoint_url"] = endpoint
        if region:
            config["region_name"] = region

        if client is None:
            self.session = boto3.session.Session()
            self.client = self.session.client(
                "s3",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                **config,
            )
        else:
            self.client = client

    def sub(self, *args):
        return S3ObjectStore(
            bucket=self.bucket, prefix=self.join_path(*args), client=self.client
        )

    def _put_bytes(
        self, key: str, data: bytes, content_type: str = None, metadata: dict = None
    ):

        return self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type or "application/octet-stream",
            ACL="private",
            # Metadata=metadata or {}
        )

    def put(self, key: str, data, metadata: dict = None):
        if dict is None:
            metadata = {}

        tc, b, size, content_type, ext = _to_bytes(data)

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
            elif r.get("ContentType") == "application/json":
                return json.loads(body.read().decode("utf8"))
            elif r.get("ContentType") == "application/x-pickle":
                return pickle.loads(body.read())
            else:
                raise IOError(
                    f"Can't understand response for get of {self.bucket}/{self.key}: content-type={r.get('ContentType')}"
                )
        except Exception as e:
            raise
            if e.code == "NoSuchKey":
                raise FileNotFoundError(
                    f"No such key bucket={self.bucket},  path={self.key}"
                )
            else:
                raise e

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self.join_path(key))
            return True
        except Exception:
            return False

    def delete(self, key: str):
        self.client.delete_object(Bucket=self.bucket, Key=self.join_path(key))

    def list(self, prefix: str) -> list:
        response = self.client.list_objects(
            Bucket=self.bucket, Prefix=self.join_path(prefix)
        )
        return [e["Key"] for e in response.get("Contents", [])]

    def __str__(self):
        return f"{self.__class__.__name__}({self.bucket}, {self.prefix})"
