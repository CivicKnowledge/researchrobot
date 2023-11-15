import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
import requests
from slugify import slugify
from tqdm.auto import tqdm


def cache_path(url, directory):

    directory = Path(directory)

    if not directory.exists():
        directory.mkdir(parents=True)

    # Slugify the URL for filename
    filename = slugify(url)
    filepath = directory.joinpath(filename)

    return filepath


def web_download(url, directory, force=False):

    filepath = cache_path(url, directory)

    # Check if the file already exists
    if filepath.exists():
        return filepath
    else:
        head = requests.head(url)
        total_length = int(head.headers.get("content-length", 0))

        # If file doesn't exist, download and save it
        response = requests.get(url)

        pb = tqdm(total=total_length, unit="B", unit_scale=True)

        block_size = 64 * 1024

        with open(filepath, "wb") as file:

            for data in response.iter_content(block_size):
                pb.update(len(data))
                file.write(data)

            c = response.content
            file.write(c)

        return filepath


def s3_download(url, directory, force=False):

    filepath = cache_path(url, directory)

    # Check if the file already exists
    if filepath.exists() and force is False:

        return filepath
    else:
        up = urlparse(url)
        s3 = boto3.client("s3")

        bucket, key = up.netloc, up.path.lstrip("/")

        meta_data = s3.head_object(Bucket=bucket, Key=key)
        total_length = int(meta_data.get("ContentLength", 0))

        pb = tqdm(total=total_length, unit="B", unit_scale=True)

        s3.download_file(bucket, key, str(filepath), Callback=pb.update)

        bucket, key = up.netloc, up.path.lstrip("/")
        s3.download_file(bucket, key, '.')

        return filepath


def cache_dl(url, directory=None, force=False):
    """Download a file to a cache directory. If the cache directory is not specified,
    use the default cache directory, specified by the environment variable
    RESEARCH_ROBOT_DEFAULT_CACHE, or 'rrcache' if not set.
    """

    if directory is None:
        directory = os.getenv("RESEARCH_ROBOT_DEFAULT_CACHE", "rrcache")

    up = urlparse(url)

    if up.scheme == "s3":
        return s3_download(url, directory, force=force)
    elif up.scheme == "http" or up.scheme == "https":
        return web_download(url, directory, force=force)
    else:
        raise Exception(f"Unknown scheme: {up.scheme}")


def share(o, bucket, key, profile=None):
    from .objectstore import _to_bytes

    b, size, content_type, ext = _to_bytes(o)

    if profile is None:
        profile = os.getenv("RESEARCH_ROBOT_DEFAULT_PROFILE", "default")
    else:
        profile = profile

    session = boto3.Session(profile_name=profile)

    s3 = session.client("s3")

    # Upload bytes in b to S3 object at bucket/key
    s3.put_object(Bucket=bucket, Key=key, Body=b, ContentType=content_type)

    return f"cache_dl('s3://{bucket}/{key}')"
