import os
from pathlib import Path
from urllib.parse import urlparse

import boto3
import requests
from slugify import slugify


def cache_path(url, directory):

    directory = Path(directory)

    if not directory.exists():
        directory.mkdir(parents=True)

    # Slugify the URL for filename
    filename = slugify(url)
    filepath = directory.joinpath(filename)

    return filepath


def web_download(url, directory):

    filepath = cache_path(url, directory)

    # Check if the file already exists
    if filepath.exists():
        return filepath
    else:
        # If file doesn't exist, download and save it
        response = requests.get(url)
        with open(filepath, "wb") as file:
            file.write(response.content)

        return filepath


def s3_download(url, directory):

    filepath = cache_path(url, directory)

    # Check if the file already exists
    if filepath.exists():
        return filepath
    else:
        up = urlparse(url)
        s3 = boto3.client("s3")
        s3.download_file(up.netloc, up.path.lstrip("/"), str(filepath))

        return filepath


def cache_dl(url, directory=None):
    """Download a file to a cache directory. If the cache directory is not specified,
    use the default cache directory, specified by the environment variable
    RESEARCH_ROBOT_DEFAULT_CACHE, or 'rrcache' if not set.
    """

    if directory is None:
        directory = os.getenv("RESEARCH_ROBOT_DEFAULT_CACHE", "rrcache")

    up = urlparse(url)

    if up.scheme == "s3":
        return s3_download(url, directory)
    elif up.scheme == "http" or up.scheme == "https":
        return web_download(url, directory)
    else:
        raise Exception(f"Unknown scheme: {up.scheme}")


def make_cache_dl(directory):
    def _cache_dl(url):
        return cache_dl(directory, url)

    return _cache_dl
