import os
from pathlib import Path

import requests
from slugify import slugify


def cache_dl(directory, url):
    directory = Path(directory)

    if not directory.exists():
        directory.mkdir(parents=True)

    # Slugify the URL for filename
    filename = slugify(url)
    filepath = directory.joinpath(filename)

    # Check if the file already exists
    if filepath.exists():
        return filepath

    # If file doesn't exist, download and save it
    response = requests.get(url)
    with open(filepath, "wb") as file:
        file.write(response.content)

    return filepath


def make_cache_dl(directory):
    def _cache_dl(url):
        return cache_dl(directory, url)

    return _cache_dl
