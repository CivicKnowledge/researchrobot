import os
from pathlib import Path

import yaml

# Configuration paths, in order of precedence
_cache_conf_paths = [
    Path("/usr/local/etc/robotcache.yaml"),
    Path().home().joinpath(".robotcache.yaml"),
    Path(os.getenv("ROBOTCACHE_CONF")) if os.getenv("ROBOTCACHE_CONF") else None,
    Path("../robotcache.yaml"),
    Path("../robotcache-secret.yaml"),
    Path("../.robotcache.yaml"),
    Path("../.robotcache-secret.yaml"),
    Path("robotcache.yaml"),
    Path("robotcache-secret.yaml"),
    Path(".robotcache.yaml"),
    Path(".robotcache-secret.yaml"),
]


def extant_paths():
    """return the configuration paths that exist"""
    return [p for p in _cache_conf_paths if p and p.exists()]


def get_config(paths=None):
    from flatten_dict import flatten, unflatten

    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    paths = [Path(p) for p in paths if p]

    if not paths:
        paths = extant_paths()

    loaded = []
    errors = []
    lines = []
    for path in paths:
        with path.open() as f:
            try:
                d = yaml.safe_load(f)
                loaded.append(path)  # Keep track of the loaded paths

                lines.extend(flatten(d, reducer="dot").items())

            except yaml.YAMLError as exc:
                errors.append((path, exc))

    return unflatten(dict(lines), splitter="dot")
