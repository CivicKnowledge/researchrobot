import sys

__version__ = None

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


def require_version(version: str):
    from semantic_version import Version

    from .exceptions import WrongVersion

    if Version(__version__) < Version(version):
        raise WrongVersion(
            f"ERROR! You must have at least version {version} of research robot! ( you have {__version__} )"
        )


from .download import cache_dl
from .objectstore import ObjectStore, oscache
