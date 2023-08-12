from pathlib import Path

from dotenv import dotenv_values


def get_config(path: Path = None):

    if path is not None and not isinstance(path, Path):
        path = Path(path)

    if path is not None and path.is_dir():
        return {
            **dotenv_values(path / ".env"),
            **dotenv_values(path / ".env.shared"),  # load shared development variables
            **dotenv_values(path / ".env.secret"),  # load sensitive variables
        }
    elif path is not None:
        return dotenv_values(path)

    else:

        all_config = []
        for p in extant_paths():
            all_config.extend(dotenv_values(p).items())

        d = dict(all_config)
        d["_loaded"] = [str(p) for p in extant_paths()]

        return d


# Configuration paths, in order of precedence
conf_paths = [
    Path("/usr/local/etc/researchrobot.env"),
    Path().home().joinpath(".researchrobot.env"),
    # **os.environ,  # override loaded values with environment variables,
    Path("../.env"),
    Path("../.env.shared"),  # load shared development variables
    Path("../.env.secret"),  # load sensitive variables
    Path(".env"),
    Path(".env.shared"),  # load shared development variables
    Path(".env.secret"),  # load sensitive variables
]


def extant_paths():
    """return the configuration paths that exist"""
    return [p for p in conf_paths if p.exists()]


config = get_config()
