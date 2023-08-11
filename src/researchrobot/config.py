from pathlib import Path

from dotenv import dotenv_values


def get_config(path: Path):

    if not isinstance(path, Path):
        path = Path(path)

    if path.is_dir():
        return {
            **dotenv_values(path / ".env"),
            **dotenv_values(path / ".env.shared"),  # load shared development variables
            **dotenv_values(path / ".env.secret"),  # load sensitive variables
        }
    else:
        return dotenv_values(path)


config = {
    **dotenv_values(Path().home().joinpath(".researchrobot.env")),
    # **os.environ,  # override loaded values with environment variables,
    **dotenv_values("../.env"),
    **dotenv_values("../.env.shared"),  # load shared development variables
    **dotenv_values("../.env.secret"),  # load sensitive variables
    **dotenv_values(".env"),
    **dotenv_values(".env.shared"),  # load shared development variables
    **dotenv_values(".env.secret"),  # load sensitive variables
}
