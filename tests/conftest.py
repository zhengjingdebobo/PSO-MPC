"""Isolate temporary test output from Windows user/sandbox shared directories."""
from tempfile import TemporaryDirectory
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Allocate a unique workspace directory for each run, then clean it up.

    Respect an explicit --basetemp. Otherwise avoid pytest-of-USER directories
    which may have been created by a different Windows execution identity.
    TemporaryDirectory only removes the new directory created by this run.
    """
    if config.option.basetemp is None:
        directory = TemporaryDirectory(prefix=".pytest-tmp-", dir=config.rootpath)
        config.add_cleanup(directory.cleanup)
        config.option.basetemp = str(Path(directory.name) / "work")
