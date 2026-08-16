"""Point the app at a throwaway database and storage dir before it is imported.

app.db builds its engine at import time from the cached settings, so the
environment has to be set here, above the app imports.
"""

import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="scenestudio-test-"))
os.environ["SCENESTUDIO_STORAGE_DIR"] = str(_tmp)
os.environ["SCENESTUDIO_DATABASE_URL"] = f"sqlite:///{_tmp / 'test.db'}"

import pytest  # noqa: E402

from app.db import init_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _create_tables():
    init_db()
