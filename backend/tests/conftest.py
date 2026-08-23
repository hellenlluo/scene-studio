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

# Blank the provider credentials so no test can reach a paid API by accident.
# The environment wins over backend/.env, and `Settings.require` rejects an empty
# value — so a stray call fails immediately and by name instead of billing.
#
# Not hypothetical: the upload test runs the pipeline in the background, and the
# moment stage 3 started calling OpenAI it went from 0.005s to 33s per run, on a
# 64x64 grey square.
for _key in ("OPENAI_API_KEY", "FAL_KEY", "REPLICATE_API_TOKEN"):
    os.environ[_key] = ""

import pytest  # noqa: E402

from app.db import init_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _create_tables():
    init_db()
