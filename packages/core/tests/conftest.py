"""Shared fixtures.

Database tests run only against TEST_DATABASE_URL, never DATABASE_URL: the app's
.env points at a real database (e.g. Neon) and tests must not touch it. CI sets
TEST_DATABASE_URL to its throwaway Postgres; locally these tests are skipped
unless you point it at a disposable database.
"""

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError


@pytest.fixture(scope="session")
def pg_engine() -> Iterator[Engine]:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    engine = create_engine(url, connect_args={"connect_timeout": 2})
    try:
        with engine.connect() as conn:
            if conn.scalar(text("SELECT to_regclass('chunks')")) is None:
                pytest.skip("test database not migrated")
    except OperationalError:
        pytest.skip("test database not reachable")
    yield engine
    engine.dispose()
