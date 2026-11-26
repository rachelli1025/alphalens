"""Walking-skeleton DAG: proves Airflow can import alphalens_core and reach the app DB.

Replaced by sec_filings_daily in W1 step 4.
"""

import pendulum
from airflow.sdk import dag, task


@dag(
    schedule=None,
    start_date=pendulum.datetime(2026, 11, 1, tz="America/New_York"),
    catchup=False,
    tags=["skeleton"],
)
def skeleton_healthcheck() -> None:
    @task
    def check_app_db() -> str:
        from sqlalchemy import create_engine, text

        from alphalens_core.config import get_settings

        engine = create_engine(get_settings().database_url)
        with engine.connect() as conn:
            version = conn.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname='vector'")
            )
        if version is None:
            raise RuntimeError("pgvector missing: run `alembic upgrade head` first")
        return str(version)

    check_app_db()


skeleton_healthcheck()
