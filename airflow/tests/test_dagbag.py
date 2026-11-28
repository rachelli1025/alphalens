"""DAG import and structure tests. Run where Airflow is installed (the Airflow image in CI).

These check wiring, not behaviour: task bodies are plain functions in
alphalens_core.pipelines with their own tests.
"""

import pathlib

import pytest

pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag  # noqa: E402

DAGS_DIR = pathlib.Path(__file__).parents[1] / "dags"

EXPECTED = {
    "sec_filings_daily": "ingest_filing",
    "xbrl_financials_daily": "ingest_financials",
}


@pytest.fixture(scope="module")
def bag() -> DagBag:
    return DagBag(dag_folder=str(DAGS_DIR))


def test_no_import_errors(bag: DagBag) -> None:
    assert bag.import_errors == {}
    assert set(EXPECTED) <= set(bag.dag_ids)


@pytest.mark.parametrize(("dag_id", "ingest_task"), EXPECTED.items())
def test_structure(bag: DagBag, dag_id: str, ingest_task: str) -> None:
    dag = bag.dags[dag_id]
    assert set(dag.task_ids) == {"get_universe", ingest_task, "validate"}
    assert dag.task_dict[ingest_task].upstream_task_ids == {"get_universe"}
    assert dag.task_dict["validate"].upstream_task_ids == {"get_universe", ingest_task}


@pytest.mark.parametrize(("dag_id", "ingest_task"), EXPECTED.items())
def test_operational_settings(bag: DagBag, dag_id: str, ingest_task: str) -> None:
    dag = bag.dags[dag_id]
    assert dag.catchup is False  # never backfill months of runs on first deploy
    assert dag.max_active_runs == 1  # runs never overlap
    ingest = dag.task_dict[ingest_task]
    assert ingest.retries == 3
    assert ingest.max_active_tis_per_dagrun == 2  # SEC rate limit
    assert dag.task_dict["validate"].trigger_rule == "all_done"
