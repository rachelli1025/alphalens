"""Every DAG must import cleanly. Runs where Airflow is installed (the Airflow image / CI)."""

import pathlib

import pytest

pytest.importorskip("airflow")

from airflow.models import DagBag  # noqa: E402

DAGS_DIR = pathlib.Path(__file__).parents[1] / "dags"


def test_no_import_errors() -> None:
    bag = DagBag(dag_folder=str(DAGS_DIR), include_examples=False)
    assert bag.import_errors == {}
    assert "skeleton_healthcheck" in bag.dag_ids
