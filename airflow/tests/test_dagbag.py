"""Every DAG must import cleanly. Runs where Airflow is installed (the Airflow image / CI)."""

import pathlib

import pytest

pytest.importorskip("airflow")

# Airflow 3.x location; DagBag no longer loads example DAGs, so no include_examples flag.
from airflow.dag_processing.dagbag import DagBag  # noqa: E402

DAGS_DIR = pathlib.Path(__file__).parents[1] / "dags"


def test_no_import_errors() -> None:
    bag = DagBag(dag_folder=str(DAGS_DIR))
    assert bag.import_errors == {}
    assert "skeleton_healthcheck" in bag.dag_ids
