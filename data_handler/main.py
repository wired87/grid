from __future__ import annotations

import json
import os
from typing import List

import dotenv
import jax.numpy as jnp
"""from google.cloud import bigquery
from google.api_core.exceptions import NotFound"""

from data_handler.load_sa_creds import load_service_account_credentials

dotenv.load_dotenv()


def load_data():
    if os.getenv("TESTING", "").lower() == "true":
        return json.loads(open("test_out.json", "r").read())
    else:
        # load data from BigQuery (not implemented yet)
        return {}


def upsert_arrays_to_bigquery(
    arrays: List[jnp.ndarray],
    env_id: str | None = None,
    project: str | None = None,
    dataset: str | None = None,
) -> None:
    """
    Upsert (append) a list of jnp.array rows into
    `<project>.<dataset>.<ENV_ID>_data`.

    - Table name: f\"{env_id}_data\" where `env_id` defaults to ENV_ID env var.
    - Column names: c0, c1, ..., one per element in the flattened array.
    """

    if env_id is None:
        env_id = os.getenv("ENV_ID")
    if not env_id:
        raise ValueError("ENV_ID env var must be set or passed explicitly.")

    if project is None:
        project = os.getenv("BQ_PROJECT")
    if dataset is None:
        dataset = os.getenv("BQ_DATASET")
    if not project or not dataset:
        raise ValueError("BQ_PROJECT and BQ_DATASET env vars must be set or passed.")

    table_id = f"{project}.{dataset}.{env_id}_data"

    creds = load_service_account_credentials(
        scopes=["https://www.googleapis.com/auth/bigquery"]
    )
    client = bigquery.Client(project=project, credentials=creds)

    if not arrays:
        return

    # Flatten all arrays first to determine max width for schema.
    flat_rows = [jnp.asarray(a).ravel().tolist() for a in arrays]
    max_len = max(len(fr) for fr in flat_rows)
    col_names = [f"c{i}" for i in range(max_len)]

    # Create table with explicit schema if it does not exist.
    try:
        client.get_table(table_id)
    except NotFound:
        schema = [
            bigquery.SchemaField(name, bigquery.SqlTypeNames.FLOAT64)
            for name in col_names
        ]
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)

    # Build row dicts, padding shorter arrays with NULLs.
    rows = []
    for flat in flat_rows:
        row = {}
        for idx, name in enumerate(col_names):
            row[name] = flat[idx] if idx < len(flat) else None
        rows.append(row)

    errors = client.insert_rows_json(table_id, rows)
    if errors:
        raise RuntimeError(f"BigQuery upsert failed: {errors}")
