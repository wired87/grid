from __future__ import annotations

import json
import os

from google.cloud import bigquery

from data_handler.bq_handler import BQCore

import dotenv
dotenv.load_dotenv()


def get_env_cfg(bqcore, user_id: str, env_id, table, select: str = "*"):
    """
    Retrieve all modules for a user.
    Groups by ID and returns only the newest entry per ID (based on created_at).
    """
    print("get_env_cfg...")
    query = f"""
        SELECT {select}
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY created_at DESC) as row_num
            FROM `{bqcore.ds_ref}.{table}`
            WHERE (user_id = @user_id) AND (status != 'deleted' OR status IS NOT NULL) AND (id = @env_id)
        )
        WHERE row_num = 1
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("env_id", "STRING", env_id)
        ]
    )

    result = bqcore.run_query(query, conv_to_dict=True, job_config=job_config)
    print("get_env_cfg result:", result)

    # filter out deleted entries
    result = [entry for entry in result if entry.get("status", None) != "deleted"]
    #print("get_env_cfg result:", result)
    return result


def load_data():
    cfg = {}
    if os.name == "nt" and os.getenv("TESTING1", None) is not None:
        cfg = open("test_out.json", "r").read()
    else:
        try:
            cfg = get_env_cfg(
                bqcore=BQCore(
                    dataset_id=os.getenv("BQ_DATASET")
                ),
                user_id=os.getenv("USER_ID"),
                env_id=os.getenv("ENV_ID"),
                table=os.getenv("PATTERN_TABLE"),
                select="pattern",
            )
            cfg = cfg[0].get("pattern")
        except Exception as e:
            print("Err get_env_cfg", e)
    return json.loads(cfg)



