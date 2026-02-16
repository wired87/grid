import base64
import json
import os
import zoneinfo
import jax.numpy as jnp

from data_handler.load_sa_creds import load_service_account_credentials
from data_handler.main import load_data
from gnn.gnn import GNN

import jax
from jax_utils.deserialize_in import parse_value


def _to_json_serializable(data):
    """Convert JAX/numpy arrays and complex to JSON-serializable (list/dict)."""
    if isinstance(data, list):
        return [_to_json_serializable(x) for x in data]
    if isinstance(data, tuple):
        return [_to_json_serializable(x) for x in data]
    if isinstance(data, dict):
        return {k: _to_json_serializable(v) for k, v in data.items()}
    if isinstance(data, bytes):
        # --- FIX: bytes (e.g. from base64.b64encode) -> decode to string for JSON ---
        # base64.b64encode returns ASCII-safe bytes, so decode as ASCII
        return data.decode('ascii')
    if hasattr(data, "shape") and hasattr(data, "tolist"):
        arr = jnp.asarray(data)
        if jnp.iscomplexobj(arr):
            return {"real": jnp.real(arr).tolist(), "imag": jnp.imag(arr).tolist()}
        return arr.tolist()
    if hasattr(data, "item"):
        v = data.item()
        if isinstance(v, (complex, jnp.complexfloating)):
            return {"real": float(jnp.real(v)), "imag": float(jnp.imag(v))}
        return float(v) if hasattr(v, "real") else v
    return data


class Guard:
    # todo prevaliate features to avoid double calculations
    def __init__(self):
        #JAX
        platform = "cpu" if os.name == "nt" else "gpu"
        jax.config.update("jax_platform_name", platform)  # must be run before jnp
        self.gpu = jax.devices(platform)[0]

        # LOAD DAT FROM BQ OR LOCAL
        self.cfg = load_data()

        AMOUNT_NODES = int(os.getenv("AMOUNT_NODES"))
        SIM_TIME = int(os.getenv("SIM_TIME"))
        DIMS = int(os.getenv("DIMS"))

        for k, v in self.cfg.items():
            self.cfg[k] = parse_value(v)

            if isinstance(self.cfg[k], dict):
                for i, o in self.cfg[k].items():
                    self.cfg[k][i] = parse_value(o)

        # layers
        self.gnn_layer = GNN(
            amount_nodes=AMOUNT_NODES,
            time=SIM_TIME,
            gpu=self.gpu,
            DIMS=DIMS,
            **self.cfg
        )

        from data_handler.bq_handler import BQCore
        self.bqclient = BQCore(
            dataset_id=os.getenv("BQ_DATASET")
        )

    def divide_vector(self, vec, divisor):
        """Divide all values of a given vector by divisor. Returns array same shape as vec."""
        v = jnp.asarray(vec)
        d = jnp.asarray(divisor)
        return v / d

    def main(self):
        self.run()
        #results = self.finish()
        print("SIMULATION PROCESS FINISHED")
        return None


    def run(self):
        # start sim on gpu
        print("run...")
        serialized_in, serialized_out = self.gnn_layer.main()
        print("run... done")

        self._export_engine_state(
            serialized_in,
            serialized_out,
        )

    def _export_data(self):
        print("_export_data...")
        dl = self.gnn_layer.db_layer

        history = []
        env_id = os.getenv("ENV_ID")

        for i, item in enumerate(dl.history_nodes):
            history.append(
                {
                    "id": f"{env_id}_{i}",
                    "data":_to_json_serializable(item),
                    "env_id": env_id
                }
            )

        # INSERT
        self.bqclient.bq_insert(
            table_id="data",
            rows = history
        )
        print("_export_data... done")

    def _(self, *args):
        return jax.


    def _export_ctlr(self):
        print("_export_ctlr...")
        dl = self.gnn_layer.db_layer
        env_id = os.getenv("ENV_ID")

        db_ctlr = {
            "id": env_id,
            "OUT_SHAPES": _to_json_serializable(dl.OUT_SHAPES),
            "SCALED_PARAMS": _to_json_serializable(dl.SCALED_PARAMS),
            "METHOD_TO_DB": _to_json_serializable(dl.METHOD_TO_DB),
            "AMOUNT_PARAMS_PER_FIELD": _to_json_serializable(dl.AMOUNT_PARAMS_PER_FIELD),
            "DB_PARAM_CONTROLLER": _to_json_serializable(dl.DB_PARAM_CONTROLLER),
            "DB_KEYS": _to_json_serializable(self.cfg["DB_KEYS"]),
            "FIELD_KEYS": _to_json_serializable(self.cfg["FIELD_KEYS"])
        }


        model_ctlr = {
            "id": env_id,
            "VARIATION_KEYS": _to_json_serializable(self.cfg["VARIATION_KEYS"]),
        }



        # INSERT
        self.bqclient.bq_insert(
            table_id="data",
            rows = [db_ctlr]
        )
        print("_export_ctlr... done")



    def _export_engine_state(self, serialized_in, serialized_out, out_path: str = "engine_output.json"):
        """Save all generated engine data (history, db, tdb, etc.) to a local .json file."""
        dl = self.gnn_layer.db_layer
        try:
            payload = {
                "serialized_out": _to_json_serializable(base64.b64encode(serialized_out).decode('ascii')),
                "serialized_in": _to_json_serializable(base64.b64encode(serialized_in).decode('ascii')),

                # CTLR
                "ENERGY_MAP": None,
            }
            if hasattr(dl, "tdb") and dl.tdb is not None:
                payload["tdb"] = _to_json_serializable(dl.tdb)

            self._upsert_generated_data_to_bq(payload)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, allow_nan=True)
            print("engine state saved to", out_path)
        except Exception as e:
            print("export_engine_state failed:", e)

    def finish(self):
        # Collect data
        history_nodes = self.gnn_layer.db_layer.history_nodes
        model_skeleton = self.gnn_layer.db_layer.model_skeleton

        # Serialization helper
        def serialize(data):
            if isinstance(data, list):
                return [serialize(x) for x in data]
            if isinstance(data, tuple):
                return tuple(serialize(x) for x in data)
            if isinstance(data, dict):
                 return {k: serialize(v) for k, v in data.items()}

            # Check for JAX/Numpy array
            if hasattr(data, 'dtype') and hasattr(data, 'real') and hasattr(data, 'imag'):
                # Check directly if complex dtype
                if jnp.iscomplexobj(data):
                    return (data.real, data.imag)
            return data

        serialized_history = serialize(history_nodes)
        serialized_model = serialize(model_skeleton)

        # Construct result dictionary
        result = {
            "DB_CONTROLLER": serialized_history,
            "MODEL_CONTROLLER": serialized_model
        }

        # --- BQ UPSERT: bytified model + full history + controller meta into BigQuery ---
        try:
            from google.cloud import bigquery
            from google.api_core.exceptions import NotFound
        except ImportError as e:
            print("BigQuery client not available, skipping upsert:", e)
            print("DATA DISTRIBUTED")
            return result

        project = os.getenv("BQ_PROJECT")
        dataset = os.getenv("DS")
        table = os.getenv("TABLE")
        model_col = os.getenv("MODEL_COL")
        data_col = os.getenv("DATA_COL")
        ctlr_col = os.getenv("CTLR_COL")

        if not (project and dataset and table and model_col and data_col and ctlr_col):
            print("BigQuery env vars missing (BQ_PROJECT/DS/TABLE/MODEL_COL/DATA_COL/CTL_RCOL), skipping upsert.")
            print("DATA DISTRIBUTED")
            return result

        table_id = f"{project}.{dataset}.{table}"

        try:
            creds = load_service_account_credentials(
                scopes=["https://www.googleapis.com/auth/bigquery"]
            )
            client = bigquery.Client(project=project, credentials=creds)
        except Exception as e:
            print("Failed to initialize BigQuery client, skipping upsert:", e)
            print("DATA DISTRIBUTED")
            return result

        # Bytify model, history (all time steps), and controller metadata
        try:
            model_bytes = self.gnn_layer.serialize(model_skeleton)
            history_bytes = self.gnn_layer.serialize(history_nodes)
            ctrl_payload = {
                "cfg": self.cfg,
                "AMOUNT_NODES": int(os.getenv("AMOUNT_NODES", "0") or 0),
                "SIM_TIME": int(os.getenv("SIM_TIME", "0") or 0),
                "DIMS": int(os.getenv("DIMS", "0") or 0),
            }
            ctrl_bytes = self.gnn_layer.serialize(ctrl_payload)
        except Exception as e:
            print("Failed to bytify model/history/controller, skipping upsert:", e)
            print("DATA DISTRIBUTED")
            return result

        # Ensure target table with BYTES columns exists
        try:
            client.get_table(table_id)
        except NotFound:
            schema = [
                bigquery.SchemaField(model_col, bigquery.SqlTypeNames.BYTES),
                bigquery.SchemaField(data_col, bigquery.SqlTypeNames.BYTES),
                bigquery.SchemaField(ctlr_col, bigquery.SqlTypeNames.BYTES),
            ]
            table_obj = bigquery.Table(table_id, schema=schema)
            client.create_table(table_obj)

        row = {
            model_col: model_bytes,
            data_col: history_bytes,
            ctlr_col: ctrl_bytes,
        }

        try:
            errors = client.insert_rows_json(table_id, [row])
            if errors:
                print("BigQuery upsert reported errors:", errors)
            else:
                print("BigQuery upsert successful to", table_id)
        except Exception as e:
            print("BigQuery upsert failed:", e)

        print("DATA DISTRIBUTED")
        return result

if __name__ == "__main__":
    Guard()._upsert_generated_data_to_bq(
        payload={
                "nodes": [],

                "history_nodes": [],
                "SCALED_PARAMS": [],
                "OUT_SHAPES": [],
                "METHOD_TO_DB": [],
                "serialized_out": [],
                "serialized_in": [],
            }
    )
