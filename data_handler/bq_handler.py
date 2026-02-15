import json
import os
import time
from typing import List

import pandas as pd
import io
import re

from google.api import policy_pb2
from google.api_core.exceptions import NotFound, Conflict
from google.cloud.bigquery.table import _EmptyRowIterator

import dotenv
dotenv.load_dotenv()

from google.cloud import bigquery
GCP_ID = os.environ['BQ_PROJECT']
from data_handler.load_sa_creds import load_service_account_credentials

_bq_client_singleton = None

class BQGroundZero:

    """
    BQ ERLAUBT NUR 5 ROW UPSERTIONS / 10sec / TABLE
    todo: migrate tables  from sp <-> bq
    """
    DEFAULT_TIMESTAMP = "CURRENT_TIMESTAMP()"

    def __init__(self, dataset_id=None):
        #print("BQGroundZero.__init__ started")
        self.pid = os.getenv("P") or os.getenv("BQ_PROJECT") or GCP_ID
        self.ds_id = dataset_id or os.getenv("DS", "")
        self.ds_ref = f"{GCP_ID}.{self.ds_id}"
        
        creds = load_service_account_credentials()
        self.bqclient = bigquery.Client(
            credentials=creds
        )


    def ensure_dataset_exists(self, ds_name=None):
        """
        Ensure that the BigQuery dataset exists before creating tables.
        """
        if ds_name is None:
            ds_name = self.ds_id

        ds_name = f"{GCP_ID}.{ds_name}"
        try:
            self.bqclient.get_dataset(ds_name)
            print(f"Dataset '{ds_name}' already exists.")
        except Exception as e:
            print(f"Dataset '{ds_name}' does not exist. Creating: {e}")
            dataset = bigquery.Dataset(ds_name)
            dataset.location = "US"
            self.bqclient.create_dataset(dataset)
            print(f"Dataset '{ds_name}' created successfully.")




    def upsert_query(self, table_id):
        return f"""
        MERGE INTO `{self.pid}.{self.ds_id}.{table_id.upper()}` T
        USING `{self.pid}.{self.ds_id}.{table_id.upper()}` S
        ON T.nid = S.nid
        WHEN MATCHED THEN
          UPDATE SET
            T.column1 = S.column1,
            T.column2 = S.column2,
            -- Update other columns as needed
            T.last_updated = CURRENT_TIMESTAMP() -- Example: update a timestamp
        WHEN NOT MATCHED THEN
          INSERT (nid, column1, column2, ...)
          VALUES (S.nid, S.column1, S.column2, ...);
        
                q_types = ""
        for k, v in schema.items():
            bq_type = f"{v},"
            bq_types += bq_type
        bq_types=bq_types[:-1]
        """

    def upsert_row_query(self, table_id: str, rows: list[dict], schema: dict) -> str:
        struct_rows = []
        for row in rows:
            val_strs = []
            for k in schema.keys():
                v = row.get(k)
                if isinstance(v, str):
                    # Escape quotes for SQL safety
                    safe_val = v.replace('"', '\\"')
                    val_strs.append(f'"{safe_val}" AS {k}')
                elif v is None:
                    val_strs.append(f'NULL AS {k}')
                else:
                    val_strs.append(f'{v} AS {k}')

            struct_rows.append(f"STRUCT({', '.join(val_strs)})")

        # Final query using 'nid' as the Primary Key
        return f"""
            MERGE INTO `{self.pid}.{self.ds_id}.{table_id}` T
            USING (SELECT * FROM UNNEST([{', '.join(struct_rows)}])) AS S
            ON T.nid = S.nid
            WHEN MATCHED THEN UPDATE SET {", ".join([f"T.{k}=S.{k}" for k in schema.keys()])}
            WHEN NOT MATCHED THEN INSERT ({", ".join(schema.keys())}) VALUES ({", ".join([f"S.{k}" for k in schema.keys()])})
        """



    def upsert_row_query(self, table_id: str, rows: list[dict], schema: dict) -> str:
        print("Upsert to", table_id)

        # Create a string representation of the rows as BigQuery STRUCTs
        # Example: STRUCT(1 AS id, "value1" AS col1), STRUCT(2 AS id, "value2" AS col1)
        struct_rows = []
        for row in rows:
            val_strs = []
            for k, v in row.items():
                # Handle different admin_data types for correct BigQuery literal representation
                if isinstance(v, str):
                    val_strs.append(f'"{v}" AS {k}')  # Use double quotes for strings
                elif isinstance(v, (int, float)):
                    val_strs.append(f'{v} AS {k}')
                elif v is None:
                    val_strs.append(f'NULL AS {k}')
                else:
                    # Attempt a generic representation, might need more specific handling
                    val_strs.append(f'{repr(v)} AS {k}')

            struct_str = ", ".join(val_strs)
            struct_rows.append(f"STRUCT({struct_str})")

        # Construct the USING clause with UNNESTing the array of STRUCTs
        # BigQuery expects the UNNEST to be the source of the MERGE
        unnested_source = f"""
            (SELECT * FROM UNNEST([
                {', '.join(struct_rows)}
            ]))
        """

        # Construct the UPDATE SET clause
        # This updates existing rows with values from the source, prioritizing source values
        update_clause_parts = []
        for k in schema.keys():
            # Use S.<column_name> for values from the source (unnested rows)
            update_clause_parts.append(f"T.{k} = S.{k}")
        update_clause = ",\n          ".join(update_clause_parts)  # Indent for readability

        # Construct the INSERT clause
        # This inserts new rows from the source
        insert_cols = ", ".join(schema.keys())
        # Use S.<column_name> for values to be inserted
        insert_vals = ", ".join(f"S.{col}" for col in schema.keys())

        # Construct the full MERGE statement
        # The alias 'S' is applied to the result of the UNNEST subquery
        query = f"""
            MERGE INTO `{self.pid}.{self.ds_id}.{table_id}` T
            USING {unnested_source} AS S
            ON T.nid = S.nid
            WHEN MATCHED THEN
              UPDATE SET
                {update_clause}
            WHEN NOT MATCHED THEN
              INSERT ({insert_cols}) VALUES ({insert_vals})
        """
        return query.strip()

    def get_parent(self, table:str):
        return f"projects/{self.pid}/datasets/{self.ds_id}/tables/{table}"


    def get_table_name(self, table):
        return f"{self.pid}.{self.ds_id}.{table}"

    def table_schema_query(self, table):
        return f"""
        SELECT
            column_name, data_type
        FROM
          `{self.pid}.{self.ds_id}.INFORMATION_SCHEMA.COLUMNS`
        WHERE
          table_name = '{table}'
        ORDER BY
          ordinal_position
        """
    def create_default_table_query(self, table_id, ttype):
        # bach query hits quota
        query= f"""
            CREATE TABLE IF NOT EXISTS `{self.pid}.{self.ds_id}.{table_id}` (
                nid STRING,
            )
            """
        #print("Table query", query)
        return query


    def add_col_query(self, col_name, table, col_value):
        col_type=self.get_bq_type(col_value)
        return f"""
            ALTER TABLE `{self.pid}.{self.ds_id}.{table}`
            ADD COLUMN `{col_name}` {col_type}
        """

    def get_id_from_table_query(self,  table,):
        return f"""
            SELECT id FROM `{self.pid}.{self.ds_id}.{table}`
        """

    def get_entry_from_table_query(self, table, key_of_interest, value_of_interest):
        return f"""
            SELECT * FROM `{self.pid}.{self.ds_id}.{table}`
            WHERE {key_of_interest} = {value_of_interest}
        """


    def entry_from_parent_entry_query(self, table, parent_entry):
        return f"""
        SELECT *
        FROM `{self.pid}.{self.ds_id}.{table}`
        WHERE EXISTS(SELECT 1 FROM UNNEST(parent) AS item 
        WHERE item = {parent_entry})
        """

    def get_bq_type(self, value):
        # Convert Python types to BigQuery types
        if isinstance(value, int):
            return "INT64"
        elif isinstance(value, float):
            return "FLOAT64"
        elif isinstance(value, bool):
            return "BOOL"
        elif isinstance(value, bytes):
            return "BYTES"
        elif isinstance(value, list):
            return "ARRAY<STRING>"  # Adjust as needed
        else:
            return "STRING"

    def schema_from_dict(self, rows: list, embed=None) -> dict[str, str]:
        schema = {}
        for data in rows:
            for key, value in data.items():
                if isinstance(value, bool):
                    f_type = "BOOLEAN"
                elif isinstance(value, int):
                    f_type = "INTEGER"
                elif isinstance(value, float):
                    f_type = "FLOAT"
                elif isinstance(value, list) and embed:
                    # BigQuery uses 'REPEATED' mode for arrays,
                    # but for a type-string dict, we label it ARRAY
                    f_type = "ARRAY<NUMERIC>"
                elif isinstance(value, bytes):
                    f_type = "BYTES"
                else:
                    f_type = "STRING"

                if key not in schema:
                    schema[key] = f_type
        return schema



    def convert_dict_shema_bq(self, schema:dict):
        s=[]
        for k, v in schema.items():
            s.append(bigquery.SchemaField(k, v, mode="NULLABLE"))
        return s

    def run_query(self, query: str or list, conv_to_dict=False, job_config=None):
        try:
            if isinstance(query, list):
                query = ";\n".join(query)

            #print("run BQ Query:", query)

            job = self.bqclient.query(query, job_config=job_config)

            result = job.result()

            #print("BQ Query Result finished")
            if result:
                try:
                    if conv_to_dict is True:
                        result = [dict(row) for row in result]
                    #print("Return", len(result))
                except Exception as e:
                    print("Query result has no len:", e)

            return result
        except Exception as e:
            print(f"Error executing query:\n", e)


class BQCore(BQGroundZero):

    def __init__(self, dataset_id=None):
        BQGroundZero.__init__(self, dataset_id)
        self.dataset_id = dataset_id or os.getenv("DS", "")
        self.batch_upload_size = 1000

    def upsert_env_payload(self, table_id: str, env_id: str, payload: dict) -> None:
        """One env row: id=env_id, one STRING col per payload key (values JSON-serialized). Creates table if missing."""






    def get_table_schema(self, table_id: str, schema: dict[str, str], create_if_not_exists: bool = True):
        """
        Defines a table's schema and optionally creates or updates it.

        This method receives a dictionary of column names and their BigQuery types.
        If the table does not exist, it creates it with the specified schema.
        If the table exists, it adds any missing columns.

        Args:
            table_id (str): The ID of the table.
            schema (dict[str, str]): A dictionary mapping column names to their BigQuery types.
                                     Example: {"nid": "STRING", "value": "FLOAT64"}
            create_if_not_exists (bool): If True, creates the table if it is missing.
        """
        print(f"Defining schema for table '{table_id}'.")
        table_ref = self.get_table_name(table_id)

        try:
            table = self.bqclient.get_table(table_ref)
            print(f"Table '{table_id}' already exists. Verifying schema.")
            existing_schema_fields = {field.name: field.field_type for field in table.schema}
            return existing_schema_fields
        except NotFound:
            if create_if_not_exists:
                print(f"Table '{table_id}' not found. Creating with provided schema.")
                bq_schema_fields = self.convert_dict_shema_bq(schema)
                new_table = bigquery.Table(table_ref, schema=bq_schema_fields)
                try:
                    self.bqclient.create_table(new_table)
                    print(f"Table '{table_id}' created successfully.")
                except Conflict:
                    print(f"Table '{table_id}' already exists (Conflict).")
            else:
                print(f"Table '{table_id}' not found. Creation is disabled.")
        except Exception as e:
            print(f"An error occurred in define_table_schema: {e}")

    def get_tables(self) -> List[str]:
        tables = []
        for table in self.bqclient.list_tables(self.dataset):
            tables.append(table.table_id)

        return tables


    def add_columns_bulk(self, table_name: str, new_columns: dict):
        """
        Adds multiple missing columns in one operation.

        :param table_name: The BigQuery table name.
        :param new_columns: Dictionary {column_name: column_type}.
        """
        table_ref = f"{self.bqclient.project}.{self.ds_id}.{table_name}"
        table = self.bqclient.get_table(table_ref)
        updated_schema = table.schema + [bigquery.SchemaField(col, col_type) for col, col_type in new_columns.items()]
        table.schema = updated_schema
        self.bqclient.update_table(table, ["schema"])
        print(f"[OK] Added missing columns: {', '.join(new_columns.keys())}")


    def bq_check_table_exists(self, table_name):
        try:
            self.bqclient.get_table(f"{self.pid}.{self.ds_id}.{table_name}")
            print("Table exists")
            return True
        except Exception as e:
            print(f"Table not {table_name} found:", e)
            return False




    def set_ds_open_src(self, dataset_ref):
        # 3. Handle Open Source (Public Access) Flag

        print("DEBUG: Setting 'allUsers' to BigQuery Data Viewer role.")

        # TODO: ASYNC IMPLEMENTATION: In a production async environment,
        # replace the synchronous get_iam_policy/set_iam_policy calls with their
        # asynchronous equivalents if using an async client, or use asyncio.to_thread().

        try:
            # 3a. Get current IAM policy (synchronous call assumed for bqclient)
            policy = self.bqclient.get_iam_policy(dataset_ref)

            # 3b. Define the role and member to add
            role_to_add = "roles/bigquery.dataViewer"
            member_to_add = "allUsers"

            # 3c. Check if the binding already exists
            existing_binding = next(
                (b for b in policy.bindings if b.role == role_to_add), None
            )

            if existing_binding:
                # Add member to existing binding if not present
                if member_to_add not in existing_binding.members:
                    existing_binding.members.append(member_to_add)
                    print(f"DEBUG: Added {member_to_add} to existing {role_to_add} binding.")
                else:
                    print(f"DEBUG: {member_to_add} already has {role_to_add}. Skipping.")
            else:
                # Create a new binding
                new_binding = policy_pb2.Binding(
                    role=role_to_add,
                    members=[member_to_add]
                )
                policy.bindings.append(new_binding)
                print(f"DEBUG: Created new binding for {role_to_add} and {member_to_add}.")

            # 3d. Set the updated IAM policy (synchronous call assumed)
            self.bqclient.set_iam_policy(dataset_ref, policy)
            print("INFO: Successfully set dataset access to public (allUsers).")

        except Exception as e:
            # Catch errors during IAM update (e.g., lack of IAM admin permission)
            print(f"ERROR: Failed to update IAM policy for public access: {e}")



    def get_create_bq_table(self, table_name, query=None, ttype="node"):
        table_exists = self.bq_check_table_exists(table_name)
        print(f"{ttype} table {table_name}", table_exists)

        try:
            if not table_exists or table_exists is False or table_exists is None:
                print(f"🛠 Creating {ttype} Table: {table_name}")
                if query is None:
                    query = self.create_default_table_query(table_id=table_name, ttype=ttype)
                table:_EmptyRowIterator=self.run_query(query)
                print("schema", table.schema)
                print("pages", table.pages)
                print("total_rows", table.total_rows)
                return table
            else:
                print("Table already exists")
        except Exception as e:
            print("Error create_table", e)

    def bq_get_table_schema(self, table_name):
        if table_name:
            try:
                table = self.bqclient.get_table(self.get_table_name(table_name))
                schema = {field.name: field.field_type for field in table.schema}
                # print(f"table schema {table_name}", schema)
                print(f"schema received: {schema}")
                return schema
            except Exception as e:
                print("table name not exists", e)
                table=self.get_create_bq_table(
                    table_name=table_name,
                    ttype="edge" if any(c.islower() for c in table_name) else "node",
                )
                return table



    def update_bq_schema(self, table, rows):
        try:
            schema = self.bq_get_table_schema(table_name=table)
            print(f"Schema for {table}:{schema}")
            all_queries = []
            for r in rows:
                for k, v in r.items():
                    if schema is not None and k not in schema:
                        all_queries.append(self.add_col_query(
                            col_name=k,
                            table=table,
                            col_value=v
                        ))

            # print("all_queries:", all_queries)
            if len(all_queries):
                print("Update BQ schema")
                for query in all_queries:
                    self.run_query(query)
            print("finished update_bq_schema")
        except Exception as e:
            print("Err update_bq_schema:", e)






    def up2bq(self, table_id, csv_data, mode="o"):
        """
        Uploads CSV admin_data from a string variable to BigQuery, auto-detecting the schema.

        Args:
            table_id: The ID of the BigQuery table.
            csv_data: The CSV admin_data as a string.

        Returns:
            None. Raises an exception if an error occurs.
        """
        print("Update BQ ROWS")
        try:
            if mode == "a": # append
                write_disposition = bigquery.WriteDisposition.WRITE_APPEND
            elif mode == "o": # overwrite
                write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
            elif mode == "u": #unique
                write_disposition = bigquery.WriteDisposition.WRITE_EMPTY
            else:
                write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

            # 2. Construct the table reference
            table_ref = f"{GCP_ID}.{self.ds_id}.{table_id}"

            # 3. Create a Pandas DataFrame from the CSV admin_data
            df = pd.read_csv(io.StringIO(csv_data))  # Read csv from string

            # 4. Infer the schema from the DataFrame
            job_config = bigquery.LoadJobConfig(
                # Schema is auto-detected from Pandas DataFrame
                source_format=bigquery.SourceFormat.CSV,
                autodetect=True,  # Important for auto-schema detection
                write_disposition=write_disposition,  # Overwrite table if exists
                max_bad_records=50,  # Allow up to 50 bad rows before failing

                # Optionally set other load job config options
                # like field delimiters, skip_leading_rows, etc.
            )

            # 5. Load the DataFrame to BigQuery
            job = self.bqclient.load_table_from_dataframe(df, table_ref, job_config=job_config)  # Make an API request.
            job.result()  # Wait for the job to complete.

            print(f"Successfully loaded CSV admin_data to {table_ref}")

        except Exception as e:
            print(f"An error occurred: {e}")
            raise  # Re-raise the exception for proper error handling

    def get_layer_from_table_name(self, input_string):
        parts = input_string.split("-")
        return parts[1] if len(parts) > 1 else None

    def get_column_values(self, table: str, column: str) -> List[str]:
        query = f"SELECT DISTINCT {column} FROM `{self.pid}.{self.ds_id}.{table}`"
        results = self.run_query(query)
        return [row[column] for row in results]

    def id_mapping(self, rows, all_ids):
        print("")
        existing_rows=[]
        new_rows=[]
        for row in rows:
            if row["nid"] in all_ids:
                existing_rows.append(row)
            else:
                new_rows.append(row)
        return existing_rows, new_rows


    def ensure_table_exists(self, table_name: str,rows:list, ds_id=None):
        """
        Ensures a BigQuery table exists, creating it if necessary.
        :param table_name: The BigQuery table name.
        """
        if ds_id is None:
            ds_id = self.ds_id
        table_ref = f"{self.bqclient.project}.{ds_id}.{table_name}"

        schema = self.convert_dict_shema_bq(
            self.schema_from_dict(rows)
        )

        try:
            self.bqclient.get_table(table_ref)
        except NotFound:
            print(f"Table {table_name} not found, creating...")
            table = bigquery.Table(table_ref, schema=schema)
            try:
                self.bqclient.create_table(table)
            except Conflict:
                print(f"Table {table_name} already exists (Conflict).")

        except Exception as e:
            print(f"Erro ensure_table_exists:{e}")
        print(f"Table {table_ref} exists")
        return table_ref, schema


    def get_ds_ref(self, ds_id=None):
        ds_id = ds_id or self.ds_id
        return f"{self.pid}.{ds_id}"


    def bq_insert(self, table_id: str, rows: List[dict], upsert=False, ds_id=None):
        table_ref, schema = self.ensure_table_exists(table_id, rows, ds_id)

        if not isinstance(rows, list):
            rows = [rows]

        # 1. Flatten all complex fields to prevent "Array specified for non-repeated field"
        cleaned_rows = []
        for row in rows:
            clean_row = {}
            for k, v in row.items():
                # If the value is a list or dict, stringify it
                if isinstance(v, (list, dict)):
                    clean_row[k] = json.dumps(v)
                else:
                    clean_row[k] = v
            cleaned_rows.append(clean_row)

        self.update_bq_schema(table_id, cleaned_rows)

        # 2. Batch processing (50 rows at a time)
        batch_size = 50
        if len(cleaned_rows) > 0:
            for i in range(0, len(cleaned_rows), batch_size):
                batch_chunk = cleaned_rows[i:i + batch_size]

                if upsert:
                    query = self.upsert_row_query(table_id, rows=batch_chunk, schema=schema)
                    self.run_query(query)
                else:
                    errors = self.bqclient.insert_rows_json(table=table_ref, json_rows=batch_chunk)
                    if errors:  # Retry logic
                        print(f"Warning: Insert failed, retrying in 5s... Errors: {errors}")
                        time.sleep(5)
                        errors = self.bqclient.insert_rows_json(table=table_ref, json_rows=batch_chunk)
                        if errors:
                             print(f"ERROR: Failed to insert rows to {table_id} after retry: {errors}")
        else:
            print("No rows to process.")
        return True


    def insert_col(self, table_id: str, column_name: str, column_type: str):
        """
        Checks if a column exists in the BigQuery table. If not, it creates it.

        Args:
            project_id (str): Google Cloud Project ID.
            dataset_id (str): BigQuery Dataset ID.
            table_id (str): BigQuery Table ID.
            column_name (str): Name of the column to check/create.
            column_type (str): BigQuery admin_data type (e.g., STRING, INT64, FLOAT64).
        """
        print("insert col")
        table_ref = f"{self.pid}.{self.ds_id}.{table_id}"

        # Get table schema
        table = self.bqclient.get_table(table_ref)
        existing_columns = [field.name for field in table.schema]

        if column_name not in existing_columns:
            print(f"[WARN] Column '{column_name}' does not exist. Adding it...")
            alter_query = f"ALTER TABLE `{table_ref}` ADD COLUMN {column_name} {column_type}"
            self.bqclient.query(alter_query).result()
            print(f"[OK] Column '{column_name}' added successfully.")
        else:
            print(f"[OK] Column '{column_name}' already exists.")


    def list_tables(self) -> list:
        """Lists all tables in a BigQuery dataset.
        ALTER TABLE aixr-401704.QBRAIN.users ADD COLUMN sm_stack_status STRING
        Args:
            client: A BigQuery client instance.
            dataset_id: The ID of the dataset.

        Returns:
            A list of bigquery.Table objects, or an empty list if no tables are found
            or if an error occurs.  Returns None if an error occurs.
        """
        try:
            dataset_ref = self.bqclient.dataset(self.ds_id)  # API request
            tables = list(self.bqclient.list_tables(dataset_ref))  # API request
            table_names = [table.table_id for table in tables]
            return table_names
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


    def get_bq_type(self, value):
        """
        Determines the BigQuery type for a given Python value.
        :param value: The value to determine the type for.
        :return: A BigQuery-compatible admin_data type.
        """
        if value is None:
            return "STRING"
        if isinstance(value, bool):
            return "BOOL"
        if isinstance(value, int):
            return "INT64"
        if isinstance(value, float):
            return "FLOAT64"
        if isinstance(value, list):

            return f"ARRAY<{self.get_bq_type(value[0]) if len(value) else 'STRING'}>"

        return "STRING"



class BigQueryGraphHandler(BQCore):
    def __init__(self):
        """Initializes the BigQuery handler."""
        super().__init__()

    def upload_graph(self, graph):
        """
        Converts a NetworkX graph into CSV format (nodes & edges) and uploads it to BigQuery.

        :param graph: A NetworkX graph object.
        """

        # Ensure tables exist
        self.ensure_table_exists("nodes")
        self.ensure_table_exists("EDGES")

        # Convert to DataFrames
        nodes_df = self.graph_to_nodes_df(graph)
        edges_df = self.graph_to_edges_df(graph)

        # Ensure schema consistency
        self.check_add_fields(self.extract_schema(nodes_df), "nodes")
        self.check_add_fields(self.extract_schema(edges_df), "EDGES")

        # Upload to BigQuery
        self.upload_dataframe_to_bq(nodes_df, "nodes")
        self.upload_dataframe_to_bq(edges_df, "EDGES")


    def graph_to_edges_df(self, graph) -> pd.DataFrame:
        """
        Converts NetworkX edges to a Pandas DataFrame.

        :param graph: A NetworkX graph object.
        :return: Pandas DataFrame containing edges.
        """
        edge_data = []
        for src, tgt, attrs in graph.edges(data=True):
            row = dict(src=src, tgt=tgt, **attrs)
            new_row = {}
            for k, v in row.items():
                new_row[re.sub(r"\.", "_", k)] = v
            edge_data.append(new_row)
        return pd.DataFrame(edge_data)

    def extract_schema(self, df: pd.DataFrame) -> dict:
        """
        Extracts a schema from a Pandas DataFrame for BigQuery.
        :param df: The Pandas DataFrame.
        :return: Dictionary mapping column names to BigQuery types.
        """
        schema = {}
        for col in df.columns:
            sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            schema[col] = self.get_bq_type(sample_value)
        print("schema", schema)
        return schema

    def upload_dataframe_to_bq(self, df: pd.DataFrame, table_name: str):
        """
        Uploads a Pandas DataFrame to BigQuery.

        :param df: The DataFrame to upload.
        :param table_name: The BigQuery table name.
        """
        table_ref = f"{self.bqclient.project}.{self.ds_id}.{table_name}"
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            autodetect=True,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            max_bad_records=50,
        )
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)  # Reset buffer position

        job = self.bqclient.load_table_from_file(csv_buffer, table_ref, job_config=job_config)
        job.result()  # Wait for completion
        print(f"[OK] Uploaded {len(df)} rows to {table_name}.")

    def check_add_fields(self, schema: dict, table_name: str):
        """
        Ensures all required columns exist in BigQuery before uploading.

        :param schema: Expected schema dictionary {column_name: column_type}.
        :param table_name: The BigQuery table name.
        """
        existing_columns = self.get_column_names(table_name)
        new_columns = {col: col_type for col, col_type in schema.items() if col not in existing_columns}
        if new_columns:
            self.add_columns_bulk(table_name, new_columns)

    def get_column_names(self, table_name: str):
        """
        Retrieves column names from a BigQuery table.
        :param table_name: The BigQuery table name.
        :return: A set of column names.
        """
        table_ref = f"{self.bqclient.project}.{self.ds_id}.{table_name}"
        try:
            table = self.bqclient.get_table(table_ref)
            return {field.name for field in table.schema}
        except Exception as e:
            print(f"[WARN] Error retrieving columns for {table_name}: {e}")
            return set()





import typing
from typing import List, Optional
from google.cloud.bigquery import ScalarQueryParameter # Import for explicit parameter typing


class BigQueryRAG(BQCore):  # Inherit from BigQueryLoader if you have one
    """
    A class to perform RAG-like vector search using Google Cloud BigQuery.
    Mirrors the functionality of SpannerRAG for vector search.
    """

    def __init__(self, dataset: str or None = None):
        BQCore.__init__(self, dataset)
        self.project = GCP_ID
        self.base_path=f"{self.pid}.{self.ds_id}"


    def bigquery_vector_search(
        self,
        data: typing.Any, # Can be text for custom=False, or already embedded admin_data if custom=True
        table_id: str,
        custom: bool = True,
        limit: int = 10,
        select: List[str] = ["id"],
        embed_column: str = "embedding", # BigQuery terminology often uses 'column'
        model_name: Optional[str] = None # Required if custom=False
    ) -> List[typing.Dict]:
        """
        Performs a vector similarity search in a BigQuery table.

        Args:
            data: The query admin_data. If custom=True, this should be the pre-calculated
                  embedding vector (List[float]). If custom=False, this should
                  be the text admin_data (str) to be embedded by BQML.
            table_id: The ID of the BigQuery table containing the embeddings.
            custom: If True, uses a pre-calculated embedding from 'admin_data'.
                    If False, uses BQML's GENERATE_EMBEDDING to embed 'admin_data'.
            limit: The maximum number of results to return.
            select: A list of column names to select from the table,
                    in addition to the calculated distance.
            embed_column: The name of the column in the table that stores the embeddings.
            model_name: Required if custom=False. The name of the BQML model
                        or the model identifier string (e.g., 'text-embedding-004')
                        used for embedding the query text. This can be a full
                        `project.dataset.model` path or just the model ID if in
                        the same dataset.

        Returns:
            A list of dictionaries, where each dictionary represents a row
            with the selected columns and the cosine distance.
        """
        full_table_path = f"`{self.base_path}.{table_id}`" # Use backticks for table names

        query_parameters = []
        selected_columns_sql = ', '.join([f't.{col}' for col in select]) # Select columns from the table alias

        if custom:
            # Assume 'admin_data' is already the embedding vector (List[float])
            if not isinstance(data, list) or not all(isinstance(i, (int, float)) for i in data):
                 raise ValueError("When custom=True, 'admin_data' must be a list of numbers (embedding vector).")

            query = f"""
            SELECT
                {selected_columns_sql},
                COSINE_DISTANCE(
                    t.{embed_column},
                    @query_embedding
                ) AS distance
            FROM {full_table_path} AS t
            WHERE t.{embed_column} IS NOT NULL
            ORDER BY distance
            LIMIT @limit;
            """
            # BigQuery uses ARRAY<FLOAT64> for vector embeddings
            query_parameters.append(ScalarQueryParameter("query_embedding", "ARRAY<FLOAT64>", data))
            query_parameters.append(ScalarQueryParameter("limit", "INT64", limit))

        else:
            # Use BQML to generate the embedding for the query text 'admin_data'
            if not isinstance(data, str):
                 raise ValueError("When custom=False, 'admin_data' must be a string (query text).")
            if not model_name:
                 raise ValueError("When custom=False, 'model_name' must be provided.")

            # Use GENERATE_EMBEDDING function with the specified model
            # Join the main table with the result of the embedding function
            query = f"""
            SELECT
                {selected_columns_sql},
                COSINE_DISTANCE(
                    t.{embed_column},
                    embeddings.vector # The column name for the embedding vector output from GENERATE_EMBEDDING is 'vector'
                ) AS distance
            FROM {full_table_path} AS t,
            ML.GENERATE_EMBEDDING(MODEL `{self.project_id}.{self.dataset_id}.{model_name}`, # Reference the model
                (SELECT @query_text AS content) # Input admin_data as a STRUCT with 'content' field
            ) AS embeddings
            WHERE t.{embed_column} IS NOT NULL
            ORDER BY distance
            LIMIT @limit;
            """
            query_parameters.append(ScalarQueryParameter("query_text", "STRING", data))
            query_parameters.append(ScalarQueryParameter("limit", "INT64", limit))
            # Note: Depending on the model type (e.g., a remote model like text-embedding-004
            # versus a BQML trained model), the MODEL reference might just be
            # `model_name` if it's a public endpoint string or a model alias.
            # Using the full path `project.dataset.model_name` is safest for
            # models created within your BQML environment.

        print("Executing BigQuery SQL:")
        print(query)
        print("Parameters:", [(p.name, p.type, p.value) for p in query_parameters])

        query_job = self.bqclient.query(query, parameters=query_parameters)

        results = query_job.result()  # Waits for the job to complete.

        # Format the results similar to the Spanner output structure
        formated_results = []
        # BigQuery row objects can be accessed by index or name and converted to dict
        selected_cols_with_distance = select + ['distance'] # Include distance in the output dict keys

        for row in results:
            row_dict = {}
            # Map selected column names (and distance) to values from the row
            for col_name in selected_cols_with_distance:
                 # Access row values by name, falling back to index if name lookup fails
                 # (though for explicitly selected columns, name lookup should work)
                 try:
                     row_dict[col_name] = row[col_name]
                 except ValueError: # Fallback for cases where column name might not be directly available as attribute/key
                     # Find index by name if needed, or assume order matches select list + distance
                     # A more robust way is to use row._fields
                     field_names = [field.name for field in results.schema]
                     if col_name in field_names:
                          row_dict[col_name] = row[field_names.index(col_name)]
                     else:
                          row_dict[col_name] = None # Should not happen if query is correct

            formated_results.append(row_dict)


        print("Results", formated_results)
        return formated_results

    def create_embedding_model(
            self,
            model_id: str,
            connection_id: str,
            connection_location: str,
            replace: bool = True
    ):
        """
        Creates a BigQuery ML remote model that points to a Vertex AI
        embedding service endpoint.

        This allows you to use the model name (e.g., `project.dataset.model_id`)
        with BQML functions like `ML.GENERATE_EMBEDDING`.

        Args:
            model_id: The name to give the BQML model (e.g., 'text_embedding_model').
            connection_id: The ID of the Google Cloud connection resource. This
                           connection must be configured to connect to Vertex AI
                           and have necessary permissions.
            connection_location: The location of the connection resource (e.g., 'us-central1').
            replace: If True, uses CREATE OR REPLACE MODEL. If False, uses CREATE MODEL.
                     Defaults to True.

        Requires:
            - A Google Cloud Connection resource already created in the specified location.
            - The Connection must be linked to Vertex AI.
            - The BigQuery service account needs Vertex AI permissions
              (e.g., `Vertex AI User` role).
        """
        full_model_path = f"`{self.base_path}.{model_id}`"
        full_connection_path = f"`{self.pid}.{connection_location}.{connection_id}`"

        create_statement = "CREATE OR REPLACE MODEL" if replace else "CREATE MODEL"

        query = f"""
        {create_statement} {full_model_path}
        REMOTE WITH CONNECTION {full_connection_path}
        OPTIONS (remote_service_type = 'CLOUD_AI_EMBEDDING');
        """
        print(f"Executing BigQuery SQL to create model {full_model_path}:")
        print(query)

        query_job = self.bqclient.query(query)

        # Wait for the job to complete
        query_job.result()

        print(f"BigQuery ML model {full_model_path} created successfully.")
        return f"BigQuery ML model {full_model_path} created successfully."




















# Example Usage (requires Google Cloud authentication and a BigQuery table with embeddings)
if __name__ == '__main__':
    # !!! IMPORTANT !!!
    # Replace 'your-gcp-project-id' and 'your_dataset_id' with your actual IDs
    # Also, make sure you have a table named 'your_embeddings_table'
    # with a column named 'embed' (ARRAY<FLOAT64>) and an 'id' column.
    # If using custom=False, make sure you have a BQML model named 'your_embedding_model'
    # or provide the appropriate model_name/path.
    try:
        bq_rag = BigQueryRAG(project_id='your-gcp-project-id', dataset_id='your_dataset_id')

        # --- Example with custom embedding (embedding done outside BQ) ---
        # You would replace this placeholder vector with the actual embedding
        # generated from your query text using your embedding model.
        # dummy_query_embedding = embed("This is a test query") # Call your actual embed function
        # print("Running custom embedding search...")
        # results_custom = bq_rag.bigquery_vector_search(
        #     admin_data=[0.01] * 768, # Replace with actual vector from embed("Your query text")
        #     table_id='your_embeddings_table',
        #     custom=True,
        #     limit=5,
        #     select=["id", "another_column"] # Example of selecting multiple columns
        # )
        # print("Custom Search Results:", results_custom)

        # --- Example with BQML embedding (embedding done inside BQ) ---
        print("\nRunning BQML embedding search...")
        results_bqml = bq_rag.bigquery_vector_search(
            data="This is another test query for BQML.",
            table_id='your_embeddings_table',
            custom=False,
            limit=5,
            select=["nid"],
            model_name='your_embedding_model' # Replace with your BQML model name/path
        )
        print("BQML Search Results:", results_bqml)

    except NotImplementedError as e:
         print(f"Error: {e}. Please implement the 'embed' function or provide a valid BQML model.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your GCP project, dataset, table, and potentially BQML model are correctly configured and accessible.")



if __name__ == "__main__":
    v=BQGroundZero()
    print(v.run_query(query="SELECT 1"))


"""
def schema_from_dict(self, rows:list, embed=None):
    schema = []
    for data in rows:
        for key, value in data.items():
            print("VALUE", value, type(value))
            if isinstance(value, int):
                field_type = "INTEGER"
            elif isinstance(value, float):
                field_type = "FLOAT"
            elif isinstance(value, bool):
                field_type = "BOOLEAN"
            elif isinstance(value, list) and embed:
                for i in value:
                    print("it", type(i))
                field_type = "ARRAY<NUMERIC>"
            else:
                field_type = "STRING"
            s = bigquery.SchemaField(key, field_type, mode="NULLABLE")
            if s not in schema:
                schema.append(s)
    return schema

upsert query def



    def upsert_row_query(self, table_id: str, rows: list[dict], schema: dict) -> str:
        print("Upsert to", table_id)

        # Create a string representation of the rows as BigQuery STRUCTs
        # Example: STRUCT(1 AS id, "value1" AS col1), STRUCT(2 AS id, "value2" AS col1)
        struct_rows = []
        for row in rows:
            val_strs = []
            for k, v in row.items():
                # Handle different admin_data types for correct BigQuery literal representation
                if isinstance(v, str):
                    val_strs.append(f'"{v}" AS {k}')  # Use double quotes for strings
                elif isinstance(v, (int, float)):
                    val_strs.append(f'{v} AS {k}')
                elif v is None:
                    val_strs.append(f'NULL AS {k}')
                else:
                    # Attempt a generic representation, might need more specific handling
                    val_strs.append(f'{repr(v)} AS {k}')

            struct_str = ", ".join(val_strs)
            struct_rows.append(f"STRUCT({struct_str})")

        # Construct the USING clause with UNNESTing the array of STRUCTs
        # BigQuery expects the UNNEST to be the source of the MERGE
        unnested_source = 
            (SELECT * FROM UNNEST([
                {', '.join(struct_rows)}
            ]))
      
        # Construct the UPDATE SET clause
        # This updates existing rows with values from the source, prioritizing source values
        update_clause_parts = []
        for k in schema.keys():
            # Use S.<column_name> for values from the source (unnested rows)
            update_clause_parts.append(f"T.{k} = S.{k}")
        update_clause = ",\n          ".join(update_clause_parts)  # Indent for readability

        # Construct the INSERT clause
        # This inserts new rows from the source
        insert_cols = ", ".join(schema.keys())
        # Use S.<column_name> for values to be inserted
        insert_vals = ", ".join(f"S.{col}" for col in schema.keys())

        # Construct the full MERGE statement
        # The alias 'S' is applied to the result of the UNNEST subquery
        query = 
            MERGE INTO `{self.pid}.{self.ds_id}.{table_id}` T
            USING {unnested_source} AS S
            ON T.nid = S.nid
            WHEN MATCHED THEN
              UPDATE SET
                {update_clause}
            WHEN NOT MATCHED THEN
              INSERT ({insert_cols}) VALUES ({insert_vals})
        return query.strip()

    def get_parent(self, table:str):
        return f"projects/{self.pid}/datasets/{self.ds_id}/tables/{table}"


    def bq_insert(self, table_id: str, rows: List[dict], upsert=False, ds_id=None):
        table_ref, schema = self.ensure_table_exists(table_id, rows, ds_id)

        # stringify rows
        for row in rows:
            if "parent" in row and isinstance(row["parent"], (dict, list)):
                row["parent"] = json.dumps(row["parent"])

        self.update_bq_schema(table_id, rows)

        if not isinstance(rows, list):
            rows = [rows]  # if just Single dict
        if len(rows) > 0:
            print(f"Insert {len(rows)} rows")
            for i in range(0, len(rows), self.batch_upload_size):
                batch_chunk = rows[i:i + self.batch_upload_size]
                print("Insert batch rows")
                if upsert is True:
                    query = self.upsert_row_query(
                        table_id, rows=batch_chunk, schema=schema
                    )
                    self.run_query(query)
                else:
                    print(f"insert {batch_chunk} to {table_ref}")
                    result = self.bqclient.insert_rows_json(table=table_ref, json_rows=batch_chunk)
                    print("result", result)
                    if len(result):
                        print("Failed insertion:")
                        time.sleep(15)
                        result=self.bqclient.insert_rows_json(table=table_ref, json_rows=batch_chunk)
                        print(f"second try result: {result}")
        else:
            print("No new rows to insert")


"""