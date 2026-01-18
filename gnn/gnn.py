"""

Update rocess
inject nodes
identify nodes e > 0
create G copy
"""
import json
import os

from flax import nnx

import jax
from dtypes import Graph
from gnn.db_layer import DBLayer
from gnn.injector import InjectorLayer
from mod import Node
from utils import SHIFT_DIRS, create_runnable

try:
    from safetensors.numpy import save_file as safetensors_save
except Exception:
    safetensors_save = None

import jax.numpy as jnp


class GNN(
    nnx.Module
):
    # todo pattern applieance and converter -< data ar request,
    # build stabel update loop

    """
    zeit

    Universal Actor for all fields to process a batch of Fields same type)
    With a GPU

    Design the system so each method has its own preprocessor (CPU)
    and processor (GPU)

    receive struct:
    [
    soa
    ],

    [
    [runnables]
    [(soa indices for params)]
    }

    WF:
    Hold all data on gpu but calc diferentyl
    pattern must come from cpu

    todo cache results to index
    """

    def __init__(
            self,
            amount_nodes,
            nodes_db,
            inj_pattern:list[int],
            modules_len,
            glob_time:int,
            chains,
            gnn_data_struct:list,
            db_layer,
    ):
        injection_pattern = os.getenv("INJECTION_PATTERN")
        self.injection_pattern = json.loads(injection_pattern)

        self.global_utils = None
        self.gnn_data_struct=gnn_data_struct
        self.index = 0

        self.db_layer = db_layer

        self.glob_time = glob_time
        self.nodes_db = nodes_db
        self.amount_nodes = amount_nodes

        self.world = []
        self.store = []
        self.module_map = []

        self.model = []

        # Generate grid coordinates based on amount_nodes dimensionality
        self.schema_grid = [
            (i,i,i)
            for i in range(amount_nodes)
        ]

        self.method_params = {}
        self.change_store = []

        self.modules_len=modules_len
        self.inj_pattern = inj_pattern

        self.inj_layer = InjectorLayer()
        self.db_layer = DBLayer(self.amount_nodes)

        method_struct = os.getenv("METHOD_LAYER")
        self.method_struct = json.loads(method_struct)

        def_out_db = os.getenv("METHOD_OUT_DB")
        self.def_out_db = jnp.array(json.loads(def_out_db))

        feature_out_gnn = os.getenv("FEATURE_OUT_GNN")
        self.feature_out_gnn = jnp.array(json.loads(feature_out_gnn))

        print("Node initialized and build successfully")



    def build_model(self):
        model = [
            # INJECTIOIN -> to get directly inj pattern (todo switch throguh db mapping)
            # DB SCHEMA
            self.db_layer.db_pattern,

            # EDGE DB -> METHOD
            self.method_struct,

            # FEATURES
            self.model_skeleton,

            # FEATURE DB
            self.def_out_db
        ]
        return model



    def main(self):
        self.build_gnn()
        self.simulate()
        model = self.build_model()
        serialized = self.serialize(model)
        self.serialized_model = serialized


    def simulate(self, steps: int = None):

        self.gnn_skeleton()

        try:
            for step in range(steps):
                jax.debug.print(
                    "Sim step {s}/{n}",
                    s=step + 1,
                    n=steps
                )

                # apply injections inside db layer
                self.inj_layer.inject(
                    step=step,
                    db_layer=self.db_layer
                )

                # prepare gs
                self.db_layer.process_time_step()

                # perform calc step
                self.process_t_step()

        except Exception as e:
            jax.debug.print(f"Err simulate: {e}")
            raise




    def process_t_step(self, old_g):
        # includes actual params ready to calc
        # todo später: behalte features und patterns in den nodes -> transferier alles am ende in einem mal in X struktur
        #
        param_struct = self.grab_db_to_method(old_g)
        # get params from DB layer and init calc
        all_features, all_results = self.calc_batch(param_struct)

        # finish step
        self.finish_step(
            all_features,
            all_results,
        )
        pass

    def calc_batch(self, param_struct):
        # calc all methods and apply result to new g

        all_features = []
        all_results = []

        for i, module in enumerate(self.method_struct):
            for def_idx, method_node in enumerate(module):

                methods_values = param_struct[i][def_idx]

                # calc single equation
                features, results = method_node(
                    self.db_layer.old_g,
                    methods_values,
                )

                # apply features
                for fidx, feild_features in enumerate(features):
                    all_features = jnp.column_stack(
                        (
                            all_features,
                            feild_features
                        )
                    )

                # result -> db
                for fidx, field_result in results:
                    all_results.append(field_result)

        return all_features, all_results



    def grab_db_to_method(self, old_g):
        jax.debug.print("grab_db_to_method...")
        param_struct = []
        for i, module in enumerate(self.def_out_db):
            for def_idx, method_node in enumerate(module):
                for fidx, field_patterns in enumerate(method_node):

                    # grab values pathways stored
                    for pattern in field_patterns:
                        # inject step
                        params = [
                            old_g[item]
                            for item in pattern
                        ]

                        param_struct[i][def_idx][fidx].append(params)

        jax.debug.print("grab_db_to_method... done")
        return param_struct


    def gnn_skeleton(self):
        # SET EMPTY STRUCTURE OF MODEL
        model_skeleton = []
        for i, module in enumerate(self.def_out_db):
            model_skeleton[i] = []

            for j, method_struct in enumerate(module):
                model_skeleton[i].append(
                    []  #
                )
                field_block_matrice = []
                for adj_entry in method_struct:
                    # IMPORTANT!
                    # each
                    # adj_entry = maps different kind of
                    # interaction

                    feature_complex = []

                    # add field blcok struct
                    field_block_matrice.append(
                        feature_complex
                    )

                # ADD FIELD BLOCK STRUCT TO MODEL
                model_skeleton[i][j].append(
                    field_block_matrice
                )

        self.model_skeleton = jnp.array(model_skeleton)


    def build_gnn(self):
        for i, module in enumerate(self.method_struct):
            for j, method_struct in enumerate(module):
                node = Node(
                    runnable=create_runnable(method_struct[0]),
                    inp_edge_map=[inp[0] for inp in method_struct],
                    outp_pattern=[inp[1] for inp in method_struct],
                    in_axes_def=method_struct[2],
                    method_id=j,
                    mod_idx=i,
                )

                # overwrite raw method entry
                self.gnn[i][j] = node


    def finish_step(
            self,
            all_features,
            all_results,
    ):
        # FEATURE -> MODEL
        self.model_skeleton.at[
            tuple(self.feature_out_gnn.T)
        ].add(all_features)

        # RESULT -> HISTORY DB todo: upsert directly to bq
        self.db_layer.history_db.at[
            tuple(self.def_out_db.T)
        ].add(all_results)

        # RESULT -> DB
        self.db_layer.nodes = self.db_layer.nodes.at[
            tuple(self.def_out_db.T)
        ].set(all_results)


    def serialize(self, data):
        import flax.serialization
        # Wandelt den State in einen Byte-String um
        binary_data = flax.serialization.to_bytes(data)
        return binary_data


    def set_shift(self, start_pos:list[tuple]=None):
        """
        Calculates neighboring node indices based on SHIFT_DIRS using pure-Python coordinate addition 
        to avoid JAX type issues during initialization.
        """
        if start_pos is None:
            # If no starting positions are provided, we initialize for all nodes in the grid
            start_pos = self.schema_grid
            
        next_index_map = []
        for pos in start_pos:
            # SHIFT_DIRS[0] + SHIFT_DIRS[1] combines positive and negative directions
            for d in (SHIFT_DIRS[0] + SHIFT_DIRS[1]):
                # Pure python addition: tuple zip sum
                neighbor_pos = tuple(a + b for a, b in zip(pos, d))
                
                if neighbor_pos in self.schema_grid:
                    next_index_map.append(
                        self.schema_grid.index(neighbor_pos)
                    )
            # include the node itself
            next_index_map.append(
                self.schema_grid.index(pos)
            )
        return next_index_map


    def get_index(self, pos: tuple) -> int:
        """Returns the integer index of a grid position."""
        try:
            return self.schema_grid.index(pos)
        except ValueError:
            # Fallback or error handling
            return -1




    def set_next_nodes(self, pos_module_field_node):
        for module in pos_module_field_node:
            for field_pos_list in module:
                field_pos_list = self.set_shift(field_pos_list)
        return pos_module_field_node


    def get_sdr_rcvr(self):
        receiver = []
        sender = []
        for i, item in enumerate(range(len(self.store))):
            for j, eitem in enumerate(self.edges):
                sender.extend([i for _ in range(len(self.edges))])
                receiver.extend(eitem)
        jax.debug.print("set direct interactions finsihed")
        return sender, receiver


