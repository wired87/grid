
import os

import jax
from jax import vmap, jit
# todo _BYTE binary_view = ' '.join(format(b, '08b') for b in text.encode('utf-8'))
from gnn.db_layer import DBLayer
from gnn.injector import InjectorLayer
from mod import Node
from utils import SHIFT_DIRS, create_runnable

import jax.numpy as jnp


class GNN:

    def __init__(
            self,
            INJECTIONS,
            amount_nodes,
            time:int,
            METHODS,
            DB,
            ITERATORS,
            gpu,
            method_to_db,
            db_to_method,
            DIMS,
    ):
        self.INJECTIONS = INJECTIONS
        self.METHODS = METHODS
        self.ITERATORS = ITERATORS
        self.DIMS=DIMS
        self.time = time
        self.amount_nodes = amount_nodes

        # Generate grid coordinates based on amount_nodes dimensionality
        self.schema_grid = [
            (i,i,i)
            for i in range(amount_nodes)
        ]

        self.db_to_method = db_to_method
        self.len_params_per_methods = {}
        self.change_store = []

        self.gpu = gpu

        self.db_layer = DBLayer(
            amount_nodes,
            self.gpu,
            ITERATORS["modules"],
            method_to_db,
            db_to_method,
            **DB,
            DIMS=int(os.environ.get("DIMS")),
            fields=ITERATORS["fields"],
        )

        self.injector = InjectorLayer(
            **INJECTIONS,
            db_layer=self.db_layer,#
            DIMS=self.DIMS,
            amount_nodes=amount_nodes

        )

        # features jsut get etend naturally to the arr
        self.model_skeleton = jnp.array([])
        print("Node initialized and build successfully")


    def main(self):
        self.build_gnn()
        self.simulate()

        serialized = self.serialize(self.model_skeleton)

        jax.debug.print(
            "serialized \n{s}",
            s=serialized,
            n=self.time
        )

        jax.debug.print("process finished.")

    def simulate(self):
        try:
            for step in range(self.time):
                jax.debug.print(
                    "Sim step {s}/{n}",
                    s=step,
                    n=self.time
                )

                # apply injections inside db layer
                self.injector.inject_process(
                    step=step,
                    db_layer=self.db_layer
                )

                # get params from DB layer and init calc
                self.calc_batch()

        except Exception as e:
            jax.debug.print(f"Err simulate: {e}")
            raise
        jax.debug.print(
            "t={n}... done",
            n=self.time
        )



    @jit
    def inject(self, step, db_layer):
        """
        Applies injections based on self.inj_pattern and current step.
        Supports the SOA structure: [module, field, param, node_index, schedule]
        where schedule is list of [time, value].
        """
        all_indices = []
        all_values = []

        for item in self.injection_pattern:
            if isinstance(item, list) and len(item) == 5 and isinstance(item[4], list):
                # SOA structure: [mod, field, param, node_idx, schedule]
                mod_idx, field_idx, param_idx, node_idx, schedule = item

                # Check schedule for current time
                for time_point, value in schedule:
                    if time_point == step:
                        # Add to update list
                        # Assuming feature index 0
                        all_indices.append([mod_idx, field_idx, param_idx, node_idx])# rm 0?
                        all_values.append(value)

        if not all_indices:
            return

        all_indices = jnp.array(all_indices)  # shape [N, 5]
        all_values = jnp.array(all_values)  # shape [N]

        # inject step
        db_layer.nodes.at[
            tuple(all_indices.T)
        ].add(all_values)




    def _workflow(self):
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


    def calc_batch(self):
        jax.debug.print("calc_batch...")
        # calc all methods and apply result to new g
        all_results = []
        for mod_idx, amount_equations in enumerate(
                self.ITERATORS["modules"]):
            # START LOOP
            for eq_idx in range(amount_equations):
                node_idx = self.get_equation(
                    mod_idx=mod_idx,
                    eq_idx=eq_idx,
                )
                node:Node = self.METHODS[node_idx]

                # extract values
                variations = self.extract_field_variations(
                    mod_idx,
                    eq_idx,
                )

                # get params for all variations
                param_kernel = vmap(
                    self.db_layer.extract_field_param_variation,
                    in_axes=0
                )
                params, axis_def = param_kernel(variations)

                # calc single equation
                features, results = node(
                    self.db_layer.old_g,
                    params,
                    axis_def,
                )

                # append all features to final struct
                jnp.stack([self.model_skeleton, features])
        self.db_layer.sort_results(all_results)
        jax.debug.print("calc_batch... done")


    def sort_features(self, eq_idx, all_features):
        len_eq_variations = self.ITERATORS["eq_variations"][eq_idx]
        start_idx = jnp.sum(self.ITERATORS["eq_variations"])[:eq_idx]

        indices = jnp.array(start_idx+i for i in range(len_eq_variations))
        # FEATURE -> MODEL
        self.model_skeleton.at[
            tuple(indices.T)
        ].add(all_features)





    def extract_field_variations(self, mod_idx, eq_idx):
        """
        Extrahiert alle Variationen für ein spezifisches Feld/Gleichung.
        Navigiert durch DB und AXIS und skaliert Parameter bei axis == 0 auf amount_nodes.
        """
        jax.debug.print("📊 Extracting Variations for Mod {m} Eq {e}", m=mod_idx, e=eq_idx)

        # sum amount equations till current + all their fields
        variation_start = sum(self.ITERATORS["fields"][:mod_idx]) + sum(self.iterator_skeleton["module"][:mod_idx])

        # get eq len
        variation_len = self.ITERATORS["fields"][mod_idx]

        variation_end = variation_start + variation_len

        # get variation block
        variations = self.db_to_method[variation_start:variation_end]
        return variations



    def get_equation(self, mod_idx, eq_idx):
        # retirn index from methdod sruct to gather either Node or str eq
        def_idx = jnp.sum(
            jnp.array(self.ITERATORS["modules"][:mod_idx])
        ) + eq_idx
        return int(def_idx)




    def build_gnn(self):
        for mod_idx, amount_equations in enumerate(
                self.ITERATORS["modules"]
        ):
            #
            for eq_idx in range(amount_equations):
                #
                def_idx = self.get_equation(
                    mod_idx, eq_idx)

                node = Node(
                    runnable=create_runnable(
                        eq_code=self.METHODS[def_idx],
                    ),
                    method_id=eq_idx,
                    mod_idx=mod_idx,
                )

                # replace mehod str with py class
                self.METHODS[def_idx] = node


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


