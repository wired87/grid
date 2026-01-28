import jax
from jax import vmap, jit
# todo _BYTE binary_view = ' '.join(format(b, '08b') for b in text.encode('utf-8'))
from gnn.db_layer import DBLayer
from gnn.injector import InjectorLayer
from jax_utils.conv_flat_to_shape import bring_flat_to_shape
from mod import Node
from utils import SHIFT_DIRS, create_runnable

import jax.numpy as jnp


class GNN:

    def __init__(
            self,
            amount_nodes,
            time:int,
            gpu,
            DIMS,
            **cfg
    ):
        # DB_TO_METHOD_EDGES
        # METHOD_PARAM_LEN_CTLR
        # METHOD_SHAPES
        for k, v in cfg.items():
            setattr(self, k, v)

        self.time = time
        self.amount_nodes = amount_nodes
        # Generate grid coordinates based on amount_nodes dimensionality
        self.schema_grid = [
            (i,i,i)
            for i in range(amount_nodes)
        ]

        self.len_params_per_methods = {}
        self.change_store = []

        self.gpu = gpu

        self.db_layer = DBLayer(
            amount_nodes,
            self.gpu,
            DIMS=DIMS,
            **cfg
        )

        self.injector = InjectorLayer(
            db_layer=self.db_layer,
            amount_nodes=amount_nodes,
            DIMS=DIMS,
            **cfg
        )

        # features jsut get etend naturally to the arr
        self.model_skeleton = jnp.array([])
        print("Node initialized and build successfully")


    def main(self):
        self.build_gnn_equation_nodes()

        self.simulate()

        serialized = self.serialize(self.model_skeleton)
        print("serialized model_skeleton", serialized)

        jax.debug.print("process finished.")

    def simulate(self):
        try:
            for step in range(self.time):
                jax.debug.print(
                    "Sim step {s}/{n}",
                    s=step,
                    n=self.time
                )

                if step in self.injector.time:
                    # apply injections inside db layer
                    self.injector.inject(
                        idx=self.injector.time.index(step),
                        step=step,
                        db_layer=self.db_layer
                    )

                # get params from DB layer and init calc
                self.calc_batch()

        except Exception as e:
            jax.debug.print(f"Err simulate: {e}")

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

    def get_mod_idx(self, abs_eq_idx):
        # get mod_idx based on eq_idx
        """total = 0
        for i, amount_mthods in enumerate(self.METHODS):
            total += amount_mthods
            if total >= abs_eq_idx:
                return i
        """
        # 1,2 -> 1,3
        cum = jnp.cumsum(jnp.array(self.MODULES))  # kumulative Summe
        # finde erstes i, bei dem cum >= abs_eq_idx
        idx = jnp.argmax(cum >= abs_eq_idx)
        return idx

    def calc_batch(self):
        jax.debug.print("calc_batch...")
        # calc all methods and apply result to new g
        all_results = []

        # START LOOP
        node: Node
        for eq_idx, node in enumerate(self.METHODS):
            mod_idx = self.get_mod_idx(eq_idx)
            print("mod_idx", mod_idx)

            """ 
            # extract values
            [[ 0,  7,  6],
            [ 0,  7,  7],
            [ 3,  0,  9],
            [ 3,  0, 10],
            [ 3,  0,  0]],
            """
            variations, amount_params = self.extract_eq_variations(
                mod_idx,
                eq_idx,
            )


            examle_variation = variations[0]
            all_ax, all_shape = self.db_layer.get_axis_shape(examle_variation)


            # transform shape along vertical ax
            # # # #
            # a 1
            # b 2
            #variations = variations.T
            transformed = [[] for _ in range(amount_params)]
            for i in range(amount_params):
                for var in variations:
                    transformed[i].append(var[i])
            transformed = jnp.array(transformed)
            print("transformed", transformed)


            # get flatten params for all variations
            # # # #
            # a 1
            # b 2
            flatten_transformed = []
            for param_grid in transformed:
                # problem: jax cant handle dynamic shapes...
                # param_grid
                new_grid = []
                for item in param_grid:
                    new_grid.extend(
                        self.db_layer.extract_flattened_grid(item)
                    )
                flatten_transformed.append(new_grid)

            #
            inputs = []
            for shape, variation_grids in zip(all_shape, flatten_transformed):
                print("variation_grids", variation_grids)
                result = bring_flat_to_shape(
                    jnp.array(variation_grids),
                    shape
                )
                inputs.append(jnp.array(result))

            # calc single equation
            features, results = node(
                grid=inputs,
                in_axes_def=tuple(all_ax),
            )

            print("features, results", features, results)

            # append all features to history struct
            jnp.stack([self.model_skeleton, features])

        # todo sort values innto eq spec grids
        self.db_layer.sort_results(all_results)
        jax.debug.print("calc_batch... done")




    def shape_variations(self, variation_grids):
        # todo build shape struct method based  meshgrid
        return



    def sort_features(self, eq_idx, all_features):
        len_eq_variations = self.ITERATORS["eq_variations"][eq_idx]
        start_idx = jnp.sum(self.ITERATORS["eq_variations"])[:eq_idx]

        indices = jnp.array(start_idx+i for i in range(len_eq_variations))
        # FEATURE -> MODEL
        self.model_skeleton.at[
            tuple(indices.T)
        ].add(all_features)



    def extract_eq_variations(self, mod_idx, eq_idx):
        """
        Extrahiert alle Variationen für ein spezifisches Gleichung.
        Navigiert durch DB und AXIS und skaliert Parameter bei axis == 0 auf amount_nodes.
        """
        jax.debug.print("📊 Extracting Variations for Mod {m} Eq {e}", m=mod_idx, e=eq_idx)

        start_sum = 0
        for i, (amount_variations, amount_params) in enumerate(
            zip(
                self.DB_CTL_VARIATION_LEN_PER_EQUATION[:eq_idx],
                self.METHOD_PARAM_LEN_CTLR[:eq_idx]
            )
        ):
            start_sum += amount_variations * amount_params

        amount_params_current_eq = self.METHOD_PARAM_LEN_CTLR[eq_idx]
        print("amount_params_current_eq", amount_params_current_eq)

        # get len of variations per equation
        amount_variations_current_eq = self.DB_CTL_VARIATION_LEN_PER_EQUATION[eq_idx]
        total_amount_params_current_eq = amount_params_current_eq * amount_variations_current_eq
        print("total_amount_params_current_eq", total_amount_params_current_eq)

        # todo must multiply each item in self.DB_CTL_VARIATION_LEN_PER_EQUATION[:eq_idx] * amount_params for specific equation
        slice = jax.lax.dynamic_slice_in_dim(
            jnp.array(self.DB_TO_METHOD_EDGES),
            start_sum,
            total_amount_params_current_eq,
        )
        print("slice", slice)

        # get chunks len
        num_chunks = total_amount_params_current_eq // amount_params_current_eq

        # Truncate extras if len(receive) is not divisible by n
        receive_truncated = slice[:num_chunks * amount_params_current_eq]

        # Reshape in (num_chunks, n, len(inner_list))
        result = receive_truncated.reshape(
            num_chunks,
            amount_params_current_eq,
            -1
        )
        print("result", result)
        return result, amount_params_current_eq



    def build_gnn_equation_nodes(self):
        # create equation_nodes
        for eq_idx, eq in enumerate(self.METHODS):
            node = Node(
                runnable=create_runnable(
                    eq_code=eq,
                ),
                #method_id=eq_idx,
            )
            # replace mehod str with py class
            self.METHODS[eq_idx] = node


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


