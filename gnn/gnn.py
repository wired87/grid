import jax
from jax import jit, vmap
# todo _BYTE binary_view = ' '.join(format(b, '08b') for b in text.encode('utf-8'))
from gnn.db_layer import DBLayer
from gnn.feature_encoder import FeatureEncoder
from gnn.injector import InjectorLayer
from jax_utils.conv_flat_to_shape import bring_flat_to_shape
from mod import Node
from utils import SHIFT_DIRS, create_runnable, debug_callable
from flax import nnx

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

        self.model_feature_dims = 64

        self.time = time
        self.amount_nodes = amount_nodes
        # Generate grid coordinates based on amount_nodes dimensionality
        self.schema_grid = [
            (i,i,i)
            for i in range(amount_nodes)
        ]
        self.feature_encoder = FeatureEncoder()
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

        self.all_axs = []
        self.all_shapes = []

        print("Node initialized and build successfully")

    def prepare(self):
        self.build_gnn_equation_nodes()

        # DB
        self.db_layer.build_db(
            self.amount_nodes,
        )

        # layer to define all ax et
        self.step_0()


    def main(self):
        self.prepare()
        self.simulate()

        jnp.stack([
            self.feature_encoder.in_ts,
            self.feature_encoder.out_ts,
        ])

        serialized = self.serialize(
            self.feature_encoder.in_ts
        )
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

                """
                
                if step in self.injector.time:
                    # apply injections inside db layer
                    self.injector.inject(
                        idx=self.injector.time.index(step),
                        step=step,
                        db_layer=self.db_layer
                    )
                
                """

                # get params from DB layer and init calc
                all_ins, all_results = self.calc_batch()

                self.feature_encoder(
                    all_ins,
                    all_results,
                    #self.all_shapes + [out_shapes],
                )

                self.db_layer.save_t_step(all_results)
                # todo just save what has changed - not the entire array
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





    def short_transformed(self, all_ax, transformed):
        new_t = []
        for ax, st in zip(all_ax, transformed):
            # Statisches Python-if: JAX sieht nur das Ergebnis
            if ax == 0:
                new_t.append(st)  # Behält Shape (12, 4)
            else:
                new_t.append(st[0])  # Reduziert auf Shape (4,)
        return new_t


    def get_mod_idx(self, abs_eq_idx):
        # get mod_idx based on eq_idx
        # 1,2 -> 1,3
        cum = jnp.cumsum(jnp.array(self.MODULES))  # kumulative Summe
        # finde erstes i, bei dem cum >= abs_eq_idx
        idx = jnp.argmax(cum >= abs_eq_idx)
        return idx


    def step_0(self):
        print("step_0...")
        for eq_idx, node in enumerate(self.METHODS):
            #mod_idx = self.get_mod_idx(eq_idx)

            variations, amount_params = self.extract_eq_variations(
                eq_idx,
            )

            examle_variation = variations[0]
            axs, shapes = self.db_layer.get_axis_shape(examle_variation)

            self.all_axs.append(axs)
            self.all_shapes.append(shapes)
        print("step_0... done")





    def build_projections(self):
        self.projection_shapes = []

        # extend
        for i in range(len(self.all_shapes)):
            self.projection_shapes.append(
                [
                    *self.all_shapes[i],
                    self.db_layer.OUT_SHAPES[i]
                ]
            )

        return [
            [
                nnx.Linear(
                    in_features=shape,
                    out_features=self.model_feature_dims,
                    rngs=nnx.Rngs
                )
                for shape in shapes
            ]
           for shapes in self.projection_shapes
        ]


    def calc_batch(self):
        jax.debug.print("calc_batch...")
        # calc all methods and apply result to new g
        all_results = []
        all_ins = []
        # START LOOP
        node: Node
        for eq_idx, node in enumerate(self.METHODS):
            mod_idx = self.get_mod_idx(eq_idx)

            # get flatten params for all variations
            # # # #
            # a b
            # 1 2
            variations, amount_params = self.extract_eq_variations(
                eq_idx,
            )

            # transform shape along vertical ax
            # # # #
            # a 1
            # b 2
            transformed = jnp.transpose(variations, (1, 0, 2))
            #print("transformed", transformed)

            transformed = self.short_transformed(
                self.all_axs[eq_idx],
                transformed,
            )
            #print("shortened transformed", transformed)

            flatten_transformed = []
            for i, (param_grid, ax) in enumerate(
                zip(
                    transformed,
                    self.all_axs[eq_idx]
                )
            ):
                # problem: jax cant handle dynamic shapes...
                # param_grid

                single_param_grid = []

                #print("param_grid", param_grid)
                if ax == 0:
                    for coords in param_grid:
                        #print("coords", coords)
                        coord_result = self.db_layer.extract_flattened_grid(coords)
                        #print(f"coord_result for {i} {coords}", len(coord_result))
                        single_param_grid.extend(
                            coord_result
                        )
                else:
                    coord_result = self.db_layer.extract_flattened_grid(param_grid)
                    #print(f"coord_result for {i} {param_grid}", len(coord_result))
                    single_param_grid.extend(
                        coord_result
                    )

                #print("single_param_grid", single_param_grid, len(single_param_grid))
                flatten_transformed.append(single_param_grid)


            # reshape flattened collection
            inputs = []
            for shape, variation_grids in zip(self.all_shapes[eq_idx], flatten_transformed):
                #print("variation_grids", variation_grids)
                result = bring_flat_to_shape(
                    jnp.array(variation_grids),
                    shape
                )
                result = jnp.array(result)
                inputs.append(jnp.array(result))

            # calc single equation
            results = node(
                grid=inputs,
                in_axes_def=tuple(self.all_axs[eq_idx]),
            )

            all_results.append(results)
            all_ins.append(inputs)

        # todo sort values innto eq spec grids
        jax.debug.print("calc_batch... done")
        return all_ins, all_results




    def extract_eq_variations(self, eq_idx):
        """
        Extrahiert alle Variationen für ein spezifisches Gleichung.
        Navigiert durch DB und AXIS und skaliert Parameter bei axis == 0 auf amount_nodes.
        """
        jax.debug.print("extract_eq_variations ")

        variations = jnp.array(self.DB_CTL_VARIATION_LEN_PER_EQUATION)

        # hier ansetzen
        params_per_eq = jnp.array(self.METHOD_PARAM_LEN_CTLR)

        # loop calc now * prev
        products = variations * params_per_eq
        offsets = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(products)
        ])

        start_sum = offsets[eq_idx]
        print("start_sum", start_sum)

        # amount params per eq
        #amount_params_current_eq = self.METHOD_PARAM_LEN_CTLR[eq_idx]
        amount_params_current_eq = jnp.take(
            jnp.array(self.METHOD_PARAM_LEN_CTLR),
            eq_idx,
        )
        print("amount_params_current_eq", amount_params_current_eq)

        # get len of variations per equation
        amount_variations_current_eq = jnp.take(
            jnp.array(
                self.DB_CTL_VARIATION_LEN_PER_EQUATION),
            eq_idx
        )
        print("amount_variations_current_eq", amount_variations_current_eq)

        total_amount_params_current_eq = jnp.int64(amount_params_current_eq * amount_variations_current_eq)
        print("total_amount_params_current_eq", total_amount_params_current_eq)

        # todo must multiply each item in self.DB_CTL_VARIATION_LEN_PER_EQUATION[:eq_idx] * amount_params for specific equation
        slice = jax.lax.dynamic_slice_in_dim(
            jnp.array(self.DB_TO_METHOD_EDGES),
            start_sum,
            total_amount_params_current_eq,
            #axis=0,
        )
        #print("slice", slice)

        # Reshape in (num_chunks, n, len(inner_list))
        result = slice.reshape(
            total_amount_params_current_eq // amount_params_current_eq,
            amount_params_current_eq,
            4,
        )

        #print("result", result)
        return result, amount_params_current_eq



    def build_gnn_equation_nodes(self):
        # create equation_nodes
        print("convert to callable...")
        for eq_idx, eq in enumerate(self.METHODS):


            runnable = create_runnable(
                eq_code=eq
            )

            #debug_callable(runnable)

            node = Node()

            # replace mehod str with py class
            self.METHODS[eq_idx] = node
        print("convert to callable... done")

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




    def sort_features(self, eq_idx, all_features):
        len_eq_variations = self.ITERATORS["eq_variations"][eq_idx]
        start_idx = jnp.sum(self.ITERATORS["eq_variations"])[:eq_idx]

        indices = jnp.array(start_idx+i for i in range(len_eq_variations))
        # FEATURE -> MODEL
        self.model_skeleton.at[
            tuple(indices.T)
        ].add(all_features)


"""
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
    

# generate model tstep (shapes must match [*inputs, outputs] → 20 items)
        def _shape_tree(x):
            if isinstance(x, (list, tuple)):
                return [_shape_tree(e) for e in x]
            return getattr(x, "shape", x)
        out_shapes = _shape_tree(all_results)

"""