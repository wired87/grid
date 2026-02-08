import pprint

import jax
from jax import jit, vmap, ops
# todo _BYTE binary_view = ' '.join(format(b, '08b') for b in text.encode('utf-8'))
from gnn.db_layer import DBLayer
from gnn.feature_encoder import FeatureEncoder
from gnn.injector import InjectorLayer
from jax_utils._segment import split_array
from jax_utils.conv_flat_to_shape import bring_flat_to_shape
from mod import Node
from utils import SHIFT_DIRS, create_runnable
from flax import nnx
"""
Build:
LINEARS, DB



"""
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
        self.len_params_per_methods = {}
        self.change_store = []

        self.gpu = gpu

        self.db_layer = DBLayer(
            amount_nodes,
            self.gpu,
            DIMS=DIMS,
            **cfg
        )

        # CHANGED: FeatureEncoder expects (AXIS, db_layer, amount_variations); was TypeError missing 2 args.
        _n_var = int(jnp.sum(self.db_layer.DB_CTL_VARIATION_LEN_PER_FIELD)) if hasattr(self.db_layer, "DB_CTL_VARIATION_LEN_PER_FIELD") else max(1, len(self.db_layer.METHOD_TO_DB))
        self.feature_encoder = FeatureEncoder(self.db_layer.AXIS, self.db_layer, amount_variations=_n_var)

        self.injector = InjectorLayer(
            db_layer=self.db_layer,
            amount_nodes=amount_nodes,
            DIMS=DIMS,
            **cfg
        )

        self.all_axs = []
        self.all_shapes = []

        print("Node initialized and build successfully")


        # CHANGED: LEN_FEATURES_PER_EQ is list of variable-length per-eq -> jnp.array(...) gives inhomogeneous ValueError. Use lengths then cumsum.
        _len_per_eq = jnp.array([len(x) for x in self.LEN_FEATURES_PER_EQ])
        self.LEN_FEATURES_PER_EQ_CUMSUM = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(_len_per_eq)
        ])
        self.LEN_FEATURES_PER_EQ_CUMSUM_UNPADDED = jnp.cumsum(_len_per_eq)


        self.FEATURES_CUMSUM = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(
                jnp.array([len(i)
                for i in self.LEN_FEATURES_PER_EQ])
            ),
        ])



    def main(self):
        self.prepare()
        self.simulate()

        serialized = self.serialize(
            self.feature_encoder.in_ts
        )
        print("serialized model_skeleton", serialized)

        jax.debug.print("process finished.")




    def prepare(self):
        # DB
        self.db_layer.build_db(
            self.amount_nodes,
        )

        # layer to define all ax et
        self.prep()




    def simulate(self):
        try:
            for step in range(self.time):
                jax.debug.print(
                    "Sim step {s}/{n}",
                    s=step,
                    n=self.time
                )

                # get params from DB layer and init calc
                all_results = self.calc_batch(
                    t=step
                )

                self.db_layer.save_t_step(
                    all_results,
                )

                # todo just save what has changed - not the entire array
        except Exception as e:
            print(f"Err simulate: {e}")


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
        print("short_transformed... ")
        new_t = []
        for ax, st in zip(all_ax, transformed):
            if ax == 0:
                new_t.append(st)  # Behält Shape (12, 4)
            else:
                new_t.append(st[0])  # Reduziert auf Shape (4,)
        print("short_transformed... done", new_t)
        return new_t


    def get_mod_idx(self, abs_eq_idx):
        # get mod_idx based on eq_idx
        # 1,2 -> 1,3
        cum = jnp.cumsum(jnp.array(self.MODULES))  # kumulative Summe
        # finde erstes i, bei dem cum >= abs_eq_idx
        idx = jnp.argmax(cum >= abs_eq_idx)
        return idx


    def prep(self):
        print("prep...")
        # direkt beim shape def i prep anand shape die in features bestimmen#
        # vorproduziert -< in runtime alles direk parat um ctlr zuverwenden
        for eq_idx, eq in enumerate(self.METHODS):
            #mod_idx = self.get_mod_idx(eq_idx)

            variations, amount_params = self.extract_eq_variations(
                eq_idx,
            )

            examle_variation = variations[0]
            axs, shapes = self.db_layer.get_axis_shape(
                examle_variation,
            )

            print("transfrom variations...")
            transformed = jnp.transpose(variations, (1, 0, 2))
            print("transfrom variations... done")

            # get all in shapes
            self.all_axs.append(axs)
            self.all_shapes.append(shapes)


            #
            transformed = self.short_transformed(
                self.all_axs[eq_idx],
                transformed,
            )

            # add empty dim for each eq (act as time feature store
            self.feature_encoder.in_store.append([])
            self.feature_encoder.out_store.append([])

            # LINEARS
            self.create_in_linears_process(
                eq_idx,
                transformed,
                axis_def=self.all_axs[eq_idx],
            )

            #
            self.create_out_linears_process(
                eq_idx,
            )

            node = self.create_node(eq, eq_idx)

            # replace mehod str with py class
            self.METHODS[eq_idx] = node
        print("prep... done")


    def create_out_linears_process(self, eq_idx):
        start = self.FEATURES_CUMSUM[eq_idx]

        mtdb_slice = jax.lax.dynamic_slice_in_dim(
            self.db_layer.METHOD_TO_DB,
            start,
            len(self.LEN_FEATURES_PER_EQ[eq_idx]),
        )

        rel_ids = self.batch_rel_idx(
            batch=mtdb_slice,
        )

        # MTDB & DB_CTL_VARIATION_LEN_PER_FIELD have same len
        unscaled_db_len = self.batch_len_unscaled(
            rel_ids, self.all_axs[eq_idx],
        )

        self.feature_encoder.create_out_linears(
            unscaled_db_len,
            feature_len_per_out=self.db_layer.DB_CTL_VARIATION_LEN_PER_FIELD
        )

    def create_in_linears_process(
            self,
            eq_idx,
            transformed,
            axis_def,
    ):
        print("create_in_linears_process... ")
        rel_ids:list[list] = self.get_rel_db_index_batch(
            eq_idx,
            transformed,
        )
        print("rel_ids", rel_ids)
        unscaled_db_len = jax.tree_util.tree_map(
            self.batch_len_unscaled,
            rel_ids,
            axis_def,
        )

        print("unscaled_db_len", unscaled_db_len)

        # create&save eq linears
        self.feature_encoder.build_linears(
            eq_idx,
            unscaled_db_len
        )


    def handle_in_features(self, grids):
        # gen in featrues

        def _handle_single_grid_features(grid, in_ax_def, t=0):
            in_features = self.feature_encoder.create_features(
                inputs=grid,
                axis_def=in_ax_def,
                time=t,
                param_idx=0
            )
            return in_features

        features = jax.tree_util.tree_map(
            _handle_single_grid_features,
            grids,
            self.all_axs
        )








    def create_node(self, eq, eq_idx) -> Node:
        runnable = create_runnable(
            eq_code=eq
        )

        node = Node(
            runnable,
            amount_variations=len(
                self.LEN_FEATURES_PER_EQ[eq_idx]
            ),
        )
        return node

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


    def calc_batch(self, t=0):
        jax.debug.print("calc_batch...")

        # calc all methods and apply result to new g
        all_results = []
        all_ins = []

        # START LOOP
        node: Node

        for eq_idx, node in enumerate(self.METHODS):
            axis_def = tuple(self.all_axs[eq_idx])

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
            print("transfrom variations...")
            transformed = jnp.transpose(variations, (1, 0, 2))
            print("transfrom variations... done")

            #
            transformed = self.short_transformed(
                self.all_axs[eq_idx],
                transformed,
            )

            #
            flatten_transformed = self.extract_flat_params(
                eq_idx,
                transformed,
            )

            # IN FEATURESS
            precomputed_results = self.feature_encoder.create_in_features(
                inputs=flatten_transformed, # flatten struct include just 1d entries -> order should be same as linears pre defined
                eq_idx=eq_idx,
                axis_def=axis_def,
            )


            # reshape flattened batch values
            inputs = self.shape_input(
                eq_idx,
                flatten_transformed,
            )
            print("flatten_transformed", flatten_transformed)


            # calc single equation
            results = node(
                unprocessed_in=inputs,
                precomputed_grid=precomputed_results,
                in_axes_def=axis_def,
            )

            # todo flatten out
            flatten_out = jnp.array([
                jnp.ravel(item)
                for item in results
            ])

            self.feature_encoder.create_out_features(
                output=flatten_out,
                eq_idx=eq_idx,
            )

            all_results.extend(results)
            all_ins.append(inputs)

        # todo sort values innto eq spec grids
        jax.debug.print("calc_batch... done")
        return all_ins, all_results


    def get_flatten_value(self, *inputs):
        # print("self.SCALED_PARAMS_CUMSUM", self.SCALED_PARAMS_CUMSUM)
        # get unscaled abs param idx
        def xtract_single(mod_idx, field_idx, rel_param_idx):
            abs_unscaled_param_idx = self.db_layer.get_rel_db_index(
                mod_idx,
                field_idx,
                rel_param_idx
            )
            return abs_unscaled_param_idx
        return vmap(xtract_single, in_axes=0)(*inputs)


    def get_scaled_idx(self, abs_unscaled_param_idx_batch):
        # get batch scaled idx from unscaled batch
        def xtract_single(abs_unscaled_param_idx):
            abs_unscaled_param_idx_and_len = self.db_layer.get_db_index(
                abs_unscaled_param_idx
            )
            return abs_unscaled_param_idx_and_len
        return vmap(xtract_single, in_axes=0)(abs_unscaled_param_idx_batch)


    def extract_flat_params(self, eq_idx, transformed):
        print("extract_flat_params...")
        flatten_transformed = []
        #flatten_sepparated = []

        for i, (param_grid, ax) in enumerate(
            zip(
                transformed,
                self.all_axs[eq_idx]
            )
        ):
            # problem: jax cant handle dynamic shapes...
            # param_grid
            single_param_grid = []

            # print("param_grid", param_grid)
            if ax == 0:
                for coords in param_grid:
                    # print("coords", coords)
                    coord_result = self.db_layer.extract_flattened_grid(
                        coords
                    )
                    # print(f"coord_result for {i} {coords}", len(coord_result))
                    single_param_grid.extend(
                        coord_result
                    )
                    """flatten_sepparated.append(
                        coord_result
                    )"""
            else:
                coord_result = self.db_layer.extract_flattened_grid(param_grid)
                # print(f"coord_result for {i} {param_grid}", len(coord_result))
                single_param_grid.extend(
                    coord_result
                )

            # print("single_param_grid", single_param_grid, len(single_param_grid))
            flatten_transformed.append(single_param_grid)
        print("flatten_transformed")
        pprint.pp(flatten_transformed)

        print("extract_flat_params... done")
        return flatten_transformed#, flatten_sepparated

    def batch_rel_idx(self, batch):
        def _wrapper(item):
            return self.db_layer.get_rel_db_index(
                *item
            )

        return vmap(
            _wrapper,
            in_axes=0
        )(
            # exclude time
            jnp.array(
                list(
                    i[1:]
                    for i in batch
                )
            )
        )



    def batch_len_unscaled(self, batch, axis):
        def _wrapper(i):
            return jnp.take(
                self.db_layer.DB_PARAM_CONTROLLER,
                i,
            )
        #def vmap_batch(batch):
        if axis == 0:
            return vmap(
                _wrapper,
                in_axes=0
            )(
                # exclude time
                batch
            )
        else:
            return jnp.take(
                self.db_layer.DB_PARAM_CONTROLLER,
                batch,
            )

    def get_rel_db_index_batch(self, eq_idx, transformed):
        print("get_rel_db_index_batch...")
        def _extract_coord_batch(ax, coord_batch):
            if ax == 0:
                result = self.batch_rel_idx(
                    coord_batch
                )
            else:
                item = coord_batch[1:]
                print(">item", item)
                result = self.db_layer.get_rel_db_index(
                    *item
                )
            return result

        rel_idx_map = jax.tree_util.tree_map(
            _extract_coord_batch,
            self.all_axs[eq_idx],
            transformed,
        )
        print("get_rel_db_index_batch... done")
        return rel_idx_map



    def shape_input(self, eq_idx, flatten_transformed):
        print("shape_input...")
        inputs = []

        for shape, variation_grids in zip(self.all_shapes[eq_idx], flatten_transformed):
            # print("variation_grids", variation_grids)
            result = bring_flat_to_shape(
                jnp.array(variation_grids),
                shape
            )
            result = jnp.array(result)
            inputs.append(jnp.array(result))
        print("shape_input... done")
        return inputs  # CHANGED: was 'input' (builtin) -> "Value after * must be an iterable, not builtin_function_or_method"

    def extract_eq_variations(self, eq_idx):
        """
        Extrahiert alle Variationen für ein spezifisches Gleichung.
        Navigiert durch DB und AXIS und skaliert Parameter bei axis == 0 auf amount_nodes.
        """
        jax.debug.print("extract_eq_variations ")

        variations = jnp.array(self.DB_CTL_VARIATION_LEN_PER_EQUATION)

        params_per_eq = jnp.array(self.METHOD_PARAM_LEN_CTLR)

        products = variations * params_per_eq

        offsets = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(products)
        ])

        start_sum = offsets[eq_idx]

        # amount params per eq
        #amount_params_current_eq = self.METHOD_PARAM_LEN_CTLR[eq_idx]
        amount_params_current_eq = jnp.take(
            jnp.array(self.METHOD_PARAM_LEN_CTLR),
            eq_idx,
        )
        print("amount_params_current_eq", amount_params_current_eq)

        # get len of variations per equation
        amount_variations_current_eq = jnp.take(
            jnp.array(self.DB_CTL_VARIATION_LEN_PER_EQUATION),
            eq_idx
        )
        print("amount_variations_current_eq", amount_variations_current_eq)

        total_amount_params_current_eq = jnp.int64(
            amount_params_current_eq * amount_variations_current_eq
        )
        print("total_amount_params_current_eq", total_amount_params_current_eq)

        # todo must multiply each item in self.DB_CTL_VARIATION_LEN_PER_EQUATION[:eq_idx] * amount_params for specific equation
        edges = jax.lax.dynamic_slice_in_dim(
            jnp.array(self.DB_TO_METHOD_EDGES),
            start_sum,
            total_amount_params_current_eq,
            #axis=0,
        )
        print("slice", slice)

        # Reshape in (num_chunks, n, len(inner_list))
        result = edges.reshape(
            total_amount_params_current_eq // amount_params_current_eq,
            amount_params_current_eq,
            4,
        )
        print("extract_eq_variations... done")
        return result, amount_params_current_eq




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

"""
Ziel: 
    # backgorund: db_layer.out_shape include shapes for each
    # out variation per field
    # get feature starting point
    feature_out_shapes_start = self.LEN_FEATURES_PER_EQ_CUMSUM[eq_idx]
    feature_out_shapes_len = self.LEN_FEATURES_PER_EQ[eq_idx]

    # pick slice
    _out_shapes_slice = jax.lax.dynamic_slice_in_dim(
        self.db_layer.OUT_SHAPES,
        feature_out_shapes_start,
        feature_out_shapes_len,
    )


    amount_feature_blocks_eq = len(self.LEN_FEATURES_PER_EQ[eq_idx])

    # group extracted shape slice into LEN_FEATURES_PER_EQ item
    group_ids = jnp.repeat(
        jnp.arange(amount_feature_blocks_eq),
        self.LEN_FEATURES_PER_EQ[eq_idx]
    )

    #
    grouped_shapes = ops.segment_sum(_out_shapes_slice, group_ids)

    ## out = grouped features based on len_f_per_eq / block
    out_linears = []


    #
    for shapes_struct in grouped_shapes:
        for single_shape in shapes_struct:
            param_len = self.get_unscaled_param_len

        out_linears.extend(
            [ # woher erhalten wir len of param ->
                # PARAM_CTLR ->
                # wie ehralten rel db id?
                nnx.Linear(
                    in_features=,
                    out_features=self.d_model,
                    rngs=self.rngs
                )
            ]
        )

"""