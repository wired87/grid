import jax
from jax import jit, vmap
# todo _BYTE binary_view = ' '.join(format(b, '08b') for b in text.encode('utf-8'))
from gnn.db_layer import DBLayer
from gnn.feature_encoder import FeatureEncoder
from gnn.injector import InjectorLayer
from jax_utils.conv_flat_to_shape import bring_flat_to_shape
from mod import Node
from utils import SHIFT_DIRS, create_runnable
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

        serialized_in = self.serialize(
            self.feature_encoder.in_store
        )
        serialized_out = self.serialize(
            self.feature_encoder.out_store
        )
        print("serialized serialized_in", serialized_in)
        print("serialized serialized_out", serialized_out)

        jax.debug.print("process finished.")
        return serialized_in, serialized_out, # todo ctlr

    def prepare(self):
        # DB
        self.db_layer.build_db(
            self.amount_nodes,
        )

        # layer to define all ax et
        self.prep()


    # ---------- Engine: simulation loop (Python loop; for JAX-traced version see engine_components.run_simulation_scan) ----------
    def simulate(self):
        try:
            for step in range(self.time):
                jax.debug.print(
                    "Sim step {s}/{n}",
                    s=step,
                    n=self.time
                )

                # --- CTLR: mark current step on feature encoder so it can track per-timestep controller data ---
                if hasattr(self, "feature_encoder"):
                    try:
                        self.feature_encoder.begin_step(step)
                    except Exception as _e_ctlr_step:
                        print("Err feature_encoder.begin_step:", _e_ctlr_step)

                # get params from DB layer and init calc
                all_results = self.calc_batch()

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
        print("short_transformed... done")
        return new_t


    def get_mod_idx(self, abs_eq_idx):
        # get mod_idx based on eq_idx
        cum = jnp.cumsum(jnp.array(self.MODULES))  # kumulative Summe
        # finde erstes i, bei dem cum >= abs_eq_idx
        idx = jnp.argmax(cum >= abs_eq_idx)
        return idx

    def reshape_variant_block(self, edges, total_amount_params_current_eq, amount_params_current_eq):
        result = edges.reshape(
            total_amount_params_current_eq // amount_params_current_eq,
            amount_params_current_eq,
            4,
        )
        return result

    def prep(self):
        print("prep...")
        # direkt beim shape def i prep anand shape die in features bestimmen#
        # vorproduziert -< in runtime alles direk parat um ctlr zuverwenden
        for eq_idx, eq in enumerate(self.METHODS):
            variations, amount_params_current_eq, total_amount_params_current_eq = self.extract_eq_variations(
                eq_idx,
            )

            variations = self.reshape_variant_block(
                variations,
                total_amount_params_current_eq,
                amount_params_current_eq,
            )


            examle_variation = variations[0]
            axs, shapes = self.db_layer.get_axis_shape(
                examle_variation,
            )

            print("transfrom variations...")
            transformed = jnp.transpose(variations, (1, 0, 2))
            print("transfrom variations... done")

            # get all in shapes (per-eq axis + shape definitions)
            self.all_axs.append(axs)
            self.all_shapes.append(shapes)

            # pre-shorten grids along axes
            transformed = self.short_transformed(
                self.all_axs[eq_idx],
                transformed,
            )

            # --- In-features: one Linear per (eq, param, variation) ---
            self.create_in_linears_process(
                eq_idx,
                transformed,
                axis_def=self.all_axs[eq_idx],
            )

            # init per-eq, per-param time stores AFTER linears exist
            # shape: in_store[eq_idx][param_idx][t] -> feature_row
            num_in_params = len(self.all_axs[eq_idx])
            self.feature_encoder.in_store.append(
                [[] for _ in range(num_in_params)]
            )

            # --- Out-features: one Linear per (eq, param_out, feature) ---
            # extract coords -> get data (!transformed) -> gen features ->
            self.create_out_linears_process(
                eq_idx,
            )

            # shape: out_store[eq_idx][t] -> stacked out-features for this eq
            self.feature_encoder.out_store.append([])

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
        unscaled_db_len = self.db_layer.batch_len_scaled(
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
        #print("rel_ids", rel_ids)

        scaled_db_len = jax.tree_util.tree_map(
            self.db_layer.batch_len_scaled,
            rel_ids,
            axis_def,
        )

        # create&save eq linears
        self.feature_encoder.build_linears(
            eq_idx,
            scaled_db_len
        )
        print("create_in_linears_process... done")





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


    def calc_batch(self):
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
            variations, amount_params_current_eq, total_amount_params_current_eq = self.extract_eq_variations(
                eq_idx,
            )

            #
            variations = self.reshape_variant_block(
                variations,
                total_amount_params_current_eq,
                amount_params_current_eq,
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

            ### IN FEATURES ###
            features_tree = self.feature_encoder.create_in_features(
                inputs=flatten_transformed, # flatten struct include just 1d entries -> order should be same as linears pre defined
                eq_idx=eq_idx,
                axis_def=axis_def,
            )

            high_score_elements = self.feature_encoder.get_precomputed_results(
                features_tree,
                axis_def,
                eq_idx,
            )

            # reshape flattened batch values (needed to get node batch size)
            inputs = self.shape_input(
                eq_idx,
                flatten_transformed,
            )
            # Node vmap batch size = size of mapped axis; use max over batched (axis-0) inputs.
            batch_size = 0
            if inputs and axis_def is not None:
                for i, inp in enumerate(inputs):
                    if hasattr(inp, "shape") and inp.shape and (i >= len(axis_def) or axis_def[i] == 0):
                        batch_size = max(batch_size, int(inp.shape[0]))
            if batch_size == 0 and inputs:
                batch_size = int(inputs[0].shape[0]) if hasattr(inputs[0], "shape") else 0

            # One blur result per variation row; len_variations = batch_size so vmap sizes match.
            if not high_score_elements or batch_size == 0:
                _blur = jnp.full((max(1, batch_size), self.feature_encoder.d_model), jnp.nan)
            else:
                _blur = self.feature_encoder.blur_result_from_in_tree(
                    eq_idx,
                    high_score_elements,
                    batch_size,
                )
            ### ###

            # calc single equation
            results = node(
                unprocessed_in=inputs,
                precomputed_grid=_blur,
                in_axes_def=axis_def,
            )

            # --- FIX: Out-features need one segment per out_linear; vmap fails if flatten_out length != len(out_linears) ---
            # Build one raveled segment per result item; pad/trim to match out_linears count.
            out_linears_eq = self.feature_encoder.out_linears[eq_idx]
            segments = [jnp.ravel(item) for item in results]
            n_linear = len(out_linears_eq)
            if len(segments) > n_linear:
                segments = segments[:n_linear]
            while len(segments) < n_linear:
                # Pad with zeros; length must match that linear's in_features (use 1 to avoid shape error).
                lin = out_linears_eq[len(segments)]
                in_len = int(getattr(lin, "in_features", 1))
                segments.append(jnp.zeros(in_len, dtype=jnp.float32))
            self.feature_encoder.create_out_features(
                output=segments,
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
                    single_param_grid.append(
                        coord_result
                    )
            else:
                coord_result = self.db_layer.extract_flattened_grid(param_grid)
                # print(f"coord_result for {i} {param_grid}", len(coord_result))
                single_param_grid.append(
                    coord_result
                )

            # print("single_param_grid", single_param_grid, len(single_param_grid))
            flatten_transformed.append(single_param_grid)

        print("extract_flat_params... done")
        return flatten_transformed

    def batch_rel_idx(self, batch):
        batch = jnp.reshape(jnp.ravel(batch), (-1, 4))
        def _wrapper(item):
            return self.db_layer.get_rel_db_index(
                *item
            )

        return vmap(
            _wrapper,
            in_axes=0
        )(
            batch[:, 1:]
        )





    def get_rel_db_index_batch(self, eq_idx, transformed):
        print("get_rel_db_index_batch...")

        def _extract_coord_batch(ax, coord_batch):
            if ax == 0:
                result = self.batch_rel_idx(
                    coord_batch
                )
            else:
                flat = jnp.ravel(jnp.asarray(coord_batch))
                result = self.db_layer.get_rel_db_index(
                    *flat[-3:]
                )
            #print(">result", result)
            return result

        rel_idx_map = []
        for item, ax in zip(transformed, self.all_axs[eq_idx]):
            rel_idx_map.append(
                _extract_coord_batch(
                    ax,
                    item,
                )
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
        Extract variations
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
        print("extract_eq_variations... done")
        return edges, amount_params_current_eq, total_amount_params_current_eq




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


    # WASTELANDS

    def create_feature_rows(self, eq_idx, variations):
        #

        def _process(variation, row_idx):
            flatten_params = jax.tree_util.tree_map(
                lambda x: self.db_layer.extract_flattened_grid(x),
                variation
            )

            linears = self.feature_encoder.get_linear_row(
                eq_idx,
                row_idx,
            )

            feature_rows = jax.tree_util.tree_map(
                lambda p, l: self.feature_encoder.gen_feature(p, l),
                flatten_params,
                linears,
            )

            return feature_rows

        try:
            kernel = jax.vmap(
                _process,
                in_axes=(0, 0),
            )

            idx_map = jnp.arange(len(variations))

            feature_rows = kernel(
                variations,
                idx_map
            )

            # save features
            self.feature_encoder._save_in_feature_method_grid(
                feature_rows,
                eq_idx
            )

            print("create_feature_rows... done")
            return feature_rows
        except Exception as e:
            print("Err create_feature_rows", e)


"""

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
"""

"""#
feature_rows = self.create_feature_rows(
    eq_idx,
    variations
)

# 
_blur = self.feature_encoder.get_precomputed_results(
    feature_rows,
    axis_def,
    eq_idx,
    
    

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


)"""