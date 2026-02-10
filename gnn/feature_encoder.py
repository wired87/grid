import jax
from jax import vmap
from flax import nnx
import jax.numpy as jnp


class FeatureEncoder(nnx.Module):
    # represents singe eq

    # todo do not get linears from db because difeent variatons (e.g. ff interaction) need diferent linears
    # tod build controller with each timestep ( track variations per eq dynamically )

    def __init__(
            self,
            AXIS,
            db_layer,
            amount_variations:int,
            d_model: int = 64,
    ):
        # Wir speichern für jede Eingabe eine Projektion auf 64 Dimensionen
        self.rngs = nnx.Rngs(42)
        self.d_model = d_model
        self.amount_variations = amount_variations
        self.db_layer = db_layer

        self.AXIS=AXIS

        # keep list and not flatten since
        self.in_store = []
        self.out_store = []
        self.out_skeleton = []
        self.in_skeleton = []
        self.in_ts = []  # CHANGED: gnn.py serialize() expects feature_encoder.in_ts

        self.in_linears = []
        self.out_linears:list[nnx.Linear] = []

        self.result_blur = .9
        self.feature_controller = []


    def get_linear_row(self, eq_idx, row_idx):
        print("get_linear_row...")
        return jax.tree_util.tree_map(
            lambda x: jnp.take(x, row_idx, axis=0),
            jnp.take(jnp.array(self.in_linears), eq_idx),
        )


    def gen_feature(
            self,
            param,
            linear,
    ):
        # embed linear:nnx.Linear
        print("gen_feature...")
        try:
            embedding = jax.nn.gelu(linear(param))
            #print("embedding", embedding)
            return embedding
        except Exception as e:
            print("Err gen_feature", e)

    def gen_in_feature_single_variation(
        self,
        param_grid,
        linear_batch,
        ax
    ):
        """
        GENERATE FEATURES WITH GIVEN LINEARS
        # todo use segment for pre segementation of
        # values instead of flatten them again
        """
        print("gen_feature_single_variation...")
        print("param_grid linear_batch,ax", type(param_grid), type(linear_batch), type(ax))


        feature_block_single_param_grid = []
        print("param_grid", param_grid)
        try:
            for i, (param, linear) in enumerate(
                    zip(param_grid, linear_batch)
            ):
                if linear:
                    try:
                        print("param, linear",param, linear)
                        embedding = jax.nn.gelu(linear(jnp.array(param)))
                        feature_block_single_param_grid.append(embedding)

                    except Exception as e:
                        print(f"Err _work_param at index {i}: {e}")

            # Am Ende alles zu einem JAX-Array zusammenfügen
            features = jnp.array(feature_block_single_param_grid)

        except Exception as e:
            print("Err gen_in_feature_single_variation", e)
            return jnp.array([])

        print("gen_feature_single_variation... done")
        return features


    def gen_out_feature_single_variation(
        self,
        param_grid,
        out_linears
    ):
        """
        GENERATE FEATURES WITH GIVEN LINEARS
        # todo use segment for pre segementation of
        # values instead of flatten them again
        """
        print("gen_feature_single_variation...")
        print("param_grid", type(param_grid))
        print("out_linears", type(out_linears))

        idx_map = jnp.arange(len(out_linears))

        def _work_param(
                flatten_param,
                idx:int,
        ):
            linear: nnx.Linear = out_linears[idx]
            # embed
            return jax.nn.gelu(
                linear(flatten_param)
            )

        try:
            param_grid = jnp.atleast_1d(jnp.asarray(param_grid))
            kernel = vmap(
                _work_param,
                in_axes=(0, 0)
            )

            feature_block_single_param_grid = kernel(
                param_grid,
                idx_map,
            )

            # working single parameter
            features = jnp.array(
                feature_block_single_param_grid
            )
        except Exception as e:
            print("Err gen_feature_single_variation", e)
            return jnp.array([])  # CHANGED: always return array so tree_map doesn't get None -> "Expected list, got (...)"
        print("gen_feature_single_variation... done")
        return features



    def _save_in_feature_method_grid(
            self,
            features_tree:jnp.array,
            eq_idx,
            item_idx
    ):
        # todo where create skeletons?
        print("_save_in_feature_method_grid...")
        try:
            # bring to 1d shspae
            self.in_store[eq_idx][item_idx].append(features_tree)
        except Exception as e:
            print("Err _save", e)
        print("_save_in_feature_method_grid... done")


    def build_linears(self, eq_idx, unscaled_db_len):
        print(f"build_linears for {eq_idx}...")

        def build_single(ilen):
            linear = nnx.Linear(
                in_features=ilen,
                out_features=self.d_model,
                rngs=self.rngs
            )
            return linear

        linears = []
        for item in unscaled_db_len:
            #print("item", item)
            var_linears = []
            item = jnp.atleast_1d(jnp.asarray(item))
            for variation in item:
                #print("item", item)
                if variation > 0:
                    var_linears.append(
                        build_single(
                            ilen=variation
                        )
                    )
                else:
                    # fallback
                    var_linears.append(None)

            linears.append(var_linears)
        self.in_linears.append(linears)
        print(f"build_linears... done")


    def create_in_features(
            self,
            inputs,
            axis_def,
            eq_idx=0,
    ):
        print("create_in_features...")
        precomputed_results = None
        _arange = jnp.arange(len(inputs))

        def _save(
                results,
                _arange
        ):
            self._save_in_feature_method_grid(
                results,
                eq_idx,
                _arange,
            )

        try:
            # in feature store ts
            results = []
            print("len comparishon", len(inputs), len(self.in_linears[eq_idx]), len(axis_def))
            # create emebddings for all scaled instances
            for input, linear_insance, ax in zip(inputs, self.in_linears[eq_idx], axis_def):
                f_in_results = self.gen_in_feature_single_variation(
                    input,
                    linear_insance,
                    ax,
                )
                results.append(f_in_results)

            print("create_in_features results generated...")

            jax.tree_util.tree_map(
                lambda leaf: _save(leaf, _arange),
                results
            )
            return results
        except Exception as e:
            print("Err create_in_features:", e)
        print("create_in_features... done")
        return precomputed_results



    def blur_result_from_in_tree(
            self,
            eq_idx,
            high_score_tree,
            len_variations,
    ):
        arange = jnp.arange(len_variations)

        def _get_blur_val(row_idx):
            score = jnp.sum(
                jax.tree_util.tree_map(
                    lambda x: jnp.take(x, row_idx, axis=0),
                    jnp.take(jnp.array(high_score_tree), eq_idx),
                )
            )

            if score <= self.result_blur:
                return jnp.take( # todo may pick from history db
                    self.out_store[eq_idx],
                    row_idx,
                )
            else:
                # -> MUST BE CALCED
                return None

        result = vmap(
            _get_blur_val,
            in_axes=(0, 0)
        )(
            arange
        )
        # todo might take tree_map
        print("blur_result_from_in_tree... done")
        return result





    def create_out_features(
            self,
            output,
            eq_idx,
    ):
        # todo include check fr prev time vals to
        print("FeatureEncoder.out_processor...")
        try:
            # calc features
            results = self.gen_out_feature_single_variation(
                output,
                out_linears=self.out_linears[eq_idx]
            )

            # save model tstep
            jnp.stack([
                self.out_store[eq_idx],
                jnp.array(results)
            ])
        except Exception as e:
            print("Err FeatureEncoder.out_processor:", e)
        print("FeatureEncoder.out_processor... done")


    def fill_blur_vals(self, feature_row, prev_params):
        # similarity search (ss) nearest neighbor filter blur
        try:
            print("fill_blur_vals...")
            embeddings = jnp.stack(prev_params, axis=0)      # (N, d_model)
            if embeddings.ndim == 1:
                embeddings = jnp.reshape(embeddings, (1, -1))

            ec = feature_row
            if jnp.ndim(ec) == 1:
                ec = jnp.reshape(ec, (1, -1))

            # L2-Distanz zu allen (axis=-1 works for 1D and 2D; axis=1 fails for 1D)
            diff = embeddings - ec

            losses = jnp.linalg.norm(diff, axis=-1)

            # min entry returns idx (everythign is order based (fixed f-len)
            idx = jnp.argmin(losses)
            min_loss = losses[idx]
            return min_loss
            # -> PRE CALCULATED RESULT BASED ON BLUR


        except Exception as e:
            print("Err fill_blur_vals", e)


    def get_precomputed_results(
            self,
            stacked_features_all_params,
            axis_def,
            eq_idx,
    ):
        print("get_precomputed_results...")

        def generate_high_score_single(param_embedding_item, pre_param_grid):
            self.fill_blur_vals(
                param_embedding_item,
                pre_param_grid,
            )

        def _create_high_score_batch(
                param_embedding_grid,
                param_idx,
        ):

            return vmap(
                generate_high_score_single,
                in_axes=(0, None)
            )(
                param_embedding_grid,
                self.in_store[eq_idx][param_idx],
            )

        try:
            # todo check from 2; avoid indexing in_store when empty (index out of bounds for axis 0 with size 0)
            # past in features
            _arange = jnp.arange(
                len(self.in_store[eq_idx])
            )

            high_score_map = jax.tree_util.tree_map(
                _create_high_score_batch,
                stacked_features_all_params,
                _arange,
            )
        except Exception as e:
            print("Err get_precomputed_results", e)
            high_score_map = []
        print("check... done")
        return high_score_map


    def convert_feature_to_rows(
            self,
            axis_def,
            in_features,
    ):
        # convert past feature steps o rows
        print("convert_feature_to_rows...")
        print("convert_feature_to_rows in_features", [len(i) for i in in_features], axis_def)

        def batch_padding():
            for i, item in enumerate(in_features):
                if len(item) == 0:
                    padding = jnp.array([
                        jnp.zeros(self.d_model)
                        for _ in range(len(in_features[0]))
                    ])
                    in_features[i] = padding
            return in_features

        def _process(*item):
            print("_process...")
            _arrays = jax.tree_util.tree_map(jnp.array, item)
            _arrays = jax.tree_util.tree_map(jnp.ravel, _arrays)

            return jnp.concatenate(
                arrays=jnp.array(_arrays),
                axis=0
            )

        try:
            in_features = batch_padding()

            kernel = jax.vmap(
                fun=_process,
                in_axes=axis_def,
            )

            feature_rows = kernel(
                *in_features
            )
            print("convert_feature_to_rows... done")
            return feature_rows
        except Exception as e:
            print("Err convert_feature_to_rows", e)


    def create_out_linears(
            self,
            unscaled_db_len,
            feature_len_per_out,
    ):
        print("create_out_linears...")

        linears= []
        for in_dim, amount_features in zip(
            unscaled_db_len,
            feature_len_per_out
        ):
            linears.extend([
                nnx.Linear(
                    in_features=in_dim,
                    out_features=self.d_model,
                    rngs=self.rngs
                )
                for _ in range(amount_features)
            ])

        # transform 1d
        self.out_linears.append(linears)
        print("create_out_linears... done")


"""
# check build out skeleton
jax.lax.cond(
    len(self.out_skeleton) == 0,
    lambda: self.build_out_skeleton(
        item=jax.tree_util.tree_map(
            lambda x: x[0], output
        )
    ),
    lambda: print("pass"),
)


        def _create_single_linear(
                item_len,
                feature_len_item
        ):
            linears = [
                nnx.Linear(
                    in_features=item_len,
                    out_features=self.d_model,
                    rngs=self.rngs
                )
                for _ in range(feature_len_item)
            ]
            return jnp.array(linears)
            
# CHANGED: was module-level; GNN calls self.feature_encoder.create_out_linears() -> add as method.
def create_out_linears(self):
    def _create_single_linear(item_len):
        return nnx.Linear(
            in_features=item_len,
            out_features=self.d_model,
            rngs=self.rngs
        )
    # CHANGED: METHOD_TO_DB, DB_CTL_VARIATION_LEN_PER_FIELD, DB_PARAM_CONTROLLER live on db_layer.
    out_rel_db_idx = self.db_layer.get_abs_unscaled_db_idx(
        self.db_layer.METHOD_TO_DB
    )
    # CHANGED: avoid iterating over JAX arrays (0-d elements raise); use range + index.
    #var_lens = jnp.atleast_1d(self.db_layer.DB_CTL_VARIATION_LEN_PER_FIELD)
    linear_batch = []
    for i in range(len(out_rel_db_idx)):
        db_idx = int(out_rel_db_idx[i]) if hasattr(out_rel_db_idx[i], "__int__") else out_rel_db_idx[i]
        len_unscaled_param = jnp.int64(
            self.db_layer.DB_PARAM_CONTROLLER[db_idx]
        )
        # CHANGED: len_unscaled_param is scalar per i; vmap needs at least one batched axis -> create one linear per iteration.
        linear_batch.append(_create_single_linear(int(len_unscaled_param)))
    print("classify_feature...")
    # CHANGED: linear_batch is list of nnx.Linear modules, not numeric -> return list, not jnp.array.
    return linear_batch
        try:
            feature_block_single_param_grid = []
            for item in _arange:
                res = _work_param(
                    item
                )
                feature_block_single_param_grid.append(res)

            # working single parameter
            features = jnp.array(feature_block_single_param_grid)
        except Exception as e:
            print("Err gen_in_feature_single_variation", e)
            return jnp.array([])
"""