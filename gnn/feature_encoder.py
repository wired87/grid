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
        # Expose short alias used in docs: feature_encoder.ctlr
        self.ctlr = self.feature_controller
        # Tracks current simulation step (set from GNN.simulate via begin_step()).
        self._current_step = 0

    # --- CTLR helpers ---------------------------------------------------------
    def begin_step(self, step: int):
        """
        Mark start of a simulation step for controller tracking.
        GNN.simulate(step) should call this once per time step.
        """
        try:
            step = int(step)
        except Exception:
            step = int(self._current_step)
        self._current_step = step

        # ensure outer list has slot for this step
        while len(self.feature_controller) <= step:
            self.feature_controller.append({"step": len(self.feature_controller), "eq": {}})

    def _get_ctlr_entry(self, eq_idx: int) -> dict:
        """
        Get (and lazily create) controller dict for current step + equation.
        """
        step = int(self._current_step)
        # ensure step slot exists
        while len(self.feature_controller) <= step:
            self.feature_controller.append({"step": len(self.feature_controller), "eq": {}})

        step_entry = self.feature_controller[step]
        if "eq" not in step_entry or step_entry["eq"] is None:
            step_entry["eq"] = {}
        if eq_idx not in step_entry["eq"]:
            step_entry["eq"][eq_idx] = {}
        return step_entry["eq"][eq_idx]


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
                        #print("param, linear",param, linear)
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
        Generate out-features: one embedding per (segment, linear) pair.
        --- FIX: Accept list of segments (one per out_linear) and loop; vmap fails when
        axis 0 sizes differ (e.g. flatten_param 36 vs idx 37). ---
        """
        print("gen_feature_single_variation...")
        #print("param_grid", type(param_grid))
        #print("out_linears", type(out_linears))

        if not out_linears:
            pg = jnp.atleast_1d(jnp.asarray(param_grid))
            if isinstance(param_grid, list):
                n = len(param_grid)
            else:
                n = pg.shape[0]
            return jnp.zeros((n, self.d_model), dtype=jnp.float32)

        # --- FIX: param_grid is list of segments (one per linear); loop to avoid vmap size mismatch ---
        if isinstance(param_grid, (list, tuple)):
            out_list = []
            for i, (seg, linear) in enumerate(zip(param_grid, out_linears)):
                seg = jnp.asarray(seg).ravel()
                in_len = int(getattr(linear, "in_features", seg.size))
                if seg.size < in_len:
                    seg = jnp.concatenate([seg, jnp.zeros(in_len - seg.size, dtype=seg.dtype)])
                elif seg.size > in_len:
                    seg = seg[:in_len]
                out_list.append(jax.nn.gelu(linear(seg)))
            print("gen_feature_single_variation... done")
            return jnp.stack(out_list, axis=0)
        # Fallback: single array (legacy); if leading dim != len(out_linears), loop to avoid vmap mismatch
        try:
            param_grid = jnp.atleast_1d(jnp.asarray(param_grid))
            n_rows = param_grid.shape[0]
            if n_rows != len(out_linears):
                out_list = []
                for i in range(len(out_linears)):
                    in_len = int(jnp.asarray(getattr(out_linears[i], "in_features", 1)).ravel()[0])
                    seg = param_grid[i] if i < n_rows else jnp.zeros(in_len)
                    seg = jnp.asarray(seg).ravel()
                    if seg.size < in_len:
                        seg = jnp.concatenate([seg, jnp.zeros(in_len - seg.size, dtype=seg.dtype)])
                    else:
                        seg = seg[:in_len]
                    out_list.append(jax.nn.gelu(out_linears[i](seg)))
                return jnp.stack(out_list, axis=0)
            idx_map = jnp.arange(len(out_linears))
            def _work_param(flatten_param, idx: int):
                return jax.nn.gelu(out_linears[idx](flatten_param))
            features = vmap(_work_param, in_axes=(0, 0))(param_grid, idx_map)
        except Exception as e:
            print("Err gen_feature_single_variation", e)
            return jnp.zeros((len(out_linears), self.d_model), dtype=jnp.float32)
        print("gen_feature_single_variation... done")
        return features



    def _save_in_feature_method_grid(
            self,
            features_tree:jnp.array,
            eq_idx,
            item_idx
    ):
        # per-eq, per-param time-series store:
        # in_store[eq_idx][item_idx][t] -> feature embedding row
        print("_save_in_feature_method_grid...")
        try:
            # ensure nested structure exists (robust against partial init)
            if len(self.in_store) <= eq_idx:
                self.in_store.extend(
                    [[] for _ in range(eq_idx + 1 - len(self.in_store))]
                )
            if len(self.in_store[eq_idx]) <= item_idx:
                self.in_store[eq_idx].extend(
                    [[] for _ in range(item_idx + 1 - len(self.in_store[eq_idx]))]
                )

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
        try:
            # per-eq, per-param feature embeddings for the current time step
            results = []
            print("len comparishon", len(inputs), len(self.in_linears[eq_idx]), len(axis_def))

            # create embeddings for all scaled instances and immediately store them
            for item_idx, (single_input, linear_instance, ax) in enumerate(
                zip(inputs, self.in_linears[eq_idx], axis_def)
            ):
                f_in_results = self.gen_in_feature_single_variation(
                    single_input,
                    linear_instance,
                    ax,
                )
                results.append(f_in_results)

                # append to in_store[eq_idx][item_idx][t]
                self._save_in_feature_method_grid(
                    f_in_results,
                    eq_idx,
                    item_idx,
                )

            print("create_in_features results generated...")
            # --- CTLR: track per-step, per-eq in-feature meta for later iteration ---
            try:
                ctl = self._get_ctlr_entry(eq_idx)
                ctl["in"] = {
                    "axis_def": tuple(axis_def) if axis_def is not None else None,
                    "n_params": len(inputs) if inputs is not None else 0,
                    "in_shapes": [
                        list(getattr(r, "shape", ())) if hasattr(r, "shape") else None
                        for r in (results or [])
                    ],
                }
            except Exception as _e_ctlr_in:
                # keep training/debug robust; controller is best-effort only
                print("Err ctlr(in):", _e_ctlr_in)

            return results
        except Exception as e:
            print("Err create_in_features:", e)
        print("create_in_features... done")
        return None



    def blur_result_from_in_tree(
            self,
            eq_idx,
            high_score_elements,
            len_variations,
    ):
        """
        high_score_elements: list of 1D arrays (one per param), each shape (len_variations,).
        Returns a list of length len_variations: per-row blur value or None (must recompute).
        """
        if len_variations == 0 or not high_score_elements:
            print("blur_result_from_in_tree... done (no variations)")
            return jnp.full((max(1, len_variations), self.d_model), jnp.nan)

        # Normalize to (n_params, len_variations) so we can stack (params may have different lengths).
        def _to_len(s):
            a = jnp.asarray(s).ravel()
            n = a.shape[0]
            if n >= len_variations:
                return a[:len_variations]
            return jnp.concatenate([a, jnp.zeros(len_variations - n, dtype=a.dtype)])
        padded = [ _to_len(s) for s in high_score_elements ]
        stacked = jnp.stack(padded, axis=0)
        row_scores = jnp.sum(stacked, axis=0)

        # Sentinel for "must recompute": same shape (d_model,) so vmap gets uniform structure.
        recompute_sentinel = jnp.full((self.d_model,), jnp.nan)

        def _get_blur_val(row_idx):
            score = row_scores[row_idx]
            if score <= self.result_blur and len(self.out_store) > eq_idx and len(self.out_store[eq_idx]) > 0:
                last_out = self.out_store[eq_idx][-1]
                n_out = last_out.shape[0] if hasattr(last_out, "shape") else len(last_out)
                if row_idx < n_out:
                    return jnp.asarray(last_out[row_idx])
            return recompute_sentinel

        result = jnp.stack([_get_blur_val(r) for r in range(len_variations)], axis=0)
        print("blur_result_from_in_tree... done")

        # --- CTLR: track blur / reuse statistics for current step & equation ---
        try:
            ctl = self._get_ctlr_entry(eq_idx)
            reuse_mask = row_scores[:len_variations] <= self.result_blur
            ctl["blur"] = {
                "len_variations": int(len_variations),
                "row_scores": jnp.asarray(row_scores[:len_variations]),
                "reuse_mask": jnp.asarray(reuse_mask),
            }
        except Exception as _e_ctlr_blur:
            print("Err ctlr(blur):", _e_ctlr_blur)

        return result





    def create_out_features(
            self,
            output,
            eq_idx,
    ):
        # Track per-time-step out-features for each equation.
        print("FeatureEncoder.out_processor...")
        try:
            # --- FIX: Pass list of segments (one per out_linear) to avoid vmap axis size mismatch ---
            results = self.gen_out_feature_single_variation(
                output,
                out_linears=self.out_linears[eq_idx],
            )

            # lazily init store in case prep() did not run as expected
            if len(self.out_store) <= eq_idx:
                self.out_store.extend(
                    [[] for _ in range(eq_idx + 1 - len(self.out_store))]
                )

            # append current time-step embedding block
            out_block = jnp.array(results)
            self.out_store[eq_idx].append(out_block)

            # --- CTLR: track per-step, per-eq out-feature meta for later iteration ---
            try:
                ctl = self._get_ctlr_entry(eq_idx)
                ctl["out"] = {
                    "n_out": int(out_block.shape[0]) if hasattr(out_block, "shape") and out_block.ndim >= 1 else 0,
                    "d_model": int(out_block.shape[1]) if hasattr(out_block, "shape") and out_block.ndim >= 2 else self.d_model,
                }
            except Exception as _e_ctlr_out:
                print("Err ctlr(out):", _e_ctlr_out)
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
            return self.fill_blur_vals(
                param_embedding_item,
                pre_param_grid,
            )

        try:
            high_score_map = []
            # one entry per param / axis in this equation
            for param_idx, param_embedding_grid in enumerate(stacked_features_all_params):
                # if we have no history yet, fall back to zeros
                if param_idx >= len(self.in_store[eq_idx]) or len(self.in_store[eq_idx][param_idx]) == 0:
                    high_score_map.append(
                        jnp.zeros(param_embedding_grid.shape[0])
                    )
                    continue

                scores = vmap(
                    generate_high_score_single,
                    in_axes=(0, None)
                )(
                    param_embedding_grid,
                    self.in_store[eq_idx][param_idx],
                )
                high_score_map.append(scores)
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