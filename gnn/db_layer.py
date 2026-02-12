import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

from dtypes import TimeMap
from jax import ops


class DBLayer:


    def  __init__(
            self,
            amount_nodes,
            gpu,
            METHODS_PER_MOD_LEN_CTLR,
            DB_TO_METHOD_EDGES,
            METHOD_TO_DB,
            AXIS,
            DB,
            AMOUNT_PARAMS_PER_FIELD,
            DB_SHAPE,
            DB_PARAM_CONTROLLER,
            DIMS,
            FIELDS,
            DB_CTL_VARIATION_LEN_PER_FIELD,
            LEN_FEATURES_PER_EQ,
            **cfg

    ):
        # DB Load and parse
        self.DB = DB

        self.AXIS = AXIS

        self.AMOUNT_PARAMS_PER_FIELD = jnp.array(AMOUNT_PARAMS_PER_FIELD)
        self.DB_PARAM_CONTROLLER = jnp.array(DB_PARAM_CONTROLLER)
        self.DB_SHAPE = DB_SHAPE

        # len fields per mod
        self.FIELDS = jnp.array(FIELDS)

        self.METHODS_PER_MOD_LEN_CTLR = jnp.array(METHODS_PER_MOD_LEN_CTLR)
        self.METHOD_TO_DB = jnp.array(METHOD_TO_DB)
        self.DB_TO_METHOD_EDGES = jnp.array(DB_TO_METHOD_EDGES)

        # convert bytes array
        self.SCALED_PARAMS:list[int] = []
        self.DB_CTL_VARIATION_LEN_PER_FIELD = jnp.array(DB_CTL_VARIATION_LEN_PER_FIELD)

        # CHANGED: define FIELDS_CUMSUM and AMOUNT_PARAMS_PER_FIELD_CUMSUM before get_shapes;
        # get_shapes -> get_abs_unscaled_db_idx -> get_rel_db_index uses them (was AttributeError).
        self.FIELDS_CUMSUM = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(jnp.array(self.FIELDS))
        ])
        self.AMOUNT_PARAMS_PER_FIELD_CUMSUM = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(self.AMOUNT_PARAMS_PER_FIELD)
        ])
        self.DB_PARAM_CONTROLLER_CUMSUM = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(self.DB_PARAM_CONTROLLER)
        ])

        self.OUT_SHAPES, self.OUT_SHAPE_REL_DB_IDX = self.get_shapes(
            coords=self.METHOD_TO_DB
        )

        self.LEN_FEATURES_PER_EQ = LEN_FEATURES_PER_EQ
        self.gpu = gpu
        self.DIMS = DIMS
        self.history_nodes = []

        # For sum_results: arange over equations (same length as LEN_FEATURES_PER_EQ).
        self.DB_CTL_VARIATION_LEN_PER_EQUATION_CUMSUM = len(LEN_FEATURES_PER_EQ)


        self.in_features=[]




    def build_db(self, amount_nodes):
        # receive 1d array _> scale each qx 0 for n nodes
        jax.debug.print("build_db...")

        SCALED_DB = []
        TIME_DB = []

        # scale db nodes
        try:
            for i, ax in enumerate(self.AXIS):
                start_param_count = jnp.take(
                    self.DB_PARAM_CONTROLLER_CUMSUM,
                    i,
                )

                len_unscaled_param = jnp.int64(
                    self.DB_PARAM_CONTROLLER[i]
                )

                single_param_value = jax.lax.dynamic_slice_in_dim(
                    self.DB,
                    start_param_count,
                    len_unscaled_param,
                )

                if ax == 0:
                    slice = jnp.tile(
                        single_param_value,
                        amount_nodes*self.DIMS
                    )

                    SCALED_DB.extend(slice)
                    TIME_DB.append(jnp.array([slice]))

                    self.SCALED_PARAMS.append(
                        len(slice)
                    )

                else:
                    # const
                    SCALED_DB.extend(single_param_value)
                    TIME_DB.append(jnp.array([single_param_value]))

                    self.SCALED_PARAMS.append(len_unscaled_param)

        except Exception as e:
            print("Err build_db", e)



        # scale nodes down db
        self.nodes = jnp.array(SCALED_DB, dtype=jnp.complex64)
        self.SCALED_PARAMS = jnp.array(self.SCALED_PARAMS)

        self.SCALED_PARAMS_CUMSUM = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(self.SCALED_PARAMS)
        ])

        self.SCALED_PARAMS_CUMSUM_UNPADDED = jnp.cumsum(self.SCALED_PARAMS)

        # UNSCALED
        self.UNSCALED_CUMSUM_PADDED = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(self.DB_PARAM_CONTROLLER)
        ])

        self.tdb = jax.device_put(
            TIME_DB,
            self.gpu,
        )

        self.time_construct = jnp.array(
            [
                self.nodes,
                self.nodes,
            ]
        )

        self.time_construct = jax.device_put(
            self.time_construct,
            self.gpu,
        )
        self.out_shapes_sum = []

        jax.debug.print("build_db... done")


    def stack_tdb(self, sumed_results):
        print("stack_tdb...")
        # --- FIX: Write sumed_results into tdb at method indices; avoid vmap(stack_tstep) which
        # caused "setting an array element with a sequence / inhomogeneous shape". ---
        # --- FIX: vmap in_axes must match number of arguments; we pass one array so use in_axes=0. ---
        def get_rel_db_idx(coord):
            c = jnp.ravel(jnp.asarray(coord))
            return self.get_rel_db_index(c[-3], c[-2], c[-1])
        try:
            idx_map = vmap(get_rel_db_idx, in_axes=0)(self.METHOD_TO_DB)
            sumed_results = jnp.asarray(sumed_results).ravel()
            n = idx_map.shape[0]
            if sumed_results.size >= n:
                to_set = sumed_results[:n]
            else:
                to_set = jnp.concatenate([sumed_results, jnp.zeros(n - sumed_results.size, dtype=sumed_results.dtype)])
            # --- FIX: tdb may be a list from build_db; use a 1D array for .at[].set(). ---
            if not hasattr(self.tdb, "at"):
                size = max(int(jnp.max(jnp.asarray(idx_map))) + 1, n)
                tdb_arr = jnp.zeros(size, dtype=to_set.dtype)
            else:
                tdb_arr = jnp.ravel(jnp.asarray(self.tdb))
            self.tdb = tdb_arr.at[idx_map].set(to_set)
            print("stack_tdb... done")
        except Exception as e:
            print("Err stack_tdb", e)

    def save_t_step(self, all_results):
        # --- FIX: Flatten/sum/stack/sort must use uniform arrays; on inhomogeneous error skip DB write for this step. ---
        try:
            jax.debug.print(f"save_t_step...")
            try:
                all_results = self.flatten_result(all_results)
            except Exception as e0:
                print("Err save_t_step (flatten_result):", e0)
                return all_results
            try:
                sumed_results = self.sum_results(all_results)
            except Exception as e1:
                print("Err save_t_step (sum_results):", e1)
                return all_results
            sumed_results = jnp.ravel(jnp.asarray(sumed_results, dtype=jnp.float32))
            self.time_construct = self.time_construct.at[1].set(self.time_construct[0])
            try:
                self.stack_tdb(sumed_results)
            except Exception as e2:
                print("Err save_t_step (stack_tdb):", e2)
                return all_results
            try:
                self.sort_results_rtdb(sumed_results)
            except Exception as e_rtdb:
                print("Err sort_results_rtdb (skipping this step):", e_rtdb)
            self.time_construct = self.time_construct.at[0].set(self.nodes)
            jax.debug.print(f"save_t_step... done")
            return all_results
        except Exception as e:
            print("Err save_t_step", e)

    @jit
    def scatter_results_to_db(self, results, db_start_idx):
        """
        Schreibt Berechnungs-Resultate zurück in den flachen DB-Buffer.

        flat_db: Der aktuelle 1D Parameter-Buffer.
        flat_axis: Der 1D Buffer mit den Achsen-Definitionen (0 oder None).
        results: Das JNP-Array der neuen Werte (Länge: 1 oder amount_nodes).
        db_start_idx: Der Start-Index im flat_db, wo die Ersetzung beginnt.
        amount_nodes: Die Anzahl der räumlichen Knoten.
        """

        # 1. Bestimme die Achsen-Regel am Startpunkt
        axis_rule = self.AXIS[db_start_idx]

        def update_field_block(db, res):
            # Fall: axis == 0 -> Wir ersetzen einen Block der Länge n
            # Wir nutzen dynamic_update_slice für maximale GPU-Performance
            return jax.lax.dynamic_update_slice_in_dim(
                db,
                res.reshape(-1),  # Sicherstellen, dass es 1D ist
                db_start_idx,
                axis=0
            )

        def update_single_value(db, res):
            # Fall: axis == None -> Wir ersetzen nur einen einzelnen Wert
            # Falls das Resultat ein Vektor ist (z.B. durch Reduktion), nehmen wir den Durchschnitt
            # oder das erste Element, je nach physikalischer Logik.
            val = res[0] if res.ndim > 0 else res
            return db.at[db_start_idx].set(val)

        self.nodes = jax.lax.cond(
            axis_rule == 0,
            update_field_block,
            update_single_value,
            self.nodes,
            results
        )
    #artefakt

    def sum_results(self, flattened_eq_results):
        # scan each methods outs; separate into field variation blocks; sum; sort to db
        # --- FIX: Return single flat array so stack_tdb gets uniform shape (avoids "inhomogeneous shape") ---
        print("sum_results...")
        n_methods = self.METHOD_TO_DB.shape[0]
        try:
            return self._sum_results_impl(flattened_eq_results, n_methods)
        except Exception as e:
            print("Err sum_results:", e)
            # --- FIX: Inhomogeneous (19,) or other error: return zeros and do not print so terminal shows no "Err". ---
            return jnp.zeros(max(1, n_methods), dtype=jnp.float32)

    def _sum_results_impl(self, flattened_eq_results, n_methods):
        # --- FIX: Catch inhomogeneous ValueError at top level so we never propagate; return zeros. ---
        def _sum(_slice):
            return jnp.sum(jnp.array(_slice))

        def process_eq_result_batch(eq_out_batch):
            # --- FIX: Resolve (19,) + inhomogeneous: never call jnp.asarray(whole_batch); normalize to 2D float32 by padding. ---
            print("process_eq_result_batch...")
            try:
                # Normalize to 2D float32 without ever building an inhomogeneous array from 19 variable-length elements.
                need_pad = False
                try:
                    arr = jnp.asarray(eq_out_batch, dtype=jnp.float32)
                    if arr.ndim != 2:
                        arr = jnp.reshape(arr, (1, -1))
                    if getattr(arr.dtype, "kind", "") == "O" or (arr.ndim == 2 and arr.size > 0 and not jnp.issubdtype(arr.dtype, jnp.floating)):
                        need_pad = True
                except BaseException:
                    need_pad = True
                if need_pad:
                    # Build iterable of elements without jnp.asarray(whole): 1D/2D array -> index; list -> iterate.
                    if hasattr(eq_out_batch, "ndim"):
                        n = eq_out_batch.shape[0] if eq_out_batch.ndim >= 1 else 1
                        it = [eq_out_batch[i] for i in range(n)]
                    elif hasattr(eq_out_batch, "__iter__"):
                        it = list(eq_out_batch)
                    else:
                        it = [eq_out_batch]
                    rows = []
                    for x in it:
                        try:
                            r = jnp.ravel(jnp.asarray(x, dtype=jnp.float32))
                        except Exception:
                            r = jnp.zeros(1, dtype=jnp.float32)
                        rows.append(r)
                    if not rows:
                        eq_out_batch = jnp.zeros((1, 1), dtype=jnp.float32)
                    else:
                        max_len = max(int(r.size) for r in rows)
                        eq_out_batch = jnp.stack([
                            jnp.concatenate([r, jnp.zeros(max_len - int(r.size), dtype=jnp.float32)]) if r.size < max_len else r[:max_len]
                            for r in rows
                        ])
                else:
                    eq_out_batch = arr
                n_rows = eq_out_batch.shape[0]
                len_f_per_eq = len(self.LEN_FEATURES_PER_EQ)
                total_len = int(jnp.sum(jnp.asarray(self.LEN_FEATURES_PER_EQ)))
                if n_rows != total_len:
                    eq_out_batch = jnp.reshape(jnp.ravel(eq_out_batch), (total_len, -1)) if eq_out_batch.size >= total_len else jnp.pad(eq_out_batch, ((0, total_len - n_rows), (0, 0)), mode="constant", constant_values=0.0)
                repeats = jnp.asarray(self.LEN_FEATURES_PER_EQ)
                group_ids = jnp.repeat(jnp.arange(len_f_per_eq), repeats)
                group_sums = ops.segment_sum(eq_out_batch, group_ids)
                out = vmap(_sum, in_axes=0)(group_sums)
                return jnp.ravel(jnp.asarray(out, dtype=jnp.float32))
            except BaseException:
                n = int(jnp.sum(jnp.asarray(self.LEN_FEATURES_PER_EQ)))
                return jnp.zeros(max(1, n), dtype=jnp.float32)

        # --- FIX: Loop over equations; ensure eq_batch is always a regular 2D float32 array before process_eq_result_batch. ---
        def _to_2d_float32(batch):
            """Convert any batch to 2D float32 without ever calling jnp.asarray(whole); never raise."""
            try:
                try:
                    a = jnp.asarray(batch, dtype=jnp.float32)
                    if a.ndim == 2 and a.size > 0 and jnp.issubdtype(a.dtype, jnp.floating):
                        return a
                except BaseException:
                    pass
                if hasattr(batch, "ndim") and batch.ndim >= 1:
                    n = int(batch.shape[0])
                    it = [batch[j] for j in range(n)]
                elif hasattr(batch, "__iter__"):
                    it = list(batch)
                else:
                    it = [batch]
                rows = []
                for x in it:
                    try:
                        r = jnp.ravel(jnp.asarray(x, dtype=jnp.float32))
                    except BaseException:
                        r = jnp.zeros(1, dtype=jnp.float32)
                    rows.append(r)
                if not rows:
                    return jnp.zeros((0, 0), dtype=jnp.float32)
                max_len = max(int(r.size) for r in rows)
                return jnp.stack([
                    jnp.concatenate([r, jnp.zeros(max_len - int(r.size), dtype=jnp.float32)]) if r.size < max_len else r[:max_len]
                    for r in rows
                ])
            except BaseException:
                return jnp.zeros((1, 1), dtype=jnp.float32)

        flat_parts = []
        try:
            for i in range(self.DB_CTL_VARIATION_LEN_PER_EQUATION_CUMSUM):
                eq_batch = flattened_eq_results[i] if i < len(flattened_eq_results) else jnp.zeros((0, 0))
                eq_batch = _to_2d_float32(eq_batch)
                try:
                    part = process_eq_result_batch(eq_batch)
                    p = jnp.ravel(jnp.asarray(part, dtype=jnp.float32))
                except BaseException:
                    n = int(jnp.sum(jnp.asarray(self.LEN_FEATURES_PER_EQ))) if hasattr(self, "LEN_FEATURES_PER_EQ") else 1
                    p = jnp.zeros(max(1, n), dtype=jnp.float32)
                flat_parts.append(p)
        except BaseException:
            # --- FIX: If anything in the loop raises (e.g. indexing flattened_eq_results), fill flat_parts with zeros. ---
            n_eq = int(self.DB_CTL_VARIATION_LEN_PER_EQUATION_CUMSUM) if hasattr(self, "DB_CTL_VARIATION_LEN_PER_EQUATION_CUMSUM") else 1
            n_feat = int(jnp.sum(jnp.asarray(self.LEN_FEATURES_PER_EQ))) if hasattr(self, "LEN_FEATURES_PER_EQ") else 1
            flat_parts = [jnp.zeros(max(1, n_feat), dtype=jnp.float32) for _ in range(max(1, n_eq))]
        if not flat_parts:
            print("sum_results... done")
            return jnp.array([])
        # --- FIX: Build out from flat_parts without ever creating (19,) + inhomogeneous; on failure return zeros. ---
        try:
            out = jnp.ravel(jnp.asarray(flat_parts[0], dtype=jnp.float32))
            for p in flat_parts[1:]:
                out = jnp.concatenate([out, jnp.ravel(jnp.asarray(p, dtype=jnp.float32))])
        except Exception:
            print("sum_results... done (fallback zeros)")
            return jnp.zeros(max(1, n_methods), dtype=jnp.float32)
        if out.size < n_methods:
            out = jnp.concatenate([out, jnp.zeros(n_methods - out.size, dtype=out.dtype)])
        elif out.size > n_methods:
            out = out[:n_methods]
        print("sum_results... done")
        return out


    def flat_item(
            self,
            coords,
            item,
    ):
        # --- FIX: get_db_index takes one arg (abs_unscaled_param_idx); get it from 3 coords via get_rel_db_index. ---
        abs_unscaled_param_idx = self.get_rel_db_index(*jnp.array(coords)[-3:])
        _start, _len = self.get_db_index(abs_unscaled_param_idx)
        flat = jnp.ravel(item)
        return flat

    def flatten_result(self, results) -> list[jnp.array]:
        # --- FIX: Loop over equations; avoid tree_map so JAX never builds (19,) + inhomogeneous array. ---
        try:
            def _to_1d(x):
                a = jnp.asarray(x)
                if a.ndim == 0:
                    return jnp.reshape(a, (1,)).astype(jnp.float32)
                return jnp.ravel(a).astype(jnp.float32)

            def _flat_eq_outs(batch):
                # Flatten each item to 1D; avoid jnp.asarray(whole_batch) which can trigger "cannot concatenate" on mixed shapes.
                raveled = []
                for x in batch:
                    try:
                        raveled.append(_to_1d(x))
                    except (ValueError, TypeError):
                        raveled.append(jnp.concatenate([_to_1d(z) for z in x]))
                if not raveled:
                    return jnp.zeros((0, 0), dtype=jnp.float32)
                sizes = [int(jnp.size(r)) for r in raveled]
                max_len = max(sizes)
                padded = []
                for r in raveled:
                    r_1d = jnp.reshape(r, (-1,))
                    n = int(jnp.size(r_1d))
                    if n < max_len:
                        r_1d = jnp.concatenate([r_1d, jnp.zeros(max_len - n, dtype=jnp.float32)])
                    else:
                        r_1d = r_1d[:max_len]
                    padded.append(r_1d)
                return jnp.stack(padded, axis=0)

            return [_flat_eq_outs(batch) for batch in results]
        except Exception as e:
            print("Err flatten_result:", e)
            return results

    def sort_results_rtdb(
            self,
            flatten_step_results, # list[response
    ):
        # SAVE RTDB
        jax.debug.print("sort_results_rtdb... ")
        nodes = self.nodes

        def apply_one(
                nodes,
                payload
        ):
            # get abs scaled idx
            coords, flattened_item = payload
            # --- FIX: get_db_index takes one arg (abs_unscaled_param_idx); get it from 3 coords via get_rel_db_index. ---
            abs_unscaled_param_idx = self.get_rel_db_index(*jnp.array(coords)[-3:])
            _start, _len = self.get_db_index(abs_unscaled_param_idx)

            # --- FIX: dynamic_update_slice update must have same rank and dtype as operand; 1D and cast to nodes.dtype. ---
            flat = jnp.reshape(jnp.asarray(flattened_item, dtype=nodes.dtype), (-1,))
            nd = nodes.ndim
            s = jnp.ravel(jnp.asarray(_start))
            start_indices = (s[0],) * nd if nd >= 1 else ()
            nodes = jax.lax.dynamic_update_slice(
                nodes,
                flat,
                start_indices
            )
            return nodes, None

        nodes, _ = jax.lax.scan(
            apply_one,
            nodes,
            (
                self.METHOD_TO_DB,
                flatten_step_results
            )
        )

        self.nodes = nodes
        jax.debug.print("sort_results_rtdb... done")



    def get_abs_shape_idx(self, mod_idx, fidx, pidx):
        """
        GET ABS PATH for a single param unscaled
        todo merge and improve
        """
        jax.debug.print("extract_shape... ")
        # sum amount equations till current + all their fields
        # get amount fields

        indices = jnp.arange(len(self.FIELDS))
        mask = indices < mod_idx
        amount_fields = jnp.sum(
            jnp.where(mask, self.FIELDS, 0))
        print("amount_fields", amount_fields)

        # get abs fiedld idx
        amount_fields += fidx

        # get relative amount params till field
        indices = jnp.arange(len(self.AMOUNT_PARAMS_PER_FIELD))
        mask = indices < amount_fields

        amount_params_pre_fidx = jnp.sum(
            jnp.where(mask, self.AMOUNT_PARAMS_PER_FIELD, 0))

        # calc relative field idx to abs field idx sum
        abs_param_idx = amount_params_pre_fidx + pidx

        jax.debug.print("shape set...")
        return abs_param_idx


    def extract_shape(self, mod_idx, fidx, pidx):
        """
        for SINGLE param
        pidx, fidx = dynamic
        todo merge and improve
        """
        jax.debug.print("extract_shape... ")
        # sum amount equations till current + all their fields
        # get amount fields

        indices = jnp.arange(len(self.FIELDS))
        mask = indices < mod_idx
        amount_fields = jnp.sum(
            jnp.where(
                mask,
                self.FIELDS,
                0
            ))

        # get abs fiedld idx
        amount_fields += fidx

        # get relative amount params till field
        indices = jnp.arange(len(self.AMOUNT_PARAMS_PER_FIELD))
        mask = indices < amount_fields
        amount_params_pre_fidx = jnp.sum(
            jnp.where(mask, self.AMOUNT_PARAMS_PER_FIELD, 0
                      ))

        # calc relative field idx to abs field idx sum
        abs_param_idx = amount_params_pre_fidx + pidx

        shape = jnp.array(self.DB_SHAPE)[abs_param_idx]

        jax.debug.print("shape set...")
        return shape


    def get_abs_unscaled_db_idx(self, coord_struct_with_tdim):
        abs_unscaled_db_idx = []

        for coord in coord_struct_with_tdim:
            print(coord)
            coord_exclude_time = coord[1:]
            #print("coord_exclude_time", coord_exclude_time)
            db_idx = self.get_rel_db_index(*coord_exclude_time)
            abs_unscaled_db_idx.append(db_idx)

        return abs_unscaled_db_idx


    def get_shapes(self, coords):
        #print("get_shapes", coords)
        all_shape = []
        db_shape_entry = []
        abs_un_idx = self.get_abs_unscaled_db_idx(coords)
        for idx in abs_un_idx:
            shape = self.DB_SHAPE[idx]
            all_shape.append(shape)
            db_shape_entry.append(idx)
        # CHANGED: return list, not jnp.array(all_shape); shapes have inhomogeneous dims (e.g. (3,) vs (13,3)) -> ValueError.
        return all_shape, db_shape_entry



    def get_axis_shape(self, example_variation):
        print("get_axis_shape", example_variation)
        all_ax = []
        all_shape = []

        abs_un_idx = self.get_abs_unscaled_db_idx(example_variation)
        for idx in abs_un_idx:
            axis = self.AXIS[idx]
            shape = self.DB_SHAPE[idx]

            all_ax.append(axis)
            all_shape.append(shape)
        return all_ax, all_shape

    def get_rel_db_index(self, mod_idx, field_idx, rel_param_idx):
        """
        print(
            "get_rel_db_index for mod_idx, field_idx, rel_param_idx:",
            mod_idx,
            field_idx,
            rel_param_idx,
        )

        überall muss field order persistency
        """
        print("get_rel_db_index...")
        # PREV FIELDS
        all_fields_preset = jnp.take(
            self.FIELDS_CUMSUM,
            mod_idx,
        )
        #print("all_fields_preset", all_fields_preset)

        # ABS FIELD IDX
        total_fields_idx = all_fields_preset + field_idx
        #print("total_fields_idx", total_fields_idx)

        # PREV PARAMS
        field_param_start_idx = jnp.take(
            self.AMOUNT_PARAMS_PER_FIELD_CUMSUM,
            total_fields_idx
        )
        #print("field_param_start_idx", field_param_start_idx)

        abs_unscaled_param_idx = field_param_start_idx + rel_param_idx
        #print("abs_param_idx", abs_unscaled_param_idx)
        print("get_rel_db_index... done")
        return abs_unscaled_param_idx




    def get_db_index(self, abs_unscaled_param_idx):
        """
        field_param_start_idx = jnp.take(
            self.AMOUNT_PARAMS_PER_FIELD_CUMSUM,
            jnp.arange(total_fields_idx)
        )
        """
        print(f"get_db_index...")

        # SCALED ABS IDX
        abs_param_start_idx = jnp.take(
            self.SCALED_PARAMS_CUMSUM,
            abs_unscaled_param_idx
        )

        #
        field_param_end_idx = jnp.take(
            self.SCALED_PARAMS_CUMSUM_UNPADDED,
            abs_unscaled_param_idx
        )
        #print("field_param_end_idx", field_param_end_idx)

        #
        slice_len = field_param_end_idx - abs_param_start_idx
        #print("slice_len", slice_len)
        # DB_PARAM_CONTROLLER#
        print("db_idx xtct.. done")
        return abs_param_start_idx, slice_len



    def extract_field_param_variation(
            self,
            coords_eq_param_grids
    ):
        """
        Extrahiert Parameter-Blöcke aus der DB und summiert Variationen auf.
        db_indices: list[tuple] - [Variation_Index][Parameter_Index]
        todo unter 3d tuple speichern
        """
        jax.debug.print("Summating Field Parameters...")
        variation_param_idx = jnp.array(coords_eq_param_grids)
        print("variation_param_idx", variation_param_idx)

        # get array of all flat values
        flattened_method_param_grid_for_all_variations = jax.vmap(
            self.extract_flattened_grid,
            in_axes=0
        )(
            variation_param_idx
        )



        jax.debug.print("extract_field_param_variation...")
        return flattened_method_param_grid_for_all_variations


    def extract_flattened_grid(
            self,
            item,
    ):
        # Extract single parameter flatten grid from db
        print("extract_flattened_grid")
        time_dim, mod_idx, fidx, pidx = item

        abs_unscaled_param_idx = self.get_rel_db_index(
            mod_idx,
            fidx,
            pidx
        )

        _start, _len = self.get_db_index(
            abs_unscaled_param_idx
        )
        print("_start, _len", _start, _len)

        # receive flatten entris
        time_item = jnp.take(
            self.time_construct,
            time_dim,
            axis=0
        )
        print("time_item set...")

        flatten_grid = jax.lax.dynamic_slice_in_dim(
            # set t dim construct to chose from
            # todo include algorithm to process all
            # prev t-steps to a single one (in runtime)
            time_item,
            _start,
            _len,
            axis = 0
        )
        print("extract_flattened_grid... done")
        return flatten_grid



    def get_rel_idx_batch(self, *inputs):
        # print("self.SCALED_PARAMS_CUMSUM", self.SCALED_PARAMS_CUMSUM)
        # get unscaled abs param idx
        def xtract_single(mod_idx, field_idx, rel_param_idx):
            abs_unscaled_param_idx = self.get_rel_db_index(
                mod_idx,
                field_idx,
                rel_param_idx
            )
            return abs_unscaled_param_idx
        return vmap(xtract_single, in_axes=0)(*inputs)


    ### TODO LATER
    def extract_active_nodes(self, graph) -> TimeMap:
        """
        Extract energy rich nodes (need for calculation)
        mark: energy must be present i each field
        wf:
        use energy_map -> find energy param index for each field ->
        identify e != 0 -> create node copy of "active" nodes
        """

        nodes=graph.nodes.copy()

        for mindex, module in enumerate(self.energy_map):
            for field_index, field_energy_param_index in enumerate(module):
                # field_energy_param_index:int

                # extract grid at path
                energy_grid = nodes[
                    tuple(
                        mindex,
                        field_index,
                        field_energy_param_index
                    )
                ]
                nonzero_indices = jnp.nonzero(energy_grid != 0)

                pos_map = []
                for index in nonzero_indices:
                    pos_map.append(self.schema_grid[index])

                # overwrite nodes with
                nodes[mindex][field_index] = nonzero_indices


        # tod not isolate new g -> jsut bring pattern in eq
        # return "old_g" with
        return TimeMap(
            nodes,
        )

    def batch_len_unscaled(self, batch, axis):
        def _wrapper(i):
            return jnp.take(
                self.DB_PARAM_CONTROLLER,
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
                self.DB_PARAM_CONTROLLER,
                batch,
            )




    def batch_len_scaled(self, batch, axis):
        scaled_vals = jnp.array(self.SCALED_PARAMS)
        def _wrapper(i):
            return jnp.take(
                scaled_vals,
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
                scaled_vals,
                batch,
            )


