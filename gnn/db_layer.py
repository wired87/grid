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


        self.in_features=[]




    def build_db(self, amount_nodes):
        # receive 1d array _> scale each qx 0 for n nodes
        jax.debug.print("build_db...")

        SCALED_DB = []
        TIME_DB = []
#

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
        # append new items to tdb which gets upserted to bq after finished sim
        """"
        receive nested sumed structures
        vmap m2db
        group arrays receive out
        paste vals
        overwrite 
        """

        def get_rel_db_idx(coord):
            rel_idx = self.get_rel_db_index(
                *coord[-3:]
            )
            return rel_idx

        def stack_tstep(arr, val):
            return jnp.stack([arr, val], axis=0)


        def extract_out_arrays(idx) -> jnp.array:
            return self.tdb[idx]

        try:
            # change dest
            idx_map = vmap(
                get_rel_db_idx,
                in_axes=(0,0)
            )(
                self.METHOD_TO_DB,
            )

            #
            arrays = vmap(
                extract_out_arrays,
                in_axes=0,
            )(
                idx_map
            )

            arrays = vmap(
                stack_tstep,
                in_axes=(0,0)
            )(
                arrays, sumed_results,
            )

            # apply change
            def _sort(node, cary):
                arr, idx = cary
                node[idx] = arr
                return node, None

            self.tdb, _ = lax.scan(
                _sort,
                self.tdb,
                (arrays, idx_map)
            )



            print("stack_tdb... done")
        except Exception as e:
            print("Err stack_tdb", e)

    def save_t_step(self, all_results):
        try:
            jax.debug.print(f"save_t_step...")

            #
            all_results = self.flatten_result(all_results)

            # sum field variations
            sumed_results = self.sum_results(all_results)

            # save NOW db in PREV db
            self.time_construct = self.time_construct.at[1].set(
                self.time_construct[0]
            )

            #
            self.stack_tdb(sumed_results)

            # SAVE RT DB (sort_results_rtdb takes step_results only)
            # edge m->db jsut takes here account
            self.sort_results_rtdb(sumed_results)

            # save new nodes (updated in sort) in db construct
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
        # scan each metods outs
        # sepparate into fields variation size blocks
        # sum the result
        # sort to db
        print("sum_results...")

        _arange = jnp.arange(len(self.DB_CTL_VARIATION_LEN_PER_FIELD))

        def _sum(_slice):
            flatten_sum = jnp.sum(jnp.array(_slice))
            print("flatten_sum", flatten_sum)
            return flatten_sum

        def process_eq_result_batch(eq_out_batch):
            print("process_eq_result_batch...")

            # separate output struct
            # into field variation blocks
            len_f_per_eq = len(self.LEN_FEATURES_PER_EQ)
            group_ids = jnp.repeat(
                jnp.arange(len_f_per_eq),
                self.LEN_FEATURES_PER_EQ
            )
            group_sums = ops.segment_sum(eq_out_batch, group_ids)

            flattened_sum_results = vmap(
                _sum,
                in_axes=(0, 0)
            )(group_sums)
            return flattened_sum_results

        eq_variation_len = jnp.arange(
            self.DB_CTL_VARIATION_LEN_PER_EQUATION_CUMSUM)

        flattened_sum_results = jax.tree_util.tree_map(
            process_eq_result_batch,
            (
                flattened_eq_results,
                eq_variation_len
            ),
        )
        print("sum_results... done")
        return flattened_sum_results


    def flat_item(
            self,
            coords,
            item,
    ):
        _start, _len = self.get_db_index(
            *jnp.array(coords)[-3:]
        )
        flat = jnp.ravel(item)
        return flat

    def flatten_result(self, results) -> list[jnp.array]:
        # tkes bare nersted results, outputs flaten nested items
        try:
            # loop each element -> apply ravel
            def flat_item(item):
                return jnp.ravel(item)

            def _flat_eq_outs(batch):
                return vmap(flat_item, in_axes=0)(batch)

            return jax.tree_util.tree_map(_flat_eq_outs, results)
        except Exception as e:
            print("Err flatten_result:", e)

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

            #
            _start, _len = self.get_db_index(
                *jnp.array(coords)[-3:]
            )

            # apply slice: start_indices must be tuple of length nodes.ndim (avoids "axis 1 is out of bounds for array of dimension 1")
            nd = nodes.ndim
            s = jnp.ravel(jnp.asarray(_start))
            start_indices = (s[0],) * nd if nd >= 1 else ()
            nodes = jax.lax.dynamic_update_slice(
                nodes,
                flattened_item,
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
        print("get_shapes", coords)
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
        return abs_unscaled_param_idx




    def get_db_index(self, abs_unscaled_param_idx):
        """
        field_param_start_idx = jnp.take(
            self.AMOUNT_PARAMS_PER_FIELD_CUMSUM,
            jnp.arange(total_fields_idx)
        )
        """
        #print("get_db_index for mod_idx, field_idx, rel_param_idx:", mod_idx, field_idx, rel_param_idx)

        # SCALED ABS IDX
        abs_param_start_idx = jnp.take(
            self.SCALED_PARAMS_CUMSUM,
            abs_unscaled_param_idx
        )

        #print("abs_param_start_idx", abs_param_start_idx)

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
        )(variation_param_idx)

        jax.debug.print("extract_field_param_variation...")
        return flattened_method_param_grid_for_all_variations


    def extract_flattened_grid(
            self,
            item,
    ):
        # Extract single parameter flatten grid from db
        #print("extract_flattened_grid", item)
        time_dim, mod_idx, fidx, pidx = item

        abs_unscaled_param_idx = self.get_rel_db_index(
            mod_idx,
            fidx,
            pidx
        )

        _start, _len = self.get_db_index(
            abs_unscaled_param_idx
        )
        #print("extract flattene grid _start, _len", _start, _len)

        # receive flatten entris
        time_item = jnp.take(self.time_construct, time_dim, axis=0)
        #print("time_item", time_item)

        flatten_grid = jax.lax.dynamic_slice_in_dim(
            # set t dim construct to chose from
            # todo include algorithm to process all
            # prev t-steps to a single one (in runtime)
            time_item,
            _start,
            _len,
            axis = 0
        )
        #print("flatten_grid", flatten_grid)
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



"""
for i, flatten_arr in enumerate(flatten_step_results):
    coords = self.METHOD_TO_DB[i]
    _start, _len = self.get_db_index(*jnp.array(coords)[-3:])
    self.nodes = jax.lax.dynamic_update_slice(self.nodes, flatten_arr, _start)

"""





