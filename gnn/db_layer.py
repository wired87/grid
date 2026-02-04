import jax
import jax.numpy as jnp
from jax import jit, vmap

from dtypes import TimeMap


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

        self.gpu = gpu
        self.DIMS=DIMS

        self.history_nodes = [] # Collects state over time

        self.FIELDS_CUMSUM = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(jnp.array(self.FIELDS))
        ])

        # Dasselbe für die unskalierten Indizes, falls nötig
        self.AMOUNT_PARAMS_PER_FIELD_CUMSUM = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(self.AMOUNT_PARAMS_PER_FIELD)
        ])

        # Dasselbe für die unskalierten Indizes, falls nötig
        self.DB_PARAM_CONTROLLER_CUMSUM = jnp.concatenate([
            jnp.array([0]),
            jnp.cumsum(self.DB_PARAM_CONTROLLER)
        ])






    def check2(self):
        # Nutze ein Set aus Tuples für den Vergleich,
        # da man JAX-Arrays nicht direkt in Listen suchen kann.
        worked_cords = set()
        print("DB", self.DB.tolist())
        print("self.nodes", self.nodes.tolist())
        print("p ctlr unscaled", self.DB_PARAM_CONTROLLER.tolist(), len(self.DB_PARAM_CONTROLLER.tolist()))
        print("p ctlr scaled", jnp.array(self.SCALED_PARAMS).tolist(), len(jnp.array(self.SCALED_PARAMS).tolist()))

        for coords in self.DB_TO_METHOD_EDGES:
            # 1. Konvertiere das Array in ein Python-Tuple
            c_tuple = tuple(coords.tolist())

            # 2. Prüfe, ob dieses Tuple bereits bearbeitet wurde
            if c_tuple in worked_cords:
                continue

            # 3. Markiere es als bearbeitet
            worked_cords.add(c_tuple)

            # Restlicher Code bleibt gleich, aber wir nutzen *coords (entpacktes Array)
            abs_unscaled_param_idx = self.get_rel_db_index(*coords)

            param_start = self.DB_PARAM_CONTROLLER_CUMSUM[abs_unscaled_param_idx]
            param_len = self.DB_PARAM_CONTROLLER[abs_unscaled_param_idx]

            single_param_value = jax.lax.dynamic_slice_in_dim(
                self.nodes,
                param_start,
                param_len,
            )

            abs_param_start_idx, slice_len = self.get_db_index(
                *coords
            )

            scaled_param_value = jax.lax.dynamic_slice_in_dim(
                self.nodes,
                abs_param_start_idx,
                slice_len
            )

            print(f"{coords}")
            print("unscaled param value:", param_start, param_len, single_param_value)
            print("scaled param value:", abs_param_start_idx, slice_len, scaled_param_value)
            print("axis param :", self.AXIS[abs_unscaled_param_idx])
            print("axis param index:", abs_unscaled_param_idx)


    def build_db(self, amount_nodes):
        # receive 1d array _> scale each qx 0 for n nodes
        jax.debug.print("build_db...")

        SCALED_DB = []
        TIME_DB = []
        # get abs sum from each parameter for index

        # scale db nodes
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
        self.OUT_SHAPES = self.get_shapes(
            coords=self.METHOD_TO_DB
        )


        jax.debug.print("build_db... done")


    def build_history_db(self, all_results):

        """
        for padded_idx, unpadded_idx in zip(
                self.SCALED_PARAMS_CUMSUM,
                self.SCALED_PARAMS_CUMSUM_UNPADDED
        ):
        """






    def save_t_step(self, all_results):
        jax.debug.print(f"save_t_step...")

        # save PREV DB
        self.time_construct = self.time_construct.at[1].set(self.nodes)


        for i, (coord, result) in enumerate(zip(
                self.METHOD_TO_DB,
                all_results
        )):
            rel_idx = self.get_rel_db_index(*coord)
            jnp.stack(self.tdb[rel_idx])

        # SAVE RT DB
        self.sort_results(all_results, result)

        # save now
        self.time_construct = self.time_construct.at[0].set(self.nodes)

        #
        jax.debug.print(f"save_t_step... done")

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


    def get_out_shapes(self):
        pass


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



    def sort_results(
            self,
            step_results, # list[response
    ):
        """
        # RESULT -> HISTORY DB todo: upsert directly to bq
        # todo: add param stacks on single eq layer
        n variations = n return
        """
        jax.debug.print("sort_results... ")
        """nodes = self.nodes

        def apply_one(
                nodes,
                payload
        ):
            coords, item = payload
            flat = self.flat_item(
                coords,
                item,
            )
            nodes = jax.lax.dynamic_update_slice(
                nodes,
                flat,
                _start
            )
            return nodes, None

        nodes, _ = jax.lax.scan(
            apply_one,
            nodes,
            (
                self.METHOD_TO_DB,
                step_results
            )
        )
        for i, res_grid in step_results:
            
        

        self.nodes = nodes"""



        for i, res in enumerate(step_results):
            coords = self.METHOD_TO_DB[i]
            _start, _len = self.get_db_index(*jnp.array(coords)[-3:])
            res = step_results[i]

            # equation may return single array or (array, list); DB write expects one array
            self.nodes = jax.lax.dynamic_update_slice(self.nodes, res, _start)


        jax.debug.print("sort_results... done")



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
        return  abs_unscaled_db_idx

    def get_shapes(self, coords):
        print("get_shapes", coords)
        all_shape = []

        abs_un_idx = self.get_abs_unscaled_db_idx(coords)
        for idx in abs_un_idx:
            shape = self.DB_SHAPE[idx]
            all_shape.append(shape)
        return all_shape





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


    def get_db_index(self, mod_idx, field_idx, rel_param_idx):
        """
        field_param_start_idx = jnp.take(
            self.AMOUNT_PARAMS_PER_FIELD_CUMSUM,
            jnp.arange(total_fields_idx)
        )
        """
        #print("get_db_index for mod_idx, field_idx, rel_param_idx:", mod_idx, field_idx, rel_param_idx)
        
        #print("self.SCALED_PARAMS_CUMSUM", self.SCALED_PARAMS_CUMSUM)
        # get unscaled abs param idx
        abs_unscaled_param_idx = self.get_rel_db_index(
            mod_idx,
            field_idx,
            rel_param_idx
        )

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

        _start, _len = self.get_db_index(
            mod_idx,
            fidx,
            pidx,
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


