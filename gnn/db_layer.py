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
        self.FIELDS = jnp.array(FIELDS)

        self.METHODS_PER_MOD_LEN_CTLR = jnp.array(METHODS_PER_MOD_LEN_CTLR)
        self.METHOD_TO_DB = jnp.array(METHOD_TO_DB)
        self.DB_TO_METHOD_EDGES = jnp.array(DB_TO_METHOD_EDGES)

        # convert bytes array

        self.SCALED_PARAMS:list[int] = []

        self.gpu = gpu
        self.DIMS=DIMS


        self.history_nodes = [] # Collects state over time

        self.build_db(
            amount_nodes,
        )



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




    def sort_results(self, step_results):
        """
        - save coords to paste vals

        # RESULT -> HISTORY DB todo: upsert directly to bq
        # todo: add param stacks on single eq layer
        n variations = n return
        """
        jax.debug.print("sort_results... ")

        def _update(res, coords):
            _start, _len = self.get_db_index(*coords)
            jax.lax.dynamic_update_slice(
                self.nodes,
                res,
                _start,
            )

        vmap(
            _update,
            in_axes=(0,0)
        )(
            step_results,
            self.METHOD_TO_DB
        )
        jax.debug.print("sort_results... done")



    def pad_param(self, p):
        # Macht aus jeder Liste/Array ein Array der Länge MAX_WIDTH
        p = jnp.atleast_1d(jnp.array(p))
        return jnp.pad(p, (0, self.db_padding - len(p)))


    def build_db(self, amount_nodes):
        # receive 1d array _> scale each qx 0 for n nodes
        jax.debug.print("build_db...")

        SCALED_DB = []

        # scale db nodes
        for i, (len_unscaled_param, ax) in enumerate(
                zip(self.DB_PARAM_CONTROLLER, self.AXIS)
        ):
            print("len_unscaled_param", len_unscaled_param)
            # len_unscaled_param:int = len unscaled single param space
            prev = int(
                jnp.sum(
                    jnp.array(self.DB_PARAM_CONTROLLER[:i])
                )
            )

            # extract param space from DB
            # e.g. prov = 0
            # len_unscaled_param = 4
            # exct pos 0-4 (0000) from DB = param
            single_param_value = jax.lax.dynamic_slice_in_dim(
                self.DB,
                prev, # start e.g. at 0

                # set end for single param
                len_unscaled_param,
            )

            # check scale value -> add nodes
            if ax == 0:
                slice = jnp.tile(
                    single_param_value,
                    (
                        amount_nodes*self.DIMS
                    )
                )

                print(
                    f"extend {len_unscaled_param} to ({len(slice)})"
                )

                SCALED_DB.extend(slice)

                self.SCALED_PARAMS.append(
                    len(slice)
                )
            else:
                # const
                SCALED_DB.append(len_unscaled_param)
                self.SCALED_PARAMS.append(len_unscaled_param)

        # scale nodes down db
        nodes = jnp.array(SCALED_DB)
        self.SCALED_PARAMS = jnp.array(self.SCALED_PARAMS)
        #print("self.SCALED_PARAMS created", self.SCALED_PARAMS)

        self.nodes = jax.device_put(nodes, self.gpu)
        jax.debug.print("build_db... done")


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
        amount_fields = jnp.sum(jnp.where(mask, self.FIELDS, 0))
        print("amount_fields", amount_fields)

        # get abs fiedld idx
        amount_fields += fidx

        # get relative amount params till field
        indices = jnp.arange(len(self.AMOUNT_PARAMS_PER_FIELD))
        mask = indices < amount_fields
        amount_params_pre_fidx = jnp.sum(jnp.where(mask, self.AMOUNT_PARAMS_PER_FIELD, 0))

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
        amount_fields = jnp.sum(jnp.where(mask, self.FIELDS, 0))
        print("amount_fields", amount_fields)

        # get abs fiedld idx
        amount_fields += fidx

        # get relative amount params till field
        indices = jnp.arange(len(self.AMOUNT_PARAMS_PER_FIELD))
        mask = indices < amount_fields
        amount_params_pre_fidx = jnp.sum(jnp.where(mask, self.AMOUNT_PARAMS_PER_FIELD, 0))

        # calc relative field idx to abs field idx sum
        abs_param_idx = amount_params_pre_fidx + pidx

        shape = jnp.array(self.DB_SHAPE)[abs_param_idx]

        jax.debug.print("shape set...")
        return shape

    def get_axis_shape(self, example_variation):
        print("example_variation", example_variation)
        all_ax = []
        all_shape = []
        for coord in example_variation:
            db_idx = self.get_rel_db_index(*coord)
            axis = self.AXIS[db_idx]
            shape = self.DB_SHAPE[db_idx]
            all_ax.append(axis)
            all_shape.append(shape)
        return all_ax, all_shape

    def get_rel_db_index(self, mod_idx, field_idx, param_in_field_idx):
        print("get_rel_db_index for mod_idx, field_idx, param_in_field_idx:", mod_idx, field_idx, param_in_field_idx)
        all_fields_preset = sum(self.FIELDS[:mod_idx])
        total_fields_idx = all_fields_preset + field_idx
        print("total_fields_idx", total_fields_idx)

        all_params_preset = len(self.AMOUNT_PARAMS_PER_FIELD[:total_fields_idx])
        rel_param_idx = all_params_preset + 1  # ü1 becaue current len


        return rel_param_idx

    def get_db_index(self, mod_idx, field_idx, param_in_field_idx):
        """
        field_param_start_idx = jnp.take(
            AMOUNT_PARAMS_PER_FIELD_CUMSUM,
            jnp.arange(total_fields_idx)
        )
        """
        print("get_db_index for mod_idx, field_idx, param_in_field_idx:", mod_idx, field_idx, param_in_field_idx)
        FIELDS_CUMSUM = jnp.cumsum(self.FIELDS)
        AMOUNT_PARAMS_PER_FIELD_CUMSUM = jnp.cumsum(
            self.AMOUNT_PARAMS_PER_FIELD
        )
        SCALED_PARAMS_CUMSUM = jnp.cumsum(
            jnp.array(self.SCALED_PARAMS)
        )

        #all_fields_preset = jnp.take(FIELDS_CUMSUM, jnp.arange(mod_idx))
        all_fields_preset = jnp.where(
            mod_idx >= 0,
            FIELDS_CUMSUM[mod_idx-1],
            0
        )
        total_fields_idx = all_fields_preset + field_idx
        print("total_fields_idx", total_fields_idx)


        field_param_start_idx = jnp.where(
            total_fields_idx >= 0,
            AMOUNT_PARAMS_PER_FIELD_CUMSUM[total_fields_idx-1],
            0
        )

        abs_param_idx = field_param_start_idx + param_in_field_idx
        print("abs_param_idx", abs_param_idx)

        abs_param_start_idx = jnp.where(
            abs_param_idx >= 0,
            SCALED_PARAMS_CUMSUM[abs_param_idx - 1],
            0 # first field
        )

        print("abs_param_start_idx", abs_param_start_idx)

        field_param_end_idx = jnp.where(
            abs_param_idx >= 0,
            SCALED_PARAMS_CUMSUM[abs_param_idx],
            0
        )
        print("field_param_end_idx", field_param_end_idx)

        slice_len = field_param_end_idx - abs_param_start_idx
        print("slice_len", slice_len)
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


    def extract_flattened_grid(self, item):
        # Extract single parameter flatten grid from db
        print("extract_flattened_grid", item)
        mod_idx, fidx, pidx = item
        print("extract_flattened_grid with mod_idx, fidx, pidx", mod_idx, fidx, pidx)
        # extract scaled idx. from db
        _start, _len = self.get_db_index(
            mod_idx, fidx, pidx
        )

        # receive flatten entris
        flatten_grid = jax.lax.dynamic_slice_in_dim(
            self.nodes,
            _start,
            _len,
        )
        print("flatten_grid", flatten_grid)
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



"""

# todo check instance str -> check if in params -> replace
        # 1. Scan dimensions
        max_m = len(db)
        max_f = 0
        max_p = 0
        max_feat = 0

        # Pre-scan for dimensions
        for m_fields in db:
            if m_fields is None: continue
            max_f = max(max_f, len(m_fields))
            for item in m_fields:
                if item is None or len(item) != 2: continue
                field, axis_def = item
                if isinstance(axis_def, str):
                    try:
                        axis_def = json.loads(axis_def)
                    except Exception as e:
                        axis_def = []
                if axis_def is None: axis_def = []

                # Count params (where axis is not None)
                p_count = 0
                for value, ax in zip(field, axis_def):
                    if ax is not None:
                        p_count += 1
                        if isinstance(value, list):
                            max_feat = max(max_feat, len(value))
                        else:
                            max_feat = max(max_feat, 1)  # scalar
                max_p = max(max_p, p_count)

        print(f"DB Shapes: M={max_m}, F={max_f}, P={max_p}, N={amount_nodes}, Feat={max_feat}")

        # 2. Allocate array
        # Shape: [M, F, P, N, Feat]
        nodes = jnp.zeros((max_m, max_f, max_p, amount_nodes, max_feat))

        # 3. Fill
        m_idx = 0
        for m_fields in db:
            if m_fields is None:
                m_idx += 1
                continue

            f_idx = 0
            for item in m_fields:
                if item is None or len(item) != 2: continue
                field, axis_def = item
                if isinstance(axis_def, str):
                    try:
                        axis_def = json.loads(axis_def)
                    except:
                        axis_def = []
                if axis_def is None: axis_def = []

                p_idx = 0
                for value, ax in zip(field, axis_def):
                    if ax is not None:
                        val_arr = jnp.array(value)
                        # Ensure shape is (Feat,)
                        if val_arr.ndim == 0:
                            val_arr = val_arr.reshape(1)

                        # Pad features if needed logic is handled by slicing assignment?
                        # JAX arrays key assignment: nodes.at[...].set(...)
                        # We want to set vector val_arr to all N nodes
                        # nodes[m, f, p, :, 0:len] = val_arr

                        current_feat_len = val_arr.shape[0]

                        # Construct update
                        # We need to broadcast val_arr to (N, current_feat_len)
                        tiled = jnp.tile(val_arr, (amount_nodes, 1))

                        nodes = nodes.at[m_idx, f_idx, p_idx, :, :current_feat_len].set(tiled)

                        p_idx += 1
                f_idx += 1
            m_idx += 1



#####


    def get_db_index(self, mod_idx, field_idx, param_in_field_idx):
      
        convert relative indexing to scaled (with extend with focus on simplicity)

        TAKE
        jax.errors.TracerIntegerConversionError: The __index__() method was called on traced array with shape int32[].
This    BatchTracer with object id 1965655827376 was created on line:
       

        # get amount fields
        amount_fields = jnp.sum(self.FIELDS[:mod_idx])

        # get abs fiedld idx
        amount_fields += field_idx

        # get relative amount params till field
        amount_params_pre_fidx = jnp.sum(self.AMOUNT_PARAMS_PER_FIELD[:amount_fields])

        # calc relative field idx to abs field idx sum
        abs_param_idx = amount_params_pre_fidx + param_in_field_idx

        # GET RUNTIME INDEX FROM PARAM SCALED
        abs_param_db_index_scaled = jnp.sum(self.SCALED_PARAMS[:abs_param_idx])

        len_scaled_param = self.SCALED_PARAMS[abs_param_idx]

        return abs_param_db_index_scaled, len_scaled_param



    def get_db_index(self, mod_idx, field_idx, param_in_field_idx):
        print("get_db_index for mod_idx, field_idx, param_in_field_idx:", mod_idx, field_idx, param_in_field_idx)
        all_fields_preset = sum(self.FIELDS[:mod_idx])
        total_fields_idx = all_fields_preset + field_idx
        print("total_fields_idx", total_fields_idx)

        all_params_preset = len(self.AMOUNT_PARAMS_PER_FIELD[:total_fields_idx])
        rel_param_idx = all_params_preset + 1 # ü1 becaue current len

        return scaled_param_len, abs_pram_len


        # todo stack on param level - save module base ctlr
        # use unscaled idx to get scaled
        scaled_param_len = sum(self.SCALED_PARAMS[:rel_param_idx])
        abs_pram_len = self.SCALED_PARAMS[rel_param_idx]
        
        all_offsets = jnp.cumsum(jnp.array(self.AMOUNT_PARAMS_PER_FIELD))
        amount_fields_base = jnp.argmax(all_offsets < total_fields_idx)
        print("amount_fields_base", amount_fields_base)
        # Absoluter Parameter Index
        abs_param_idx = amount_params_pre_fidx + param_in_field_idx
        print("abs_param_idx", abs_param_idx)

        # 3. Skalierte Werte berechnen
        # Maske für SCALED_PARAMS
        mask_scaled = jnp.arange(jnp.array(self.SCALED_PARAMS).shape[0]) < abs_param_idx
        abs_param_db_index_scaled = jnp.sum(jnp.where(mask_scaled, jnp.array(self.SCALED_PARAMS), 0))
        print("abs_param_db_index_scaled", abs_param_db_index_scaled)

        len_scaled_param = jnp.take(jnp.array(self.SCALED_PARAMS), abs_param_idx)
        print("len_scaled_param", len_scaled_param)
        
        return scaled_param_len, abs_pram_len
        
        
        
    def get_db_index(self, mod_idx, field_idx, param_in_field_idx):
        #print("get_db_index for mod_idx, field_idx, param_in_field_idx:", mod_idx, field_idx, param_in_field_idx)
        FIELDS_CUMSUM = jnp.cumsum(self.FIELDS)
        AMOUNT_PARAMS_PER_FIELD_CUMSUM = jnp.cumsum(self.AMOUNT_PARAMS_PER_FIELD)
        SCALED_PARAMS_CUMSUM = jnp.cumsum(jnp.array(self.SCALED_PARAMS))

        all_fields_preset = jnp.sum(jnp.take(FIELDS_CUMSUM, jnp.arange(mod_idx)))
        print("all_fields_preset", all_fields_preset)

        total_fields_idx = all_fields_preset + field_idx
        print("total_fields_idx", total_fields_idx)

        field_param_start_idx = jnp.sum(jnp.take(AMOUNT_PARAMS_PER_FIELD_CUMSUM, jnp.arange(total_fields_idx)))

        abs_param_idx = field_param_start_idx + param_in_field_idx

        field_param_start_idx = jnp.sum(
            jnp.take(
                SCALED_PARAMS_CUMSUM,
                jnp.arange(total_fields_idx-1)
            )
        )

        field_param_end_idx = jnp.sum(
            jnp.take(
                SCALED_PARAMS_CUMSUM,
                jnp.arange(total_fields_idx)
            )
        )

        return field_param_start_idx, field_param_end_idx - field_param_start_idx




    def get_db_index(self, mod_idx, field_idx, param_in_field_idx):
        # WICHTIG: self.FIELDS, self.SCALED_PARAMS etc. sollten jnp.arrays sein.
        # Wir wandeln sie hier zur Sicherheit um, idealerweise machst du das im __init__.
        fields = jnp.array(self.FIELDS)
        scaled_params = jnp.array(self.SCALED_PARAMS)

        # 1. all_fields_preset = sum(self.FIELDS[:mod_idx])
        # Wir berechnen die kumulative Summe und verschieben sie, damit der
        # Index 0 auch 0 ergibt (daher das jnp.roll oder jnp.pad).
        fields_cumsum = jnp.cumsum(fields)
        # Nutze jnp.where, um das Slicing zu simulieren:
        all_fields_preset = jnp.sum(jnp.where(jnp.arange(fields.shape[0]) < mod_idx, fields, 0))

        total_fields_idx = all_fields_preset + field_idx

        # 2. all_params_preset = len(self.AMOUNT_PARAMS_PER_FIELD[:total_fields_idx])
        # In JAX ist die Länge eines Slices [:n] einfach n.
        all_params_preset = total_fields_idx
        rel_param_idx = all_params_preset + 1

        # 3. scaled_param_len = sum(self.SCALED_PARAMS[:rel_param_idx])
        # Wieder Maskierung statt Slicing
        mask_scaled = jnp.arange(scaled_params.shape[0]) < rel_param_idx
        scaled_param_len = jnp.sum(jnp.where(mask_scaled, scaled_params, 0))

        # 4. abs_pram_len = self.SCALED_PARAMS[rel_param_idx]
        # In vmap nutzt man jnp.take für dynamische Indizes
        abs_pram_len = jnp.take(scaled_params, rel_param_idx)

        return scaled_param_len, abs_pram_len

"""

"""

FIELDS_CUMSUM = jnp.cumsum(self.FIELDS)
AMOUNT_PARAMS_PER_FIELD_CUMSUM = jnp.cumsum(self.AMOUNT_PARAMS_PER_FIELD)
SCALED_PARAMS_CUMSUM = jnp.cumsum(jnp.array(self.SCALED_PARAMS))

all_fields_preset = jnp.where(
mod_idx > 0,
FIELDS_CUMSUM[mod_idx - 1],
0
)

total_fields_idx = all_fields_preset + field_idx

field_param_start_idx = jnp.where(
total_fields_idx > 0,
SCALED_PARAMS_CUMSUM[total_fields_idx - 1],
0
)

field_param_end_idx = SCALED_PARAMS_CUMSUM[total_fields_idx]

return field_param_start_idx, field_param_end_idx - field_param_start_idx
"""



