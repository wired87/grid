import json
import os

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, vmap

from dtypes import TimeMap
from jax_utils.conv_flat_to_shape import bring_flat_to_shape
from test import get_context


class DBLayer(nn.Module):


    def  __init__(
            self,
            amount_nodes,
            gpu,
            module_struct,
            method_to_db,
            AXIS,
            DB,
    ):
        # DB Load and parse
        super().__init__()
        self.db_struct = json.loads(DB)
        self.AXIS = json.loads(AXIS)

        self.module_struct = module_struct

        for k,v in self.db_struct.items():
            setattr(self, k, json.loads(v))

        # convert bytes array
        self.DB = jnp.array(
            jnp.frombuffer(
                self.DB,
                dtype=jnp.complex64
            )
        )

        self.SCALED_PARAMS:list[int] = []

        self.gpu = gpu

        self.build_db(
            amount_nodes,
        )

        self.method_to_db = method_to_db

        self.old_g = self.nodes
        self.history_nodes = [] # Collects state over time




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
        axis_rule = self.db_layer.axis[db_start_idx]

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

        self.db_layer.new_g = jax.lax.cond(
            axis_rule == 0,
            update_field_block,
            update_single_value,
            self.db_layer.new_g,
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
            self.method_to_db
        )
        jax.debug.print("sort_results... done")



    def pad_param(self, p):
        # Macht aus jeder Liste/Array ein Array der Länge MAX_WIDTH
        p = jnp.atleast_1d(jnp.array(p))
        return jnp.pad(p, (0, self.db_padding - len(p)))


    def build_db(self, amount_nodes):
        # receive 1d array _> scale each qx 0 for n nodes
        jax.debug.print("build_db...")

        nodes = []

        # scale db nodes
        for i, (n_space, ax) in enumerate(zip(self.DB_PARAM_CONTROLLER, self.AXIS)):
            prev = jnp.sum(self.DB_PARAM_CONTROLLER[:i])

            # extract param space
            param = jax.lax.dynamic_slice_in_dim(
                self.DB,
                prev,
                n_space,
            )

            if ax == 0:
                slice = jnp.tile(
                    param,
                    (amount_nodes,)
                )
                nodes.extend(slice)
                self.SCALED_PARAMS.append(len(slice))
            else:
                # const
                nodes.append(n_space)
                self.SCALED_PARAMS.append(len(n_space))

        # scale nodes down db
        nodes = jnp.array(nodes)
        self.nodes = jax.device_put(nodes, self.gpu)
        jax.debug.print("build_db... done")


    def extract_shape(self, mod_idx, fidx, pidx):
        """
        for SINGLE param
        pidx, fidx = dynamic
        todo merge and improve
        """
        jax.debug.print("extract_shape... ")
        # sum amount equations till current + all their fields
        # get amount fields
        amount_fields = jnp.sum(self.FIELDS[:mod_idx])

        # get abs fiedld idx
        amount_fields += fidx

        # get relative amount params till field
        amount_params_pre_fidx = jnp.sum(self.AMOUNT_PARAMS_PER_FIELD[:amount_fields])

        # calc relative field idx to abs field idx sum
        abs_param_idx = amount_params_pre_fidx + pidx

        shape = self.SHAPE[abs_param_idx]

        jax.debug.print("shape set...")
        return shape


    def get_db_index(self, mod_idx, field_idx, param_in_field_idx):
        """
        convert relative indexing to scaled (with extend with focus on simplicity)
        """

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


    def extract_field_param_variation(
            self,
            variation_param_list
    ):
        """
        Extrahiert Parameter-Blöcke aus der DB und summiert Variationen auf.

        db_indices: list[tuple] - [Variation_Index][Parameter_Index]
        todo unter 3d tuple speichern
        """
        jax.debug.print("Summating Field Parameters...")

        idx_matrix = jnp.array(variation_param_list)
        extracted_params = jax.vmap(
            self.get_parameter_vector,
            in_axes=0
        )(idx_matrix)

        jax.debug.print("extract_field_param_variation...")
        return extracted_params

    def get_parameter_vector(self, item) -> jnp.array:
        # Extract single parameter flatten grid from db
        mod_idx, fidx, pidx = item

        # extract scaled idx. from db
        _start, _len = self.get_db_index(mod_idx, fidx, pidx)

        # receive flatten entris
        flatten_grid = jax.lax.dynamic_slice_in_dim(
            self.db_layer.old_g,
            _start,
            _len,
        )

        # get shape
        shape = self.extract_shape(
            mod_idx, fidx, pidx
        )

        # reshape received entries
        reshaped_grid = bring_flat_to_shape(
            flatten_grid,
            shape
        )
        jax.debug.print("get_parameter_vector ... dones")
        return reshaped_grid


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

"""


