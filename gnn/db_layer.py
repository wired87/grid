import json
import os

import jax
import jax.numpy as jnp
from flax import linen as nn

from dtypes import TimeMap


class DBLayer(nn.Module):


    def __init__(self, amount_nodes):
        # DB Load and parse
        super().__init__(self)

        db_str = os.getenv("DB")
        self.db_pattern = json.loads(db_str)

        self.build_db(amount_nodes, self.db_pattern)
        self.nodes = None
        self.old_g = self.nodes
        self.new_g = self.nodes

        self.model_history = jnp.array([])



    def process_time_step(self):
        # todo FILTERED G (Active nodes extraction)
        """time_map = self.extract_active_nodes(
            graph=old_g,
        )"""

        # todo
        self.old_g = self.nodes
        return



    def build_db(self, amount_nodes, db):
        # create world
        jax.debug.print("build_db...")

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

        self.nodes = jax.device_put(nodes, self.gpu)
        jax.debug.print("build_db... done")


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


