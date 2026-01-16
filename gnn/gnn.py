"""

Update rocess
inject nodes
identify nodes e > 0
create G copy
"""
import json
import os

from flax import nnx

import jax
import jax.numpy as jnp
from dtypes import Graph, TimeMap
from utils import SHIFT_DIRS

try:
    from safetensors.numpy import save_file as safetensors_save
except Exception:
    safetensors_save = None




class GNN(
    nnx.Module
):
    # todo pattern applieance and converter -< data ar request,
    # build stabel update loop

    """
    zeit

    Universal Actor for all fields to process a batch of Fields same type)
    With a GPU

    Design the system so each method has its own preprocessor (CPU)
    and processor (GPU)

    receive struct:
    [
    soa
    ],

    [
    [runnables]
    [(soa indices for params)]
    }

    WF:
    Hold all data on gpu but calc diferentyl
    pattern must come from cpu

    todo cache results to index
    """

    def __init__(
            self,
            amount_nodes,
            nodes_db,
            inj_pattern:list[int],
            modules_len,
            glob_time:int,
            chains,
            gnn_data_struct:list
    ):
        injection_pattern = os.getenv("INJECTION_PATTERN")
        self.injection_pattern = json.loads(injection_pattern)

        self.global_utils = None
        self.gnn_data_struct=gnn_data_struct
        self.index = 0

        self.glob_time = glob_time
        self.nodes_db = nodes_db

        self.world = []
        self.amount_nodes = amount_nodes
        self.store = []
        self.module_map = []

        # Generate grid coordinates based on amount_nodes dimensionality
        self.schema_grid = [
            (i,i,i)
            for i in range(amount_nodes)
        ]

        self.method_params = {}
        self.change_store = []

        self.modules_len=modules_len
        self.inj_pattern = inj_pattern

        self.chains = chains

        print("Node initialized and build successfully")

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

    def run_all_chains(self, old_g_batch, new_g_batch, active_pattern_map):
        result_stack = []
        for chain in self.chains:
            result = chain(
                old_g=old_g_batch,
                new_g=new_g_batch,
                time_map=active_pattern_map,
            )
            result_stack.append(result)
        jax.debug.print("chain kernel initialized successfully")
        return result_stack


    def main(self):
        # create old_G
        self.old_g = Graph(
            self.graph.nodes,
        )
        self.simulate()
        self.save_model("model.safetensors")




    def set_next_nodes(self, pos_module_field_node):
        for module in pos_module_field_node:
            for field_pos_list in module:
                field_pos_list = self.set_shift(field_pos_list)
        return pos_module_field_node


    def save_model(self, filename="model.safetensors"):
        """Saves the learned parameters of all chains to a .safetensors file."""
        if safetensors_save is None:
            jax.debug.print("safetensors not installed. Skipping save.")
            return

        all_tensors = {}
        for i, chain in enumerate(self.chains):
            # Extract parameters
            state = nnx.state(chain, nnx.Param)
            # Convert state into a flat dict of arrays
            flat_state, _ = jax.tree_util.tree_flatten(state)
            
            # Add to all_tensors with prefix
            for j, p in enumerate(flat_state):
                all_tensors[f"chain_{i}_param_{j}"] = jnp.asarray(p)

        try:
            safetensors_save(all_tensors, filename)
            print(f"Model parameters successfully saved to {filename}.")
        except Exception as e:
            print(f"Failed to save safetensors: {e}")

    def get_sdr_rcvr(self):
        receiver = []
        sender = []
        for i, item in enumerate(range(len(self.store))):
            for j, eitem in enumerate(self.edges):
                sender.extend([i for _ in range(len(self.edges))])
                receiver.extend(eitem)
        jax.debug.print("set direct interactions finsihed")
        return sender, receiver



"""
Extract pattern
# wie linken wir das grid an die richtige stelle?
provide altimes entire grid 
loop pattern 
"""























































