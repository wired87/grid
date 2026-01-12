"""

Update rocess
inject nodes
identify nodes e > 0
create G copy
"""
from typing import Callable, Any, Sequence, List, NamedTuple
from flax import nnx

import jax
import jax.numpy as jnp
from jax import jit

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
            nodes,
            inj_pattern:list[int],
            updator_pattern,
            modules_len,
            glob_time:int,
            chains,
            energy_map  # Added energy_map parameter
    ):
        self.global_utils = None

        self.index = 0
        self.glob_time = glob_time

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
        self.energy_map = energy_map  # Store energy_map


        self.graph = Graph(
            nodes=jnp.stack(nodes) if isinstance(nodes, list) else nodes,
            edges=updator_pattern,
        )

        self.chains = chains

        print("Node initialized and build successfully")


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


    def main(self):
        # create old_G
        self.old_g = Graph(
            self.graph.nodes,
            self.graph.edges,
        )

        self.simulate()
        self.save_model("model.safetensors")


    def simulate(self, steps: int = None):
        try:
            if steps is None:
                steps = self.glob_time

            # Initialize connectivity
            self.set_shift()

            for step in range(steps):
                jax.debug.print(
                    "Sim step {s}/{n}",
                    s=step+1,
                    n=steps
                )

                old_g = self.graph.copy()
                new_g = self.graph.copy()

                # apply energy
                updated_nodes = self.apply_inj(
                    step=step,
                    nodes=old_g.nodes,
                )

                old_g = Graph(nodes=updated_nodes, edges=old_g.edges)

                # FILTERED G
                time_map: TimeMap = self.extract_active_nodes(
                    graph=old_g,
                )

                # inclid ein time_map in sequence chains item
                # RUN ITER
                # In a simplified version, we assume run_all_chains updates the new_g
                new_g = run_all_chains(
                    self.chains,
                    old_g_batch=old_g,
                    new_g_batch=new_g,
                    active_pattern_map=time_map
                )


                # AFTER RUN:
                # set shift
                # write results to Graph -> switch
                self.set_next_nodes(time_map.nodes)


        except Exception as e:
            jax.debug.print(f"Err simulate: {e}")
            raise

        # FILTERED G
        time_map:TimeMap = self.extract_active_nodes(
            graph=old_g,
        )

        # AFTER RUN:
        # set shift
        # write results to Graph -> switch
        self.set_next_nodes(time_map.nodes)

        # todo collect weights, bias,... 4modules
        # todo upsert data copy
        # todo transform shift

        # rotate graphs: new becomes old for next iter
        old_g = new_g.copy()
        new_g = Graph(
            nodes=jnp.zeros_like(old_g.nodes),
            edges=old_g.edges, # assume edges are static or handled similarly
        )

    # ich muss interessierter sein

    def set_next_nodes(self, pos_module_field_node):
        for module in pos_module_field_node:
            for field_pos_list in module:
                field_pos_list = self.set_shift(field_pos_list)
        return pos_module_field_node


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
                ]  # assume energy_grid is jnp array

                # find indices where energy != 0
                #nonzero_indices = jnp.argwhere(energy_grid != 0.0)
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


    def apply_inj(self, step, nodes):
        """
        Applies injections based on self.inj_pattern and current step.
        Supports the SOA structure: [module, field, param, node_index, schedule]
        where schedule is list of [time, value].
        """
        all_indices = []
        all_values = []

        # Check if inj_pattern uses the SOA structure (flat list)
        # We assume if it's a list and the first item is a list of length 5 (or similar check)
        # But specifically checking for the DEMO_INPUT style
        
        for item in self.inj_pattern:
            if isinstance(item, list) and len(item) == 5 and isinstance(item[4], list):
                # SOA structure: [mod, field, param, node_idx, schedule]
                mod_idx, field_idx, param_idx, node_idx, schedule = item
                
                # Check schedule for current time
                for time_point, value in schedule:
                    if time_point == step:
                        # Add to update list
                        # Assuming feature index 0
                        all_indices.append([mod_idx, field_idx, param_idx, node_idx, 0])
                        all_values.append(value)
            
            # Fallback or other structure handling can be added here if needed
            # For now, we prioritize the SOA structure as requested

        if not all_indices:
            return nodes

        all_indices = jnp.array(all_indices)  # shape [N, 5]
        all_values = jnp.array(all_values)    # shape [N]

        # Apply using .at[].add() for "injerease" (increase) or .set() if strictly setting
        # User said "value injerease" -> increase. 
        # But also "right value to set". 
        # Injections usually add. We will use add.
        nodes = nodes.at[tuple(all_indices.T)].add(all_values)
        return nodes

    def save_model(self, filename="model.safetensors"):
        """Saves the learned parameters of all chains to a .safetensors file."""
        if safetensors_save is None:
            jax.debug.print("safetensors not installed. Skipping save.")
            return

        all_tensors = {}
        for i, chain in enumerate(self.chains):
            # Extract parameters
            state = chain.extract(nnx.Param)
            # Convert state into a flat dict of arrays
            flat_state, _ = jax.tree_util.tree_flatten(state)
            
            # Add to all_tensors with prefix
            for j, p in enumerate(flat_state):
                all_tensors[f"chain_{i}_param_{j}"] = jnp.asarray(p)

        try:
            safetensors_save(all_tensors, filename)
            jax.debug.print("Model parameters successfully saved to {f}.", f=filename)
        except Exception as e:
            jax.debug.print("Failed to save safetensors: {e}", e=str(e))

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





def run_all_chains(chains, old_g_batch, new_g_batch, active_pattern_map):
    result_stack = []
    for chain in chains:
        result = chain(
            old_g=old_g_batch,
            new_g=new_g_batch,
            time_map=active_pattern_map,
        )
        result_stack.append(result)
    jax.debug.print("chain kernel initialized successfully")
    return result_stack

















































