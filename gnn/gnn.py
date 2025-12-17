"""

Update rocess
inject nodes
identify nodes e > 0
create G copy
"""


from dataclasses import dataclass
from typing import Callable, Any, Sequence, List

from flax import nnx
from jax import jit
from jax.tree_util import tree_map, register_pytree_node_class
import os

import jax
import jax.numpy as jnp

from gnn.chain import GNNChain, simulation_step
from main import SHIFT_DIRS
from mod import Node

try:
    from safetensors.numpy import save_file as safetensors_save
except Exception:
    safetensors_save = None


@register_pytree_node_class
@dataclass
class Graph:
    nodes: jnp.ndarray
    edges: jnp.ndarray

    def copy(self) -> "Graph":
        # shallow copy arrays (safe for immutable jax arrays)
        # nodes is array, so just pass it (immutable)
        return Graph(self.nodes, self.edges)

    def xtract_from_indices(self, mindex, findex, map):
        # use for isolated time mapping
        return self.nodes[mindex][findex][map]

    def __getitem__(self, key):
        if isinstance(key, tuple):
             return self.nodes[key[0]]
        return self.nodes[key]

    def tree_flatten(self):
        return ((self.nodes,), self.edges)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], aux_data)

@dataclass
class TimeMap:
    nodes: jnp.ndarray


@register_pytree_node_class
@dataclass
class Payload:
    grid_nodes: jnp.ndarray
    old: jnp.ndarray
    new: jnp.ndarray
    edges: jnp.ndarray

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
            glob_time:int
    ):
        self.global_utils = None

        self.index = 0
        self.glob_time = glob_time

        self.world = []
        self.amount_nodes = amount_nodes
        self.store = []
        self.module_map = []

        self.schema_grid = [
            (i, i, i)
            for i in range(len(self.amount_nodes))
        ]

        self.method_params = {}
        self.change_store = []

        self.modules_len=modules_len
        self.inj_pattern = inj_pattern


        self.graph = Graph(
            nodes=jnp.stack(nodes) if isinstance(nodes, list) else nodes,
            edges=updator_pattern,
        )

        print("Node initialized and build successfully")


    def set_shift(self, start_pos:list[tuple]):
        # !mark: the pm filter procedure is performed
        # in method
        next_index_map = []
        for pos in start_pos:
            for dir in list(SHIFT_DIRS[0]+SHIFT_DIRS[1]):
                neighbor = dir+pos
                next_index_map.append(
                    self.schema_grid.index(neighbor)
                )
            next_index_map.append(
                self.schema_grid.index(pos)
            )
        return next_index_map


    def main(
            self,
            energy_map:list[list[int]],
    ):
        self.chains: Sequence[GNNChain] = self.create_modules(
            self.updator_pattern
        )

        self.energy_map=energy_map

        # create old_G
        self.old_g = Graph(
            self.graph.nodes,
            self.graph.edges,
        )
        self.simulate()


    def simulate(self):
        for step in range(self.glob_time):
            jax.debug.print(
                "Sim step {s}/{n}",
                s=step+1,
                n=self.glob_time
            )

            old_g = self.graph.copy()
            new_g = self.graph.copy()

            # apply energy
            old_g.nodes = self.apply_inj(
                inj_pattern_module_map=self.inj_pattern[0],
                nodes=old_g.nodes,
            )

            # FILTERED G
            time_map:TimeMap = self.extract_active_nodes(
                graph=old_g,
            )




            # inclid ein time_map in sequence chains item
            # RUN ITER
            new_g = run_all_chains(
                self.chains, #
                batched_old_g=old_g,
                batched_new_g=new_g,
                active_pattern_map=time_map
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
                [
                    jnp.array(jnp.zeros_like(n))
                    for n in old_g.nodes],
                [
                    jnp.array(jnp.zeros_like(e))
                    for e in old_g.edges],
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



    def apply_inj(self, inj_pattern_module_map, nodes):
        """
        inj_pattern_module_map : current cluster def for injections
        inj_pattern[current_time]:
            list[module] -> list[field] -> list[param] -> list[pos_struct]
            pos_struct = [pos, value]
        nodes: jax.numpy array, shape = [modules, fields, params, positions, features]
        """

        # Flatten all positions into single arrays for vectorized scatter
        all_indices = []
        all_values = []

        for i, module in enumerate(inj_pattern_module_map):
            for j, field in enumerate(module):
                for k, param in enumerate(field):
                    for o, pos_struct in enumerate(param):
                        # pos_struct: [pos, value]
                        idx = self.get_index(pos_struct[0])  # integer index
                        all_indices.append(
                            [i, j, k, o, idx]
                        )
                        all_values.append(pos_struct[1])

        all_indices = jnp.array(all_indices)  # shape [N,5]
        all_values = jnp.array(all_values)  # shape [N]

        # Vectorized update
        nodes = nodes.at[tuple(all_indices.T)].set(all_values)

        return nodes

    def get_sdr_rcvr(self):
        receiver = []
        sender = []
        for i, item in enumerate(range(len(self.store))):
            for j, eitem in enumerate(self.edges):
                sender.extend([i for _ in range(len(self.edges))])
                receiver.extend(eitem)
        jax.debug.print("set direct interactions finsihed")
        return sender, receiver


@jit
def run_all_chains(chains, old_g_batch, new_g_batch, active_pattern_map):

    def single_chain_call(chain, old_g_instance, new_g_instance, active_pattern_map):
        # Calc a pathway
        return chain(
            old_g=old_g_instance,
            new_g=new_g_instance,
            time_map=active_pattern_map,
        )

    # nnx.vmap splittet die Parameter der 'chains' und die Eingaben 'old_g_batch'
    vmap = nnx.vmap(
        single_chain_call,
        in_axes=(
            0,
            None,
            None,
            None, # overall same
        )
    )

    # KERNEL DOESNT RETURN -
    # MODULES WRITE UPDATES BY OWN 2 GLOB
    result_stack = vmap(
        chains, old_g_batch, new_g_batch, active_pattern_map
    )
    
    # perform transformation

    
    jax.debug.print("chain kernel initialized successfully")




    def set_nodes(self, mid, data:list[list or Any]):
        self.graph.nodes[mid] = data


    def set_edges(self, mid, index, pattern):
        self.graph.edges[mid][index] = pattern


    def set_field_store_order(self, field_pattern):
        self.store = field_pattern
        self.edges = field_pattern
        self.axis_def = field_pattern
        self.eqm_map = field_pattern

        # create modules zeit wecheslt die instanz nicht die materie (daten des env ändern sich)


    def receive_field(
            self,
            data,
            axis_def,
            mid,
            runnable_map: list[Callable],
            index: int,
            mindex,
            edges=None,
    ):

        """
        receive a specific field and pattern
        """

        try:
            self.store[index]: list[jnp.array] = data

            """
            self.edges[index]: list[list] = edges

            self.axis_def[index]: list[list] = axis_def
            """
            # just receive here nodes
            self.module_map[index] = Node(
                runnable_map=runnable_map,
                index=index,
                eq_module_index=mid,
                axis_def=axis_def,
                amount_nodes=self.amount_nodes,
                mid=mid,
            )

            Graph.nodes[mindex][index] = data
            Graph.edges[mindex][index] = edges

            return True
        except Exception as e:
            jax.debug.print("Err receive_field", e)
        return False


    def _save_model_safetensors(self, gnn_chain: GNNChain, output_dir="outputs"):
        """Saves the learned parameters to a .safetensors file."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Extract Parameters and State from the trained chain
        params = gnn_chain.extract('params')

        tensors_to_save = tree_map(lambda x: x.value, params)

        jax.debug.print("Model parameters successfully extracted and saved to SafeTensors.")
        pass  # Placeholder for actual saving implementation


    def run_simulation(self, steps: int, target_data: jnp.ndarray, modules_list: Sequence[Node]):
        # 1. Setup: Create the GNN Chain and Optimizer
        gnn_chain = GNNChain(modules_list)
        # Use an optimizer (example: Adam)
        optimizer = nnx.optim.Adam(gnn_chain.extract('params'))

        current_g = self.graph.nodes  # Get initial data (ensure it's a JAX array on GPU)

        # 2. Simulation Loop (JAX-compiled core)
        for step in range(steps):
            # The entire training step is compiled and runs on GPU
            predicted_g, gnn_chain, loss = simulation_step(
                gnn_chain, current_g, optimizer, target_data[step]
            )
            current_g = predicted_g  # New state becomes the old state for the next step

        # 3. Final Saving after simulation
        self._save_model_safetensors(gnn_chain)
        return "Simulation complete and model saved."












































