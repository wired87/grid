"""
bias kb
Ich habe das Bias-Element in das Modul eingefügt, da es ein Goldstandard in den meisten modernen ML-Modellen ist und notwendig ist, um die Expressivität deiner Gleichung zu erhöhen.📝 Was ist das Bias?BegriffErklärung für Nicht-ML-SpezialistenWarum du es brauchstBias ($\mathbf{b}$)Stell dir das Bias als den Startwert oder den Grund-Offset einer Gleichung vor. Es ist ein einzelner, lernbarer Wert, der zur gewichteten Summe der Eingangsdaten addiert wird (z.B. $y = Wx + \mathbf{b}$).Es ermöglicht der Gleichung, ein Ergebnis von Null zu verschieben. Ohne Bias müsste das gesamte Netzwerk durch die Gewichte einen perfekten Nullpunkt treffen, was oft unmöglich ist, wenn die Daten selbst nicht zentriert sind.Der Wert [0.0]Dies ist der Startwert (Initialisierung). Er sagt: "Starte mit null Offset." Das Training wird diesen Wert später anpassen.Wenn du mit [0.0] startest, lässt du das Netzwerk von einem neutralen Punkt aus lernen.🎯 Kannst du in jedem Modul den gleichen Startwert verwenden?Ja, als Startwert (Initialisierung) kannst du in jedem Modul den gleichen Wert ([0.0]) verwenden. Dies ist ein häufiges, neutrales Vorgehen.Wichtig: Der gelernte Wert des Bias wird in jedem Modul unterschiedlich sein, da jedes Modul eine andere Gleichung auf verschiedenen Graphen-Regionen löst.📐 Wie wählst du die Shape?Die Shape (Form/Dimension) des Bias muss mit der Shape des Outputs deines Moduls übereinstimmen.Dein Fall: Da dein Modul eine einzelne Gleichung darstellt und vermutlich einen einzelnen Wert (oder einen Vektor von Werten) an einer bestimmten Stelle im neuen Graphen (new_g.nodes) berechnet, ist (1,) oder eine ähnliche Form (wie jnp.array([0.0])) oft korrekt.Regel: Wenn dein Modul 5 Features ausgibt, muss das Bias-Array ebenfalls 5 Elemente haben, damit es elementweise addiert werden kann.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import vmap
from typing import Callable, List, Tuple

from gnn.gnn import Graph
from utils import SHIFT_DIRS, DIM


class ModuleUtils:

    def __init__(self, amount_nodes):
        self.amount_nodes=amount_nodes
        self.ref_tower:list[tuple] = [
            (i for _ in range(DIM))
            for i in range(len(self.amount_nodes))
        ]


    def get_item_index(self, pos) -> int:
        return self.ref_tower.index(pos)


    def set_next_neighbors(
        self,
        pos_map:list[int]
    ) -> list[int]:
        """
        Use jsut to set index pos fr next iter
        """
        nmap=[]
        for posid in pos_map:
            for s in SHIFT_DIRS:
                pos:tuple = self.ref_tower[posid]
                neighbor_pos = pos+s
                nmap.append(self.ref_tower.index(neighbor_pos))
        print("ITEM POS SET")
        return nmap


    def extract_index_from_center(self, index:list[tuple or int]):
        # append center pos for self interaction on single point
        if isinstance(index[0], tuple):
            index = [self.ref_tower.index(npos) for npos in index]

        index_map = set(*index)
        pos_list = self.set_next_neighbors(index)

        for pos in pos_list:
            index_map.add(self.get_item_index(pos))

        # create
        return index_map

    def crate_key_map(self, field_type, attr_item) -> list[str]:
        attr_keys = list(attr_item.keys())
        if field_type in G_FIELDS:
            # Just here attrs not
            attr_keys.extend(self.gauge_field_keys)
        return attr_keys


class Node(nnx.Module):

    """
    DEF
    NNX Module representing a single GNN equation with learnable weights.
    """

    # Example parameter (weight) to be learned
    # Use nnx.Param for weights
    def __init__(
            self,
            runnable: Callable,
            inp_patterns: List[Tuple],  # method_list[param_map[]]
            outp_pattern: Tuple, #
            in_axes_def: Tuple, #
            method_id:str, #
            rngs: nnx.Rngs,
    ):
        self.weight = nnx.Param(
            jax.random.normal(
                rngs.params(),
                (1,)
            )
        )
        self.bias = nnx.Param(jnp.array([0.0]))  # Simple bias for the equation

        # Static pattern definitions - these should NOT be JAX arrays to allow tuple indexing
        self.runnable = jax.tree_util.Partial(runnable)
        self.inp_patterns = inp_patterns # Keep as list of tuples
        self.outp_pattern = outp_pattern
        self.in_axes_def = in_axes_def
        self.method_id = method_id

    def __call__(
            self,
            old_g: Graph,  # Use jnp.ndarray for GPU efficiency
            new_g: Graph,
            time_map
    ) -> jnp.ndarray:
        #
        self.old_g=old_g
        self.new_g=new_g

        # Iterate over static patterns for this module
        # Note: vmapped_kernel iterates over self.inp_patterns
        # Since self.inp_patterns is a list, we might need a standard loop 
        # or list comprehension here to keep it static.
        results = []
        for p_idx, patterns in enumerate(self.inp_patterns):
             # patterns here is the nested tuple list for one pattern execution
             res = self.proces_collection(patterns, time_map)
             results.append(res)
             
        # Stack results or merge them into new_g
        # For this demo, we assume the kernel writes to new_g, 
        # but pure functions should return the updated bits.
        return self.new_g


    def proces_collection(
            self,
            patterns,
            time_map,
    ):
        # Debug: check what we're working with
        inputs = self.get_inputs(patterns, time_map)
        jax.debug.print(
            "proces_collection - method_id: {mid}, patterns: {p}, in_axes_def: {axes}, num_inputs: {n}",
            mid=self.method_id,
            p=patterns,
            axes=self.in_axes_def,
            n=len(inputs)
        )
        
        vmapped_kernel = vmap(
            self.runnable,
            in_axes=self.in_axes_def,
        )

        # each inp  ut val represents a stack
        # collection of all
        field_based_calc_result = vmapped_kernel(*inputs)
        # media merge modular
        # Return the result instead of side-effect
        return field_based_calc_result

    def get_inputs(self, patterns, emap):
        # Receive alledge mapping for all params        param_emap = []
        method_params = []
        for p in patterns:
            # Extract the slice for this specific pattern p=(m, f, p)
            # p should be a tuple of indices
            # If p=(m, f, p), param_grid is the [Pos, Feat] array.
            
            # DEBUG: Handle both int and iterable cases
            if isinstance(p, int):
                # If p is a single integer, wrap it in a tuple
                param_grid_map = (p,)
            else:
                # If p is already iterable (list/tuple), convert to tuple
                param_grid_map = tuple(p)
            
            param_grid = self.old_g.nodes[param_grid_map]
            
            # emap.nodes[tuple(p)] is the [Pos] mask for this field.
            #mask = emap.nodes[param_grid_map]
            
            # Slice it: emap_item is [ActivePos, Feat]
            #emap_item = param_grid[mask]
            #param_emap.append(emap_item)
            method_params.append(param_grid)
        return method_params



    def mark_dmu_surrounding_acitve_neighbor(
            self,
            mu,
            active,
            dx
    ):
        # todo
        mu_eff = mu * active
        lap = (
            jnp.roll(mu_eff, 1, 0) + jnp.roll(mu_eff, -1, 0) +
            jnp.roll(mu_eff, 1, 1) + jnp.roll(mu_eff, -1, 1) +
            jnp.roll(mu_eff, 1, 2) + jnp.roll(mu_eff, -1, 2) -
            6 * mu_eff
        ) / dx ** 2
        return lap * active