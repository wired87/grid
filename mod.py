"""
bias kb
Ich habe das Bias-Element in das Modul eingefügt, da es ein Goldstandard in den meisten modernen ML-Modellen ist und notwendig ist, um die Expressivität deiner Gleichung zu erhöhen.📝 Was ist das Bias?BegriffErklärung für Nicht-ML-SpezialistenWarum du es brauchstBias ($\mathbf{b}$)Stell dir das Bias als den Startwert oder den Grund-Offset einer Gleichung vor. Es ist ein einzelner, lernbarer Wert, der zur gewichteten Summe der Eingangsdaten addiert wird (z.B. $y = Wx + \mathbf{b}$).Es ermöglicht der Gleichung, ein Ergebnis von Null zu verschieben. Ohne Bias müsste das gesamte Netzwerk durch die Gewichte einen perfekten Nullpunkt treffen, was oft unmöglich ist, wenn die Daten selbst nicht zentriert sind.Der Wert [0.0]Dies ist der Startwert (Initialisierung). Er sagt: "Starte mit null Offset." Das Training wird diesen Wert später anpassen.Wenn du mit [0.0] startest, lässt du das Netzwerk von einem neutralen Punkt aus lernen.🎯 Kannst du in jedem Modul den gleichen Startwert verwenden?Ja, als Startwert (Initialisierung) kannst du in jedem Modul den gleichen Wert ([0.0]) verwenden. Dies ist ein häufiges, neutrales Vorgehen.Wichtig: Der gelernte Wert des Bias wird in jedem Modul unterschiedlich sein, da jedes Modul eine andere Gleichung auf verschiedenen Graphen-Regionen löst.📐 Wie wählst du die Shape?Die Shape (Form/Dimension) des Bias muss mit der Shape des Outputs deines Moduls übereinstimmen.Dein Fall: Da dein Modul eine einzelne Gleichung darstellt und vermutlich einen einzelnen Wert (oder einen Vektor von Werten) an einer bestimmten Stelle im neuen Graphen (new_g.nodes) berechnet, ist (1,) oder eine ähnliche Form (wie jnp.array([0.0])) oft korrekt.Regel: Wenn dein Modul 5 Features ausgibt, muss das Bias-Array ebenfalls 5 Elemente haben, damit es elementweise addiert werden kann.
"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from flax import nnx
from jax import vmap
from typing import Callable, List, Tuple

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

    def crate_key_map(self, field_type, attr_item, G_FIELDS=None) -> list[str]:
        attr_keys = list(attr_item.keys())
        if field_type in G_FIELDS:
            # Just here attrs not
            attr_keys.extend(self.gauge_field_keys)
        return attr_keys


class Node(nnx.Module):

    """
    DEF
    NNX Module representing a single GNN equation with
    learnable weights.

    calc -> generate gnn stuff -> paste to gnn_ds -> return None


    """

    # Example parameter (weight) to be learned
    # Use nnx.Param for weights
    def __init__(
            self,
            runnable: Callable,
            in_axes_def: Tuple, #
            method_id:str, #
            mod_idx:int
    ):
        self.embedding_dim = 64
        self.bias = nnx.Param(jnp.array([0.0]))  # Simple bias for the equation

        # Static pattern definitions - these should NOT be JAX arrays to allow tuple indexing
        self.runnable = jax.tree_util.Partial(runnable)
        self.in_axes_def = in_axes_def
        self.method_id = method_id
        self.mod_idx = mod_idx
        self.projector = nn.Dense(
            self.embedding_dim,
            name=str(self.method_id)
        )


    def __call__(
            self,
            old_g,
            field_variation_structs,
    ):
        #
        self.old_g = old_g
        features = []

        results = []
        #calculate new EQ step for all modules fields
        for p_idx, field_pattern in enumerate(field_variation_structs):
             # patterns here is the nested tuple list for one pattern execution
             outs, ins = self.process_equation(field_pattern, time_map)

             features_t = jnp.concatenate([ins, outs], axis=-1)
             features.append(features_t)

             # sum calc result for variations for single field eq run
             result = jnp.sum(outs)

             # apply field result -> db
             results.append(result)
        return features, results


    def upscale_feature_variations(self):
        # 1. Wir wandeln alles in eine Liste von JAX-Arrays um
        # Dabei stellen wir sicher, dass alles mindestens 1D ist
        arrays = [jnp.atleast_1d(jnp.array(x)) for x in field_eq_param_struct]

        # 2. Die Ziel-Länge n bestimmen (entspricht deinem 'len(long)')
        n = max(arr.shape[0] for arr in arrays)

        # 3. Hochskalieren und Stacken
        # jnp.broadcast_to simuliert die Vervielfältigung, ohne Speicher zu verschwenden
        res = jnp.stack([jnp.broadcast_to(arr, (n,)) for arr in arrays], axis=-1)
        return res


















    def transform_feature(
            self,
            inputs,
            outputs,
    ):
        jax.debug.print("transform_feature...")

        feature_matrix_eq_tstep = []
        # combine input output for each field -< todo improve with jax
        for field_in, field_out in zip(inputs, outputs):
            field_grid_features = []

            for point_in, point_out in zip(field_in, field_out):
                # Concatenate for persistency: [1, 2, 4, 9]
                combined = jnp.concatenate([point_in, point_out], axis=-1)

                # Project to GNN hidden state space
                field_grid_features = nn.Dense(self.embedding_dim)(combined)

            feature_matrix_eq_tstep.append(
                field_grid_features
            )
        jax.debug.print("transform_feature... done")
        return feature_matrix_eq_tstep





    def process_equation(
            self,
            patterns,
            time_map,
    ):
        # Debug: check what we're working with
        inputs = self.get_inputs(patterns, time_map)

        
        vmapped_kernel = vmap(
            self.runnable,
            in_axes=self.in_axes_def,
        )

        field_based_calc_result = vmapped_kernel(*inputs)

        return field_based_calc_result, inputs

    def get_inputs(self, patterns, emap):
        len_params_per_methods = []
        for p in patterns:
            # DEBUG: Handle both int and iterable cases
            if isinstance(p, int):
                param_grid_map = (p,)
            else:
                # If p is already iterable (list/tuple), convert to tuple
                param_grid_map = tuple(p)
            
            param_grid = self.old_g.nodes[param_grid_map]

            len_params_per_methods.append(param_grid)
        return len_params_per_methods



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





    def transform_feature(
            self,
            inputs,
            outputs,
    ):
        jax.debug.print("transform_feature...")

        feature_matrix_eq_tstep = []
        # combine input output for each field -< todo improve with jax
        for field_in, field_out in zip(inputs, outputs):
            field_grid_features = []

            for point_in, point_out in zip(field_in, field_out):
                # Concatenate for persistency: [1, 2, 4, 9]
                combined = jnp.concatenate([point_in, point_out], axis=-1)

                # Project to GNN hidden state space
                field_grid_features = nn.Dense(self.embedding_dim)(combined)

            feature_matrix_eq_tstep.append(
                field_grid_features
            )
        jax.debug.print("transform_feature... done")
        return feature_matrix_eq_tstep
"""


"""