"""
bias kb
Ich habe das Bias-Element in das Modul eingefügt, da es ein Goldstandard in den meisten modernen ML-Modellen ist und notwendig ist, um die Expressivität deiner Gleichung zu erhöhen.📝 Was ist das Bias?BegriffErklärung für Nicht-ML-SpezialistenWarum du es brauchstBias ($\mathbf{b}$)Stell dir das Bias als den Startwert oder den Grund-Offset einer Gleichung vor. Es ist ein einzelner, lernbarer Wert, der zur gewichteten Summe der Eingangsdaten addiert wird (z.B. $y = Wx + \mathbf{b}$).Es ermöglicht der Gleichung, ein Ergebnis von Null zu verschieben. Ohne Bias müsste das gesamte Netzwerk durch die Gewichte einen perfekten Nullpunkt treffen, was oft unmöglich ist, wenn die Daten selbst nicht zentriert sind.Der Wert [0.0]Dies ist der Startwert (Initialisierung). Er sagt: "Starte mit null Offset." Das Training wird diesen Wert später anpassen.Wenn du mit [0.0] startest, lässt du das Netzwerk von einem neutralen Punkt aus lernen.🎯 Kannst du in jedem Modul den gleichen Startwert verwenden?Ja, als Startwert (Initialisierung) kannst du in jedem Modul den gleichen Wert ([0.0]) verwenden. Dies ist ein häufiges, neutrales Vorgehen.Wichtig: Der gelernte Wert des Bias wird in jedem Modul unterschiedlich sein, da jedes Modul eine andere Gleichung auf verschiedenen Graphen-Regionen löst.📐 Wie wählst du die Shape?Die Shape (Form/Dimension) des Bias muss mit der Shape des Outputs deines Moduls übereinstimmen.Dein Fall: Da dein Modul eine einzelne Gleichung darstellt und vermutlich einen einzelnen Wert (oder einen Vektor von Werten) an einer bestimmten Stelle im neuen Graphen (new_g.nodes) berechnet, ist (1,) oder eine ähnliche Form (wie jnp.array([0.0])) oft korrekt.Regel: Wenn dein Modul 5 Features ausgibt, muss das Bias-Array ebenfalls 5 Elemente haben, damit es elementweise addiert werden kann.
"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from flax import nnx
from jax import vmap

from gnn.feature_encoder import FeatureEncoder
from utils import SHIFT_DIRS, DIM

def test(*args):
    return True

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
    todo check similar pre computed node
    """

    # Example parameter (weight) to be learned
    # Use nnx.Param for weights
    def __init__(self):
        self.feature_encoder = FeatureEncoder()
        self.embedding_dim = 64
        self.runnable = test
        self.param_blur = .99
        self.result_blur = .01

    def __call__(
            self,
            grid,
            in_axes_def
    ):
        # patterns here is the nested tuple list for one pattern execution

        # gen in featrues
        in_features = self.feature_encoder(
            inputs=grid
        )

        # receive list with None vales for
        # non pre-computed
        precomputed_results = self.get_precomputed_results(
            in_features,
            in_axes_def
        )

        outs = self.process_equation(
            grid,
            precomputed_results,
            in_axes_def,
        )



        # save in features
        self.feature_encoder.stack_in_features(
            feature_res=in_features
        )

        # create featuure time step
        feature_tstep = self.feature_encoder(
            inputs=grid,
            outputs=outs,
        )

        return outs, feature_tstep


    def get_precomputed_results(
            self,
            in_features: list,
            axis_def,
    ):
        print("get_precomputed_results...")
        out_feature_map = []
        if in_features is None:
            return out_feature_map
        try:
            feature_rows = self.convert_feature_to_rows(axis_def, in_features)
            print("get_precomputed_results feature_rows", [f.shape for f in feature_rows])

            # todo check from 2
            if len(self.feature_encoder.in_ts):
                # past in features
                def _make(*grids):
                    return self.convert_feature_to_rows(
                        axis_def, grids
                    )

                past_feature_rows = vmap(
                    _make,
                    in_axes=(0,*axis_def)
                )(
                    *self.feature_encoder.in_ts
                )
                print("past_feature_rows created")
                jnp.stack(
                    past_feature_rows,
                    axis=0,
                )

                out_feature_map = vmap(
                    self.fill_blur_vals,
                    in_axes=(0, None)
                )(
                    embedding_current=feature_rows,
                    prev_params=past_feature_rows
                )
            else:
                out_feature_map = [None for _ in range(len(feature_rows))]

        except Exception as e:
            print("Err get_precomputed_results", e)
            out_feature_map = []
        print("check... done")
        return out_feature_map




    def convert_feature_to_rows(self, axis_def, in_features):
        """

        Etract flattened
        :param axis_def:
        :param in_features:
        :return:
        """
        print("convert_feature_to_rows...")
        if in_features is None:
            return []

        def _process(*item):
            return jnp.concatenate([jnp.ravel(
                i,
                ) for i in item],axis=0)

        kernel = jax.vmap(
            fun=_process,
            in_axes=axis_def,
        )

        feature_rows = kernel(
            *in_features
        )

        print("feature_rows shape", [f.shape for f in feature_rows])
        return feature_rows



    def features(self, in_axes_def, inputs, input_shapes):
        kernel = jax.vmap(
            fun=self.feature_encoder,
            in_axes=(*in_axes_def, 0, None)
        )
        features = kernel(
            inputs
        )
        return features



    def fill_blur_vals(self, embedding_current, prev_params):
        print("fill_blur_vals...")
        embeddings = jnp.stack(prev_params, axis=0)      # (N, d_model)

        # L2-Distanz zu allen
        losses = jnp.linalg.norm(
            embeddings -
            embedding_current[None, :],
            axis=1
        )

        # min entry returns idx (everythign is order based (fixed f-len)
        idx = jnp.argmin(losses)
        min_loss = losses[idx]

        # -> PRE CALCULATED RESULT BASED ON BLUR
        if min_loss <= self.result_blur:
            return jnp.take(
                self.feature_encoder.out_ts,
                idx,
            )
        else:
            # -> MUST BE CALCED
            return None


    def process_equation(
            self,
            inputs,
            blur_results,
            in_axes_def,
    ):
        print("process_equation...")

        # DEBUG: (1) cond originally returned self.runnable (a function) -> "not a valid JAX type".
        # (2) Branches must have same pytree/type: true_fun was leaf, false_fun list -> match structure.
        # (3) Same dtype: true_fun bool[], false_fun float32[] -> cast both to float32.
        # (4) Same shape: true_fun float32[1], false_fun float32[0] -> compute both, pad bres to
        #     run_out.shape so cond accepts. (5) bres can contain None -> skip in _use_bres.
        def _run(_x):
            out = self.runnable(*_x)
            if isinstance(out, (list, tuple)):
                parts = [jnp.ravel(jnp.asarray(a, dtype=jnp.float32)) for a in out]
                return jnp.concatenate(parts) if parts else jnp.array([], dtype=jnp.float32)
            return jnp.asarray(out, dtype=jnp.float32).ravel()

        def _use_bres(bres_val, _x):
            if not isinstance(bres_val, (list, tuple)):
                return jnp.asarray(bres_val, dtype=jnp.float32).ravel()
            parts = [jnp.ravel(jnp.asarray(a, dtype=jnp.float32)) for a in bres_val if a is not None]
            return jnp.concatenate(parts) if parts else jnp.array([], dtype=jnp.float32)

        def _calc(bres, *item):
            run_out = _run(item)
            bres_out = _use_bres(bres, item)
            bres_padded = jnp.resize(bres_out, run_out.shape)
            return jax.lax.cond(
                bres is None,
                lambda: run_out,
                lambda: bres_padded,
            )

        kernel = jax.vmap(
            fun=_calc,
            in_axes=(0, *in_axes_def)
        )

        result = kernel(
            blur_results,
            *inputs
        )
        print("process_equation... done")
        return result, inputs



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
    def stack(self, axis_def, in_features):
        print("stack...")

        def _get_row(*items):
            return [*items]

        # Vmap über die Features
        in_features = vmap(
            _get_row,
            in_axes=axis_def,
        )(*in_features)

        processed_features = [jnp.squeeze(jnp.array(f)) for f in in_features]

        print("in_features (processed)", [f.shape for f in processed_features])

        # Jetzt sind alle (36, 64) und können gestapelt werden
        feature_rows = jnp.stack(
            processed_features,
            axis=1,
        )

        print("feature_rows shape", feature_rows.shape)
        return feature_rows

"""