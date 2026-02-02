from typing import List

import jax
from jax import vmap
from flax import nnx
import jax.numpy as jnp


class FeatureEncoder(nnx.Module):
    def __init__(
            self,
            d_model: int = 64,
    ):
        """
        Erstellt für jede unterschiedliche Input-Shape einen eigenen Linear-Layer.
        """

        # Wir speichern für jede Eingabe eine Projektion auf 64 Dimensionen
        self.rngs = nnx.Rngs(42)
        self.d_model = d_model
        self.feature_time_store = []


    def gen_feature_single_variation(
        self,
        param_grid,
        shape,
    ):
        # receive grid all variations
        #return linears for each param
        def _work_param(param):
            jax.nn.gelu(
                nnx.Linear(
                    in_features=shape,
                    out_features=self.d_model,
                    rngs=self.rngs
                )(param)
            )

        kernel = vmap(
            _work_param,
            in_axes=(0, None)
        )

        feature_block_single_param_grid = kernel(
            param_grid,
            shape
        )

        # working single parameter
        return feature_block_single_param_grid


    def gen_feature_batch(
        self,
        in_eq,
        shapes
    ):
        # receive all ins and all outs for eq
        kernel = vmap(
            self.gen_feature_single_variation,
            in_axes=(0, 0)
        )

        feature_block = kernel(
            in_eq, # keep map vertical
            shapes
        )
        return feature_block

    def __call__(
            self,
            inputs: List[List[jax.Array]],
            outputs,
            shapes,
    ):
        # todo include check fr prev time vals to
        print("FeatureEncoder.__call__...")
        try:
            #
            results = jax.tree_util.tree_map(
                self.gen_feature_batch,
                [inputs, outputs],
                shapes,
            )

            # flatten
            self.model_skeleton = jnp.concatenate(
                *results
            )

            # save model tstep
            jnp.stack([
                self.feature_time_store,
                self.model_skeleton
            ])
        except Exception as e:
            print("Err FeatureEncoder.__call__:", e)
        print("FeatureEncoder.__call__... done")




