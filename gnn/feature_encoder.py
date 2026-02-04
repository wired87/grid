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
        self.in_ts = []
        self.out_ts = jnp.array([])


    def gen_feature_single_variation(
        self,
        param_grid,
    ):
        #print("param_grid", param_grid)
        try:
            # receive grid all variations
            #return linears for each param

            def _work_param(param):
                # flat _param
                flat = jnp.ravel(param)

                # embed
                return jax.nn.gelu(
                    nnx.Linear(
                        in_features=len(flat),
                        out_features=self.d_model,
                        rngs=self.rngs
                    )(flat)
                )

            kernel = vmap(
                _work_param,
                in_axes=0
            )

            feature_block_single_param_grid = kernel(
                param_grid,
            )

            # working single parameter
            return jnp.array(feature_block_single_param_grid)
        except Exception as e:
            print("Err gen_feature_single_variation", e)


    def __call__(
            self,
            inputs: jnp.array,
    ):
        # todo include check fr prev time vals to
        print("FeatureEncoder.__call__...")
        try:
            self.idx_grid = jnp.arange(len(inputs))

            #
            results = jax.tree_util.tree_map(
                self.gen_feature_single_variation,
                inputs,
            )
            print("linears created", [f.shape for f in results])

            return results
        except Exception as e:
            print("Err FeatureEncoder.__call__:", e)
        print("FeatureEncoder.__call__... done")


    def stack_in_features(self, feature_res):
        # save in features
        vmap(
            self.stack_new_time_steps,
            in_axes=(0, 0)
        )(
            feature_res,
            self.idx_grid
        )


    def stack_new_time_steps(self, grid_param_item, index_map):
        # param dim for all time steps
        jnp.stack([ # todo extend
            self.in_ts[index_map],
            jnp.array(grid_param_item)
        ])

    def check_feature(self):
        pass



    def out_processor(
            self,
            output
    ):
        # todo include check fr prev time vals to
        print("FeatureEncoder.__call__...")
        try:
            # calc features
            results = self.gen_feature_single_variation(
                output
            )

            # save model tstep
            jnp.stack([
                self.out_ts,
                jnp.array(results)
            ])
        except Exception as e:
            print("Err FeatureEncoder.__call__:", e)
        print("FeatureEncoder.__call__... done")



"""

def gen_feature_batch(
    self,
    eq_in_out_grids,
    #shapes
):
    print("gen_feature_batch...")
    try:
        flatten_grid = []
        # receive all ins and all outs for eq
        for item in eq_in_out_grids:
            result = self.gen_feature_single_variation(item)
            flatten_grid.append(result)
        return flatten_grid
    except Exception as e:
        print("Err gen_feature_batch", e)
    print("gen_feature_batch... done")

"""
