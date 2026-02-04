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
        # DEBUG: vmap(_work_param) + nnx.Linear(..., rngs=self.rngs) caused "Cannot mutate RngCount
        # from a different trace level". Use Python loop over param_grid rows so Rngs are not
        # mutated inside a traced function; return jnp.array([]) on exception for tree_map safety.
        try:
            param_grid = jnp.asarray(param_grid)
            if param_grid.size == 0:
                return jnp.array([])
            out = []
            for i in range(param_grid.shape[0]):
                flat = jnp.ravel(param_grid[i])
                out.append(
                    jax.nn.gelu(
                        nnx.Linear(
                            in_features=len(flat),
                            out_features=self.d_model,
                            rngs=self.rngs,
                        )(flat)
                    )
                )
            return jnp.stack(out)
        except Exception as e:
            print("Err gen_feature_single_variation", e)
            return jnp.array([])


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
            # DEBUG: results can be nested tree; [f.shape for f in results] fails. Use tree_map for shapes.
            try:
                print("linears created", jax.tree_util.tree_map(lambda x: getattr(x, "shape", x), results))
            except Exception:
                pass

            return results
        except Exception as e:
            print("Err FeatureEncoder.__call__:", e)
        print("FeatureEncoder.__call__... done")


    def stack_in_features(self, feature_res):
        # save in features (loop over groups; vmap within group to avoid inconsistent sizes)
        flat_res = list(feature_res) if isinstance(feature_res, (list, tuple)) else [feature_res]
        while len(self.in_ts) < len(flat_res):
            self.in_ts.append(jnp.array([]))
        for i in range(len(flat_res)):
            arr = jnp.asarray(flat_res[i])
            if arr.size == 0:
                continue
            vmap(lambda row: self.stack_new_time_steps(row, i), in_axes=0)(arr)


    def stack_new_time_steps(self, grid_param_item, index_map):
        # param dim for all time steps. in_ts length ensured in stack_in_features; index_map may be
        # JAX scalar -> use int(index_map) / .item() for list indexing.
        idx = int(index_map) if hasattr(index_map, "item") else index_map
        jnp.stack([
            self.in_ts[idx],
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
