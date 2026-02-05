import jax
from jax import vmap
from flax import nnx
import jax.numpy as jnp


class FeatureEncoder(nnx.Module):
    def __init__(
            self,
            amount_variations:int,
            d_model: int = 64,
    ):
        # Wir speichern für jede Eingabe eine Projektion auf 64 Dimensionen
        self.rngs = nnx.Rngs(42)
        self.d_model = d_model
        self.in_ts = []
        self.amount_variations = amount_variations
        self.out_ts = jnp.array([])

        #
        self.skeleton = jnp.array([])
        self.store = []

    def gen_feature_single_variation(
        self,
        param_grid,
        ax,
        skeleton_idx:int,
    ):
        print("gen_feature_single_variation...")

        def _work_param(linear_instance, param):
            # flat _param
            flat = jnp.ravel(param)

            # embed
            return jax.nn.gelu(
                linear_instance(flat)
            )

        try:
            kernel = vmap(
                _work_param,
                in_axes=(ax)
            )

            feature_block_single_param_grid = kernel(
                param_grid,
                self.skeleton[skeleton_idx]
            )

            # working single parameter
            features = jnp.array(feature_block_single_param_grid)

            # save features
            self._save(features)

            # finished
            return features
        except Exception as e:
            print("Err gen_feature_single_variation", e)
        print("gen_feature_single_variation... done")







    def create_features(
            self, inputs, axis_def, time, param_idx=0
    ):
        try:
            jax.lax.cond(
                time == 0,
                lambda: self.build_model_skeleton(
                    param_idx=param_idx
                ),
                lambda: print("pass"),
            )

            results = jax.tree_util.tree_map(
                self.gen_feature_single_variation,
                inputs,axis_def
            )




            return results
        except Exception as e:
            print("Err create_in_features:", e)



    def _save(self, features:jnp.array):
        pass






    def build_model_skeleton(self, param_idx):
        print("build_model_skeleton...")

        def build_in_skeleton(item):
            print("define model in skeleton")
            self.skeleton = [
                [
                    nnx.Linear(
                        in_features=len(_item),
                        out_features=self.d_model,
                        rngs=self.rngs
                    )
                    for _ in range(self.amount_variations)
                ]
                for _item in item
            ]
            self.store=[[] for _ in range(len(item))]

        def build_out_skeleton(item):
            print("define model in skeleton")
            out_skeleton = [
                nnx.Linear(
                    in_features=len(item),
                    out_features=self.d_model,
                    rngs=self.rngs
                )
                for _ in range(self.amount_variations)
            ]
            # add last element
            self.skeleton.at[-1].add(out_skeleton)
            self.store.append([])  # to store feeatures in

        jax.lax.cond(
            param_idx != -1,
            lambda: build_in_skeleton,
            lambda: build_out_skeleton,
        )
        print("build_model_skeleton... done")






    def out_processor(
            self,
            output
    ):
        # todo include check fr prev time vals to
        print("FeatureEncoder.out_processor...")
        try:

            # check build out skeleton
            jax.lax.cond(
                len(self.out_skeleton) == 0,
                lambda: self.build_out_skeleton(
                    item=jax.tree_util.tree_map(
                        lambda x: x[0], output
                    )
                ),
                lambda: print("pass"),
            )

            # calc features
            results = self.gen_feature_single_variation(
                output,
                ax=0,
            )

            # save model tstep
            jnp.stack([
                self.out_ts,
                jnp.array(results)
            ])
        except Exception as e:
            print("Err FeatureEncoder.out_processor:", e)
        print("FeatureEncoder.out_processor... done")


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
