import jax
import jax.numpy as jnp
from flax import nnx
from jax import vmap



class Iterator:
    """


    """

    def __init__(self):
        pass




    def pattern_recall(
        self,
        param_grid,
        time_map:jnp.array,
    ):
        """
        Use reconstruct anything based on single parameter
        """
        #print("param_grid", param_grid)
        try:
            # receive grid all variations
            #return linears for each param
            def _extract_most_similar_pts(
                    param_current,
                    *prev_params
            ):
                # most similar past-time-step (pts)
                flat = jnp.ravel(param_current)

                # embed
                embedding_current = jax.nn.gelu(
                    nnx.Linear(
                        in_features=len(flat),
                        out_features=self.d_model,
                        rngs=self.rngs
                    )(flat)
                )

                closest_idx = self.closest_embedding(
                    embedding_current, prev_params
                )
                # return time step idx with most matching param
                return closest_idx

            def _extract_value_indices(indice):
                return jnp.take(time_map, indice)


            indices = vmap(
                _work_param,
                in_axes=(
                    0,
                    0
                    for _ in range(len(time_map))
                )
            )(
                param_grid,
                *time_map
            )


            # working single parameter
            return jnp.array(feature_block_single_param_grid)
        except Exception as e:
            print("Err gen_feature_single_variation", e)

