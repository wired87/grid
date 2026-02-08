import jax.numpy as jnp
from jax import ops

def split_array(
        _batch,
        pieces:int,
        size:jnp.array
):
    group_ids = jnp.repeat(
        jnp.arange(pieces),
        size,
        axis=0,
    )
    group_sums = ops.segment_sum(_batch, group_ids)
    return group_sums
