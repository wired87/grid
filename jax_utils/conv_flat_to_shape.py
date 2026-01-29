import jax.numpy as jnp


def bring_flat_to_shape(lst, shape):
    lst = jnp.array(lst)

    # Sicherstellen, dass shape mindestens [1] ist, falls [] übergeben wurde
    if len(shape) == 0:
        shape = (1,)

    shape_jnp = jnp.array(shape)
    block_size = jnp.prod(shape_jnp)

    # Vermeide Division durch Null, falls block_size 0 wäre
    n_blocks = len(lst) // block_size

    trimmed = lst[:n_blocks * block_size]

    # Reshape
    reshaped = trimmed.reshape((n_blocks, *shape))
    print("trgt shape", shape)
    return reshaped

