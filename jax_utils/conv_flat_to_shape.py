import jax.numpy as jnp

def bring_flat_to_shape(lst, shape):
    """
    Convert a flat list into consecutive blocks with given shape using jax.numpy.

    Parameters:
    - lst: list or 1D array of items
    - shape: tuple, the target shape of each block, e.g. (2,2)

    Returns:
    - jnp.ndarray of shape (n_blocks, *shape)
    """
    lst = jnp.array(lst)
    block_size = jnp.prod(jnp.array(shape))  # number of elements per block
    n_blocks = len(lst) // block_size

    # trim list to fit full blocks
    trimmed = lst[:n_blocks * block_size]

    # reshape into blocks
    reshaped = trimmed.reshape((n_blocks, *shape))
    return jnp.array(reshaped)
