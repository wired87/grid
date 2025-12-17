from typing import NamedTuple
import jax.numpy as jnp

class Graph(NamedTuple):
    nodes: jnp.ndarray
    edges: jnp.ndarray