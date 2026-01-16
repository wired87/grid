from typing import Callable, Any, Sequence, List, NamedTuple
import jax.numpy as jnp

class NodeDataStore(NamedTuple):
    modules: jnp.ndarray


class GnnDataStore(NamedTuple):
    modules: jnp.ndarray

class Graph(NamedTuple):
    nodes: jnp.ndarray
    #edges: jnp.ndarray

    def copy(self) -> "Graph":
        return Graph(self.nodes, self.edges)

    def xtract_from_indices(self, mindex, findex, map):
        return self.nodes[mindex][findex][map]

    def __getitem__(self, key):
        # Ensure we return the correct slice of nodes
        return self.nodes[key]

    def tree_flatten(self):
        # Standard: return (leaves, aux_data)
        # nodes and edges are leaves as they are jax arrays
        return (self.nodes, self.edges), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

class TimeMap(NamedTuple):
    nodes: jnp.ndarray


class Payload(NamedTuple):
    grid_nodes: jnp.ndarray
    old: jnp.ndarray
    new: jnp.ndarray
    edges: jnp.ndarray