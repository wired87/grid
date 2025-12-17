
import jax
import jax.numpy as jnp
from flax import nnx
from jax.tree_util import Partial
from gnn.gnn import Graph
from mod import Node

def test_mod_only():
    print("Test Mod Only")
    nodes = jnp.ones((2, 5, 2))
    graph = Graph(nodes=nodes, edges=[])

    rngs = nnx.Rngs(0)
    
    def my_op(x):
        return x + 1.0

    mod = Node(
        runnable=my_op, 
        inp_patterns=[(0,)],
        outp_pattern=(0,),
        in_axes_def=(0,),
        method_id="test",
        rngs=rngs
    )
    
    print("Running Module...")
    res_g = mod(graph, graph)
    print("Success!")
    print(res_g.nodes[0])

if __name__ == "__main__":
    test_mod_only()
