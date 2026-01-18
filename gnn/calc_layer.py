import json
import os

from flax import linen as nn
import jax.numpy as jnp

from mod import Node
from utils import create_runnable
import jax

class CalcLayer(nn.Module):

    """
    Entry for all calcualtions within the GNN
    """

    def __init__(self):
        # DB Load and parse
        super().__init__(self)


        method_struct = os.getenv("METHOD_LAYER")
        self.method_struct = json.loads(method_struct)







    # ABRISSBIRNE
    def create_modules(self):
        chains = []
        for i, module in enumerate(self.updator_pattern):
            chain_nodes = []

            ###
            for j, method_struct in enumerate(module):
                chain_nodes.append(
                    Node(
                        runnable=create_runnable(method_struct[0]),
                        inp_edge_map=[inp[0] for inp in method_struct],
                        outp_pattern=[inp[1] for inp in method_struct],
                        in_axes_def=method_struct[2],
                        method_id=j,
                        mod_idx=i,
                    )
                )

            # represents a module with all equatioins
            chain = GnnModuleChain(
                modules_methods=chain_nodes,
            )
            chains.append(chain)
        return chains

