import json
import os

from flax import linen as nn

from mod import Node
from utils import create_runnable


class CalcLayer(nn.Module):

    """
    Entry for all calcualtions within the GNN
    """

    def __init__(self):
        # DB Load and parse
        super().__init__(self)

        method_struct = os.getenv("METHOD_LAYER")
        self.method_struct = json.loads(method_struct)
        self.build_method_structs()

        def_out_db = os.getenv("METHOD_OUT_DB")
        self.def_out_db = json.loads(def_out_db)

        method_struct = os.getenv("METHOD_LAYER")
        self.method_struct = json.loads(method_struct)


    def calc_batch(self, new_g, old_g):
        # calc all methods and apply result to new g
        for i, module in enumerate(self.method_struct):
            for j, node in enumerate(module):
                result, features = node(
                    old_g,
                    new_g,
                    time_map
                )

                # apply result to fields param (overwrite)
                new_g[i][j][
                    self.def_out_db[i][j]
                ] = result

        return new_g













    def build_method_structs(self):
        for i, module in enumerate(self.method_struct):
            for j, method_struct in enumerate(module):
                node = Node(
                    runnable=create_runnable(method_struct[0]),
                    inp_edge_map=[inp[0] for inp in method_struct],
                    outp_pattern=[inp[1] for inp in method_struct],
                    in_axes_def=method_struct[2],
                    method_id=j,
                    mod_idx=i,
                )

                # overwrite raw method entry
                self.method_struct[i][j] = node

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