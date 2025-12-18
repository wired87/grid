import os
from typing import Sequence

import ray

from gnn.chain import GNNChain
from gnn.gnn import GNN
from main import build_db
from mod import Node
from test import get_demo_db, get_demo_data

@ray.remote(num_cpus=1, num_gpus=1)
class Guard:

    def __init__(self):
        #JAX
        import jax
        jax.config.update("jax_platform_name", "gpu")  # must be ron before jnp
        self.gpu = jax.devices("gpu")[0]

        #ENV
        self.db = os.getenv("DB", get_demo_db())
        self.updator_pattern = os.getenv("UPDATOR_PATTERN", get_demo_data())
        self.amount_nodes = int(os.getenv("AMOUNT_NODES", 10))
        self.injection_pattern = os.getenv("INJECTION_PATTERN")
        self.time = os.getenv("TIME")
        self.energy_map = os.getenv("ENERGY_MAP")

        #PARAMS
        self.chains: Sequence[GNNChain] = self.create_modules(
            self.updator_pattern
        )

    def create_modules(
            self,
            updator_pattern,
    ):
        # CREATE MODULE CHAINS BASED ON GIVEN
        for chain_struct in updator_pattern:
            chain_struct = GNNChain( # module
                modules_list=[
                    Node( # all method
                        *args
                    )
                    for args in chain_struct
                ],
            )
        return updator_pattern

    def main(self):
        self.prepare()
        self.run()
        self.finish()
        print("SIMULATION PROCESS FINISHED")

    def prepare(self):
        # laod env_vars
        self.nodes = build_db(
            self.amount_nodes,
            self.db,
            self.gpu
        )
        self.gnn = GNN(
            amount_nodes=self.amount_nodes,
            modules_len=len(self.db),
            updator_pattern=self.updator_pattern,
            nodes=self.nodes,
            inj_pattern=self.injection_pattern,
            glob_time=self.time,
            chains=self.chains
        )
        print("PREPARE FINISHED")


    def run(self):
        # start sim on gpu

        ray.get(
            self.gnn.main()
        )
        print("RUN FINISHED")

    def finish(self):
        # todo upser bq
        print("DATA DISTRIBUTED")




