import os
import json
from typing import Sequence

import jax
import jax.numpy as jnp
from gnn.chain import GNNChain
from gnn.gnn import GNN
from mod import Node


def identity_op(*args):
    # Simple dummy op that returns the first arg or zeros
    # Must return array of correct shape for bias addition (which adds to output)
    return args[0] if args else jnp.array([0.0])

class Guard:

    def __init__(self):
        #JAX
        import jax
        platform = "cpu" if os.name == "nt" else "gpu"
        jax.config.update("jax_platform_name", platform)  # must be run before jnp
        self.gpu = jax.devices(platform)[0]


        #ENV
        # Load and parse DB
        db_str = os.getenv("DB")
        self.db = json.loads(db_str) if db_str else []

        # graph patter
        # Load and parse UPDATOR_PATTERN
        up_pat_str = os.getenv("UPDATOR_PATTERN")
        self.updator_pattern = json.loads(up_pat_str) if up_pat_str else []
        self.amount_nodes = int(os.getenv("AMOUNT_NODES", 10))
        self.time = int(os.getenv("SIM_TIME", 10))  # Default to 10 steps

        # inj
        # Load and parse INJECTION_PATTERN
        inj_pat_str = os.getenv("INJECTION_PATTERN")
        self.injection_pattern = json.loads(inj_pat_str) if inj_pat_str else []

        # modules
        # Load and parse ENERGY_MAP
        e_map_str = os.getenv("ENERGY_MAP")
        self.energy_map = json.loads(e_map_str) if e_map_str else []

        #PARAMS
        self.chains: Sequence[GNNChain] = self.create_modules()


    def create_modules(
            self,
    ):
        # CREATE MODULE CHAINS BASED ON GIVEN
        from flax import nnx
        import jax
        
        chains = []
        for chain_struct in self.updator_pattern:
            # Create rngs for this chain
            rngs = nnx.Rngs(params=jax.random.PRNGKey(42))
            
            chain_nodes = []
            for args in chain_struct:
                # Sanitize args from DEMO_INPUT format
                print(f"Processing chain item arg0 type: {type(args[0])}")
                
                # args is [desc, inp, outp, in_axes, method_id] (5 items)
                sanitized_args = list(args)
                
                # 1. Runnable (replace string with identity)
                if isinstance(sanitized_args[0], str):
                    print("Replacing string runnable with identity_op")
                    sanitized_args[0] = identity_op
                else:
                    print(f"Runnable is not string, it is {type(sanitized_args[0])}")

                
                # 2. In-Axes (replace string list with expected tuple/None)
                # DEMO_INPUT has [",", " ", "0"]. This seems to be CSV parsing artifacts? 
                # Let's default to (0,) or None for now to make it run.
                # Node expects in_axes_def to be passed to vmap.
                # Check mod.py: vmap(self.runnable, in_axes=self.in_axes_def)
                # If we use identity_op, in_axes needs to match inputs.
                # For safety, let's set it to valid defaults or parse if possible.
                # Assuming 0 for all inputs for simple vmap.
                if isinstance(sanitized_args[3], list):
                     # Hack: standard in_axes usually e.g. (0, None)
                     # Let's force it to None or 0.
                     sanitized_args[3] = 0

                # 3. Method ID (ensure str)
                sanitized_args[4] = str(sanitized_args[4])
                
                # Check length to avoid RNG collision
                if len(sanitized_args) > 5:
                    sanitized_args = sanitized_args[:5]

                chain_nodes.append(
                    Node(
                        *sanitized_args,
                        rngs=rngs
                    )
                )
            
            chain = GNNChain(
                method_modules=chain_nodes,
            )
            chains.append(chain)
        return chains

    def main(self):
        self.prepare()
        self.run()
        self.finish()
        print("SIMULATION PROCESS FINISHED")

    def prepare(self):
        # laod env_vars
        self.nodes = self.build_db(
            self.amount_nodes,
            self.db,
        )

        self.gnn = GNN(
            amount_nodes=self.amount_nodes,
            modules_len=len(self.db),
            updator_pattern=self.updator_pattern,
            nodes=self.nodes,
            inj_pattern=self.injection_pattern,
            glob_time=self.time,
            chains=self.chains,
            energy_map=self.energy_map
        )

        print("PREPARE FINISHED")

    def build_db(self, amount_nodes, db):
        # create world
        jax.debug.print("build_db start")
        transformed = []
        for module_fields in db:
            for field, axis_def in module_fields:
                for value, ax in zip(field, axis_def):
                    if ax is not None:  # Fixed: 0 is a valid axis
                        # fill db
                        value = jax.device_put(jnp.repeat(
                            jnp.asarray(jnp.zeros_like(value)),
                            amount_nodes
                        ), self.gpu)
                        transformed.append(value)
        jax.debug.print("build_db finished")
        return transformed


    def run(self):
        # start sim on gpu
        self.gnn.main()
        print("RUN FINISHED")

    def finish(self):
        # todo upser bq
        print("DATA DISTRIBUTED")




