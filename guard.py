import os
import json
from typing import Sequence
import jax
import jax.numpy as jnp
from dtypes import NodeDataStore, TimeMap
from gnn.calc_layer import CalcLayer
from gnn.chain import GnnModuleChain
from gnn.db_layer import DBLayer
from gnn.gnn import GNN
from gnn.injector import InjectorLayer
from mod import Node
from utils import create_runnable


def identity_op(*args):
    # Simple dummy op that returns the first arg or zeros
    # Must return array of correct shape for bias addition (which adds to output)
    return args[0] if args else jnp.array([0.0])


class Guard:
    # todo prevaliate features to avoid double calculations
    def __init__(self):
        #JAX
        import jax
        platform = "cpu" if os.name == "nt" else "gpu"
        jax.config.update("jax_platform_name", platform)  # must be run before jnp
        self.gpu = jax.devices(platform)[0]

        self.model = self.set_model_skeleton()

        # SET PATTERNS
        patterns = os.getenv("PATTERNS")
        self.patterns = json.loads(patterns)
        for key, pattern in self.patterns.items():
            setattr(self, key.lower(), json.loads(pattern) if pattern else [])

        self.amount_nodes = int(os.getenv("AMOUNT_NODES", 10))
        self.time = int(os.getenv("SIM_TIME", 10))  # Default to 10 steps


        # layers
        self.inj_layer = InjectorLayer()
        self.db_layer = DBLayer(self.amount_nodes)

        self.gnn_layer = GnnLayer()
        self.calc_layer = CalcLayer()

        # PARAMS
        self.chains: Sequence[
            GnnModuleChain
        ] = self.create_modules()



    def gnn_skeleton(self):
        # SET EMPTY STRUCTURE OF MODEL
        model_skeleton = []
        for i, module in enumerate(self.updator_pattern):
            model_skeleton[i] = []

            for j, method_struct in enumerate(module):
                model_skeleton[i].append(
                    [] #
                )
                field_block_matrice = []
                for adj_entry in method_struct:

                    # adj_entry = entry with 2 inputs
                    input_adj = adj_entry[0]
                    return_adj = adj_entry[1]

                    # merge in/out adj
                    adj_mtx = [*[input_adj], return_adj]

                    feature_complex = []

                    # store
                    feature_store = [feature_complex, adj_mtx]

                    # add field blcok struct
                    field_block_matrice.append(
                        feature_store
                    )

                # ADD FIELD BLOCK STRUCT TO MODEL
                model_skeleton[i][j].append(
                    field_block_matrice
                )
        return model_skeleton





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
            updator_pattern=None, # self.updator_pattern,
            nodes=self.nodes,
            inj_pattern=self.injection_pattern,
            glob_time=self.time,
            chains=self.chains,
            energy_map=self.energy_map
        )

        print("PREPARE FINISHED")



    def run(self):
        # start sim on gpu
        self.gnn.main()
        print("RUN FINISHED")

    def finish(self):
        # todo upser bq
        print("DATA DISTRIBUTED")


    def simulate(self, steps: int = None):
        try:
            if steps is None:
                steps = self.glob_time

            # Initialize connectivity
            self.set_shift()

            # History collection
            self.history = []

            for step in range(steps):
                jax.debug.print(
                    "Sim step {s}/{n}",
                    s=step + 1,
                    n=steps
                )

                # apply injections inside db layer
                self.inj_layer.inject(
                    step=step,
                    db_layer=self.db_layer
                )

                # prepare index map
                self.db_layer.process_time_step(

                )
                self.gnn_layer.process_nodes()

                new_g = self.calc_layer.calc_chains()

        except Exception as e:
            jax.debug.print(f"Err simulate: {e}")
            raise















"""
            for potential_args in chain_struct:
                if potential_args is None:
                    continue
                # Handle potential extra nesting level (e.g. [[arg, ...], [arg, ...]])
                # flattening it into the chain sequence
                
                # Normalize to a list of args_list
                if isinstance(potential_args, list) and len(potential_args) > 0 and isinstance(potential_args[0], list):
                    methods_list = potential_args
                else:
                    methods_list = [potential_args]

                for args in methods_list:
                    # Sanitize args from DEMO_INPUT format
                    # args is [desc, inp, outp, in_axes, method_id] (5 items)
                    sanitized_args = list(args)

                    # Ensure minimal length
                    while len(sanitized_args) < 5:
                        sanitized_args.append(None)
                    
                    # 1. Runnable (replace string with identity)
                    if isinstance(sanitized_args[0], str):
                        sanitized_args[0] = identity_op
                    
                    # 2. In-Axes / Pattern detection
                    # Correct argument mapping if JSON has patterns at index 3 or 4 but Node expects them at index 1 and 2
                    
                    # Check for Input Pattern at Index 3 (Common case in this JSON)
                    if len(sanitized_args) > 3 and isinstance(sanitized_args[3], list) and (len(sanitized_args[3]) > 0 and isinstance(sanitized_args[3][0], list)):
                        # If index 1 is not a list (e.g. 0), move it there
                        if not isinstance(sanitized_args[1], list):
                            sanitized_args[1] = sanitized_args[3]
                            sanitized_args[3] = 0 # Reset in_axes to 0
                    
                    # Check for Output Pattern (or secondary pattern) at Index 4
                    if len(sanitized_args) > 4 and isinstance(sanitized_args[4], list):
                        # If index 2 (outp_pattern) is None/0/not list, move it there
                        if not isinstance(sanitized_args[2], tuple): # outp_pattern is Tuple
                             # But here we assume the list from JSON is what we want.
                             # If arg 2 is None or 0.
                             if sanitized_args[2] is None or sanitized_args[2] == 0:
                                sanitized_args[2] = tuple(sanitized_args[4]) # Convert to tuple? Or keep list if Node handles it?
                                # Node expects Tuple for outp_pattern.
                                # But let's just move it.
                                sanitized_args[2] = sanitized_args[4]
                                # Clean up index 4 (method_id)
                                sanitized_args[4] = "generated_method_id"
                        elif not isinstance(sanitized_args[1], list):
                             # Fallback: if input pattern was NOT found at 3, maybe it's at 4?
                             sanitized_args[1] = sanitized_args[4]
                             sanitized_args[4] = "generated_method_id"

                    # Check inputs again


                    # Fallback for index 3 if it is a list of strings (axes) -> replace with 0 for safety
                    if len(sanitized_args) > 3 and isinstance(sanitized_args[3], list) and not isinstance(sanitized_args[3][0], list):
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

"""