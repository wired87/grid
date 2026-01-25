import json
import os
import jax.numpy as jnp

from data_handler.main import load_data
from gnn.gnn import GNN


def identity_op(*args):
    # Simple dummy op that returns the first arg or zeros
    # Must return array of correct shape for bias addition (which adds to output)
    return args[0] if args else jnp.array([0.0])


DB_CONTROLLER = "DB_CONTROLLER"
MODEL_CONTROLLER = "MODEL_CONTROLLER"

class Guard:
    # todo prevaliate features to avoid double calculations
    def __init__(self):
        #JAX
        import jax
        platform = "cpu" if os.name == "nt" else "gpu"
        jax.config.update("jax_platform_name", platform)  # must be run before jnp
        self.gpu = jax.devices(platform)[0]

        # LOAD DAT FROM BQ OR LOCAL
        self.cfg = load_data()

        AMOUNT_NODES = int(os.getenv("AMOUNT_NODES", 10))
        SIM_TIME = int(os.getenv("SIM_TIME"))  # Default to 10 steps

        for k,v in self.cfg.items():
            setattr(self, k, json.loads(v))

        # layers
        self.gnn_layer = GNN(
            INJECTIONS,
            amount_nodes,
            method_to_db,
            time=SIM_TIME,
            db_to_method,
            METHODS,
            AXIS,
            DB,
            modules,
            gpu,
        )






    def main(self):
        self.run()
        results = self.finish()
        print("SIMULATION PROCESS FINISHED")
        return results


    def run(self):
        # start sim on gpu
        self.gnn_layer.main()
        print("run... done")

    def finish(self):
        # Collect data
        history_nodes = self.gnn_layer.db_layer.history_nodes
        model_skeleton = self.gnn_layer.model_skeleton
        
        # Serialization helper
        def serialize(data):
            if isinstance(data, list):
                return [serialize(x) for x in data]
            if isinstance(data, tuple):
                return tuple(serialize(x) for x in data)
            if isinstance(data, dict):
                 return {k: serialize(v) for k, v in data.items()}
            
            # Check for JAX/Numpy array
            if hasattr(data, 'dtype') and hasattr(data, 'real') and hasattr(data, 'imag'):
                # Check directly if complex dtype
                if jnp.iscomplexobj(data):
                    return (data.real, data.imag)
            return data

        serialized_history = serialize(history_nodes)
        serialized_model = serialize(model_skeleton)

        # Construct result dictionary
        result = {
            DB_CONTROLLER: serialized_history,
            MODEL_CONTROLLER: serialized_model
        }
        
        print("DATA DISTRIBUTED")
        return result


