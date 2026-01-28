
import os
import jax.numpy as jnp

from data_handler.main import load_data
from gnn.gnn import GNN

import jax

from jax_utils.deserialize_in import parse_value


class Guard:
    # todo prevaliate features to avoid double calculations
    def __init__(self):
        #JAX
        platform = "cpu" if os.name == "nt" else "gpu"
        jax.config.update("jax_platform_name", platform)  # must be run before jnp
        self.gpu = jax.devices(platform)[0]

        # LOAD DAT FROM BQ OR LOCAL
        self.cfg = load_data()

        AMOUNT_NODES = int(os.getenv("AMOUNT_NODES"))
        SIM_TIME = int(os.getenv("SIM_TIME"))
        DIMS = int(os.getenv("DIMS"))

        for k, v in self.cfg.items():
            self.cfg[k] = parse_value(v)

            if isinstance(self.cfg[k], dict):
                for i, o in self.cfg[k].items():
                    self.cfg[k][i] = parse_value(o)

        # layers
        self.gnn_layer = GNN(
            amount_nodes=AMOUNT_NODES,
            time=SIM_TIME,
            gpu=self.gpu,
            DIMS=DIMS,
            **self.cfg
        )


    def main(self):
        self.run()
        results = self.finish()
        print("SIMULATION PROCESS FINISHED")
        return results


    def run(self):
        # start sim on gpu
        print("run...")
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
            "DB_CONTROLLER": serialized_history,
            "MODEL_CONTROLLER": serialized_model
        }

        print("DATA DISTRIBUTED")
        return result


