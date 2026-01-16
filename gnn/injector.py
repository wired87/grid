import json
import os
from flax import linen as nn
import jax.numpy as jnp


class InjectorLayer(nn.Module):

    # gnn entry -> injects in db

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        injection_pattern = os.getenv("INJECTION_PATTERN")
        self.injection_pattern = json.loads(injection_pattern)


    def inject(self, step, db_layer):
        """
        Applies injections based on self.inj_pattern and current step.
        Supports the SOA structure: [module, field, param, node_index, schedule]
        where schedule is list of [time, value].
        """
        all_indices = []
        all_values = []

        for item in self.injection_pattern:
            if isinstance(item, list) and len(item) == 5 and isinstance(item[4], list):
                # SOA structure: [mod, field, param, node_idx, schedule]
                mod_idx, field_idx, param_idx, node_idx, schedule = item

                # Check schedule for current time
                for time_point, value in schedule:
                    if time_point == step:
                        # Add to update list
                        # Assuming feature index 0
                        all_indices.append([mod_idx, field_idx, param_idx, node_idx])# rm 0?
                        all_values.append(value)

        if not all_indices:
            return

        all_indices = jnp.array(all_indices)  # shape [N, 5]
        all_values = jnp.array(all_values)  # shape [N]

        # inject step
        db_layer.nodes.at[
            tuple(all_indices.T)
        ].add(all_values)


