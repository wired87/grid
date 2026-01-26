import jax
import jax.numpy as jnp
from jax import jit, vmap


class InjectorLayer:

    def __init__(self, time, indices, values, db_layer, DIMS, amount_nodes):
        self.time = time
        self.indices = jnp.array(indices)
        self.values = jnp.array(values)

        self.db_layer=db_layer

        self.amount_nodes=amount_nodes
        self.DIMS=DIMS

        self.transfrom_injections_indices()

    def transfrom_injections_indices(self):
        print("transfrom_injections_indices...")

        def _transform_single(item):
            midx, fi, param_trgt_index, pos_index_slice = item

            _start, _len = self.db_layer.get_db_index(midx, fi, param_trgt_index)

            shape = self.db_layer.extract_shape(midx, fi, param_trgt_index)

            multiplicator = jax.lax.cond(
                jnp.array(shape).shape[0] > 0,
                lambda _: jnp.prod(jnp.array(shape)).astype(jnp.int32),  # True-Fall
                lambda _: 1,  # False-Fall
                operand=None
            )
            index = _start + (multiplicator * self.amount_nodes * self.DIMS)
            return index

        def _transform_batch(batch):
            return vmap(_transform_single, in_axes=0)(batch)

        # transform
        self.indices = vmap(_transform_batch)(self.indices)
        jax.debug.print(
            "transfrom_injections_indices... done {i}",
            i=self.indices
        )


    def inject_process(self, step, db_layer):
        """
        Applies injections based on self.inj_pattern and current step.
        Supports the SOA structure: [module, field, param, node_index, schedule]
        where schedule is list of [time, value].
        """
        jax.debug.print("inject_process...")

        def _inject():
            #tidx = self.time.index(step)
            tidx = jnp.argmax(self.time == step)
            # inject step
            db_layer.nodes.at[
                tuple(self.indices[tidx].T)
            ].add(self.values[tidx])

        is_step = jnp.any(self.time == step)

        jax.lax.cond(
            is_step,
            lambda _: _inject(),
            lambda _: 1,
            operand=None
        )
        jax.debug.print("inject_process... done")



"""
for item in self.INJECTIONS:
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
"""