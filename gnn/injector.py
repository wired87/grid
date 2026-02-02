import jax
import jax.numpy as jnp
from jax import vmap


class InjectorLayer:

    def __init__(
            self,
            INJECTOR_TIME,
            INJECTOR_INDICES,
            INJECTOR_VALUES,
            db_layer,
            DIMS,
            amount_nodes,
            **cfg
    ):
        self.time = INJECTOR_TIME
        self.INJECTOR_INDICES = jnp.array(INJECTOR_INDICES)
        self.values = jnp.array(INJECTOR_VALUES)

        self.db_layer=db_layer

        self.amount_nodes=amount_nodes
        self.DIMS=DIMS

        self.transfrom_injections_indices()

    def transfrom_injections_indices(self):
        print("transfrom_injections_indices...")
        if len(self.INJECTOR_INDICES):
            def _transform_single(item):
                midx, fi, param_trgt_index, pos_index_slice = item

                _start, _len = self.db_layer.get_db_index(
                    midx, fi, param_trgt_index
                )

                # get shape for targeted
                param_len = self.db_layer.get_abs_shape_idx(
                    midx, fi, param_trgt_index
                )
                index = _start + (
                        self.amount_nodes *
                        self.DIMS *
                        self.db_layer.DB_PARAM_CONTROLLER[
                            param_len
                        ]
                )
                return index

            def _transform_batch(batch):
                return vmap(_transform_single, in_axes=0)(batch)

            # transform
            self.indices = vmap(_transform_batch)(self.INJECTOR_INDICES)
            jax.debug.print(
                "transfrom_injections_indices... done {i}",
                i=self.indices
            )





    def inject(self, idx, db_layer, step):
        #tidx = self.time.index(step)
        jax.debug.print("inject_process...")
        print("time", self.time)
        print("step", step)

        print("self.indices[tidx]", self.indices[idx])
        print("self.values[tidx]", self.values[idx])
        # inject step
        db_layer.nodes.at[
            tuple(self.indices[idx].squeeze().T)
        ].add(self.values[idx])

        jax.debug.print("inject_process... done")

