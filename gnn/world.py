import jax
import jax.numpy as jnp

class CreateWorld:

    """

    HANDLES THE CREATION OF A CLUSTER
    Creator for bare empty sism World
    Start/Change ->
    Adapt Config ->
    Model Changes ->
    Results Upload Graph ->
    Render.

    """

    def __init__(
            self,
            amount_nodes,
            db,
            gpu,
    ):
        self.amount_nodes=amount_nodes
        self.db=db
        self.gpu=gpu
        self.soa = {}

    def build_db(self):
        # create world
        jax.debug.print("build_db start")
        transformed = []
        for module_fields in self.db:
            for field, axis_def in module_fields:
                for value, ax in zip(field, axis_def):
                    if ax:  # 0
                        # fill db
                        value = jax.device_put(jnp.repeat(
                            jnp.asarray(jnp.zeros_like(value)),
                            self.amount_nodes
                        ), self.gpu)
        jax.debug.print("build_db fisniehed")

