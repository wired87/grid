from typing import Sequence
from flax import nnx
import jax.numpy as jnp


from mod import Node


class GnnModuleChain(nnx.Module):

    """
    Kapselt die gesamte GNN-Kette für eine Simulations-Iteration (t -> t+1).
    """

    modules_methods: Sequence[Node]

    def __init__(
            self,
            modules_methods: Sequence[Node]
    ):
        self.modules_methods = modules_methods

    def __call__(self, old_g, new_g=None, time_map=None) -> jnp.ndarray:
        """
        Runs one simulation step (t -> t+1) 100% on GPU.
        """
        # Efficiently copy the nodes for the new state (G(t+1))
        self.old_g = old_g
        self.gnn_ds = gnn_ds
        self.new_g = new_g

        # calc each modules method
        for method_module in self.modules_methods:
            self.new_g, _features = self.processor(method_module, time_map)


        for i, feature in enumerate(features):
            self.new_g[
                self.mod_idx
            ][
                self.method_id
            ][
                i, # field index
            ].append(
                # projections matrices for all ime steps
                # weights
                # biases
                (
                    feature,
                )
            )

        return self.new_g, features

    def processor(self, module:Node, time_map):
        """
        Method process all
        """
        # jax.debug.print(
        #     "processor start for {defid}",
        #     defid=module.method_id
        # )

        updated_g = module(
            old_g=self.old_g,
            new_g=self.new_g,
            time_map=time_map
        )
        # jax.debug.print("Simulations Iteration finished for: {defid}", defid=module.method_id)
        return updated_g











"""
def simulation_step(
        gnn_chain: GnnModuleChain,
        current_g: jnp.ndarray,
        optimizer,
        target_g: jnp.ndarray
) -> Tuple[jnp.ndarray, Any]:
    
    Performs one simulation step, calculates loss, and updates parameters.
    This function is often decorated with @jit in the main training loop.
    

    # Loss function for the entire chain
    def model_loss(model: GnnModuleChain, input_g: jnp.ndarray, target_g: jnp.ndarray):
        predicted_g = model(input_g)
        return compute_loss(predicted_g, target_g), predicted_g

    # Calculate values and gradient
    (loss, predicted_g), grads = nnx.value_and_grad(model_loss, 'params')(
        gnn_chain, current_g, target_g
    )

    # Apply gradient update
    gnn_chain = optimizer.update(gnn_chain, grads)

    return predicted_g, gnn_chain, loss
"""