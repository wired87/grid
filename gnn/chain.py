from typing import Sequence, Any, Tuple

import jax
from flax import nnx
from jax import jit
import jax.numpy as jnp

from loss import compute_loss
from mod import Node


from jax.tree_util import tree_map

class GNNChain(nnx.Module):

    """
    Kapselt die gesamte GNN-Kette für eine Simulations-Iteration (t -> t+1).
    """

    method_modules: Sequence[Node]

    def __init__(self, method_modules: Sequence[Node]):
        self.method_modules = method_modules

    @jit
    def __call__(self, old_g, new_g=None) -> jnp.ndarray:
        """
        Runs one simulation step (t -> t+1) 100% on GPU.
        """
        # Efficiently copy the nodes for the new state (G(t+1))
        self.old_g=old_g
        
        if new_g is None:
            self.new_g = tree_map(jnp.zeros_like, old_g)
        else:
            self.new_g=new_g

        # calc each module call
        for method_module in self.method_modules:
            self.new_g = self.processor(method_module)

        return self.new_g

    def processor(self, module:Node):
        """
        Method process all
        """
        jax.debug.print(
            "processor start for {defid}",
            defid=module.method_id
        )

        updated_g = module(
            old_g=self.old_g,
            new_g=self.new_g,
        )
        jax.debug.print("Simulations Iteration finished for: {defid}", defid=module.method_id)
        return updated_g


def simulation_step(
        gnn_chain: GNNChain,
        current_g: jnp.ndarray,
        optimizer,
        target_g: jnp.ndarray
) -> Tuple[jnp.ndarray, Any]:
    """
    Performs one simulation step, calculates loss, and updates parameters.
    This function is often decorated with @jit in the main training loop.
    """

    # Loss function for the entire chain
    def model_loss(model: GNNChain, input_g: jnp.ndarray, target_g: jnp.ndarray):
        predicted_g = model(input_g)
        return compute_loss(predicted_g, target_g), predicted_g

    # Calculate values and gradient
    (loss, predicted_g), grads = nnx.value_and_grad(model_loss, 'params')(
        gnn_chain, current_g, target_g
    )

    # Apply gradient update
    gnn_chain = optimizer.update(gnn_chain, grads)

    return predicted_g, gnn_chain, loss