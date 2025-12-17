import jax.numpy as jnp
import optax

def compute_loss(predicted: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Mean Squared Error (MSE) loss between predicted and target values.
    
    Args:
        predicted: The predicted values from the model.
        target: The target values.
        
    Returns:
        The scalar MSE loss.
    """
    return optax.l2_loss(predicted, target).mean()
