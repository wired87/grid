import jax
import jax.numpy as jnp



db = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

index_map = jnp.array([
    [0, 1, 2],
    [3, 4, 5],
    [1, 2, 6]
])

# 3. Die "Extraktion" auf der GPU (Blitzschnell)
def compute_node(params_for_one_node):
    return jnp.sum(params_for_one_node) # Deine Gleichung hier

# Wir mappen über die index_map, nicht über die Parameter direkt
def solve_all(flat_p, idx_map):
    # flat_p[idx_map] erzeugt ein Array der Form (100000, 3)
    # JAX macht das als hocheffizienten "Gather" auf der GPU
    node_params_batched = flat_p[idx_map]
    return jax.vmap(compute_node)(node_params_batched)

print(solve_all(db, index_map))