import jax
import jax.numpy as jnp


SHIFT_DIRS = [
     [
        [ 1, 0, 0], [0,  1, 0], [0, 0,  1],
        [ 1, 1, 0], [1, 0,  1], [0, 1,  1],
        [ 1,-1, 0], [1, 0, -1], [0, 1, -1],
        [ 1, 1, 1], [1, 1,-1], [1,-1, 1], [1,-1,-1],
     ],
     [
        [-1, 0, 0], [0, -1, 0], [0, 0, -1],
        [-1,-1, 0], [-1, 0,-1], [0,-1,-1],
        [-1, 1, 0], [-1, 0, 1], [0,-1, 1],
        [-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
    ]
]





def build_db(amount_nodes, db, gpu):
    # create world
    jax.debug.print("build_db start")
    transformed = []
    for module_fields in db:
        for field, axis_def in module_fields:
            for value, ax in zip(field, axis_def):
                if ax: # 0
                    # fill db
                    value = jax.device_put(jnp.repeat(
                        jnp.asarray(jnp.zeros_like(value)),
                        amount_nodes
                    ), gpu)
    jax.debug.print("build_db fisniehed")


def set_shift(start_pos, shift_dirs, schema_grid):
    next_index_map = []
    for pos in start_pos:
        for dir in shift_dirs:
            neighbor = dir+pos
            next_index_map.append(
                schema_grid.index(neighbor)
            )
        next_index_map.append(
            schema_grid.index(pos)
        )
    return next_index_map

"""
progress: 
- world cfg -> grid size, len etc
- modules -> set fields and params -> link
- inj_pattern -> set stimuli for all fields
"""

def main():
    print(f"Initializing Simulation with JAX backend: {jax.devices()[0]}")
    # todo demo data structures



if __name__ == "__main__":
    main()
