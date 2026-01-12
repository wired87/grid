import jax

from guard import Guard









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

GUARD = None
def deploy_guard():
    ref = Guard.options(
        lifetime="detached",
        name="GUARD"
    ).remote()

    ref.main.remote()
    print("deploy_guard finished")
    return ref

def test():
    result = Guard().main()

def main():
    print(f"Initializing Simulation with JAX backend: {jax.devices()[0]}")
    # todo demo data structures
    GUARD = deploy_guard()


if __name__ == "__main__":
    test()
