
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



GUARD = None
def deploy_guard():
    ref = Guard.options(
        lifetime="detached",
        name="GUARD"
    ).remote()

    ref.main.remote()
    print("deploy_guard finished")
    return ref


if __name__ == "__main__":
    from guard import Guard
    guard = Guard()
    guard.main()
