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

DIM = 3



def create_runnable(
        eq_code,
        eq_key,
        xtrn_mods
):
    print(f"create_runnable, {eq_key}")
    try:
        local_vars = {}

        exec(eq_code, xtrn_mods, local_vars)

        # Extract the function (must be defined as def f(...))
        f = local_vars.get(eq_key)

        if f is None:
            raise ValueError(f"Function {eq_key} not found in eq_code")
        return f
    except Exception as e:
        print(f"Err create_runnable ({eq_key}):", e)

