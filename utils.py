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

import jax.numpy as jnp
import jax

LIBS={
    "jax": jax,
    "vmap": jax.vmap,
    "jnp": jnp,
    "jit": jax.jit,
}


def create_runnable(eq_code):
    try:
        local_vars = {}
        exec(eq_code, LIBS, local_vars)

        # Suche alle aufrufbaren Objekte, die NICHT aus den LIBS kommen
        # und keine internen Python-Attribute (__name__ etc.) sind
        callables = [
            v for k, v in local_vars.items()
            if callable(v) and not k.startswith("__")
        ]

        if not callables:
            raise ValueError("Keine aufrufbare Funktion im eq_code gefunden.")

        # Nimm die letzte definierte Funktion
        return callables[-1]

    except Exception as e:
        print(f"Err create_runnable:", e)
        raise e
