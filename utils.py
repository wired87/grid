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
        namespace = {}

        # Wir fügen die LIBS direkt in den globalen Scope des exec ein
        exec(eq_code, LIBS, namespace)

        # Filtere alle Funktionen heraus
        callables = {
            k: v for k, v in namespace.items()
            if callable(v) and not k.startswith("__")
        }

        if not callables:
            raise ValueError("Keine Funktion im eq_code gefunden.")

        func_name = list(callables.keys())[-1]
        func = callables[func_name]

        """
        def wrapper(*args):
            return func(*args)
        """
        #print("func", func)
        return func
    except Exception as e:
        print(f"Err create_runnable: {e}")
        raise e


import inspect


def debug_callable(func):
    sig = inspect.signature(func)
    params = sig.parameters
    print(f"--- Debugging Callable: {func.__name__} ---")
    print(f"Anzahl Parameter (Signature): {len(params)}")
    print(f"Parameter Namen: {list(params.keys())}")

    # Prüfen auf Closures (versteckte Variablen)
    if hasattr(func, "__closure__") and func.__closure__:
        print(f"Anzahl versteckter Variablen (Closures): {len(func.__closure__)}")

    # Prüfen, ob es eine gebundene Methode ist (self-Problem)
    if hasattr(func, "__self__"):
        print("WARNUNG: Diese Funktion ist an ein Objekt gebunden (enthält 'self')!")
    return len(params)
