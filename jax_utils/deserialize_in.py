import json, base64, binascii
import numpy as np
import jax.numpy as jnp

def parse_value(o):
    if isinstance(o, str):
        s = o.strip()

        # JSON?
        if s.startswith(("{", "[", '"', "true", "false", "null")):
            try:
                o = json.loads(s)
                if isinstance(o, list):
                    return jnp.asarray(o)
            except json.JSONDecodeError:
                pass

        # Base64?
        try:
            raw = base64.b64decode(s, validate=True)
            arr = np.frombuffer(raw, dtype=np.complex64)
            return jnp.array(arr)
        except (binascii.Error, ValueError):
            return o

    return o
