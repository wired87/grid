import jax
import jax.numpy as jnp
from flax import nnx
from jax import vmap

# --- Engine: optional components for simulation / training / query (ejkernel-style iteration) ---
try:
    from engine_components import (
        run_simulation_scan,
        run_training_step as _run_training_step_component,
        run_query_scan,
    )
    _ENGINE_COMPONENTS_AVAILABLE = True
except ImportError:
    _ENGINE_COMPONENTS_AVAILABLE = False


class Iterator:

    def __init__(self):
        pass

    # ---------- Engine: simulation ----------
    def run_simulation(self, carry, num_steps, step_fn, *, unroll=1):
        """Stable simulation loop (lax.scan). Use when simulation must be JAX-traceable."""
        if not _ENGINE_COMPONENTS_AVAILABLE:
            raise RuntimeError("engine_components not available; install or add to path.")
        return run_simulation_scan(carry, num_steps, step_fn, unroll=unroll)

    # ---------- Engine: training ----------
    def run_training_step(self, params, batch, loss_fn, *, opt_state=None, opt_update=None):
        """Single training step with optional optimizer update."""
        if not _ENGINE_COMPONENTS_AVAILABLE:
            raise RuntimeError("engine_components not available; install or add to path.")
        return _run_training_step_component(params, batch, loss_fn, opt_state=opt_state, opt_update=opt_update)

    # ---------- Engine: query ----------
    def run_query_scan(self, init_carry, inputs, query_fn, *, unroll=1):
        """Stable query/rollout over sequence (lax.scan)."""
        if not _ENGINE_COMPONENTS_AVAILABLE:
            raise RuntimeError("engine_components not available; install or add to path.")
        return run_query_scan(init_carry, inputs, query_fn, unroll=unroll)

    def pattern_recall(
        self,
        param_grid,
        time_map:jnp.array,
    ):
        """
        Use reconstruct anything based on single parameter
        """
        #print("param_grid", param_grid)
        try:
            # receive grid all variations
            #return linears for each param
            def _extract_most_similar_pts(param_current, prev_params):
                # Most similar past-time-step (pts). prev_params: (T, ...) from time_map.
                flat = jnp.ravel(param_current)
                embedding_current = jax.nn.gelu(
                    nnx.Linear(
                        in_features=len(flat),
                        out_features=self.d_model,
                        rngs=self.rngs,
                    )(flat),
                )
                closest_idx = self.closest_embedding(embedding_current, prev_params)
                return closest_idx

            def _extract_value_indices(indice):
                return jnp.take(time_map, indice)


            # One param per row; time_map broadcast (same for all params).
            indices = vmap(
                _extract_most_similar_pts,
                in_axes=(0, None),
            )(param_grid, time_map)

            # Optional: map indices to time values.
            # value_indices = _extract_value_indices(indices)
            return indices
        except Exception as e:
            print("Err gen_feature_single_variation", e)

