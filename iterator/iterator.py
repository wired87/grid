import jax
import jax.numpy as jnp
from jax import lax, vmap

# --- Engine: optional components ---
try:
    from engine_components import (
        run_simulation_scan,
        run_training_step as _run_training_step_component,
        run_query_scan,
    )
    _ENGINE_COMPONENTS_AVAILABLE = True
except ImportError:
    _ENGINE_COMPONENTS_AVAILABLE = False

# --- Time ctlr: (in_grid, out_grid). in_grid=(T,N_in,D), out_grid=(T,N_out,D). No str. ---
def build_time_ctlr(in_store, out_store, d_model=64):
    """(in_grid, out_grid) from ragged stores. Padded (T,N,D)."""
    def to_grid(store):
        T = max((len(hist) for eq in store for hist in eq), default=1)
        rows = []
        for t in range(T):
            row_t = []
            for eq in store:
                for hist in eq:
                    t < len(hist) and row_t.append(jnp.reshape(jnp.asarray(hist[t]), (-1, jnp.size(jnp.asarray(hist[t])))))
            n = max((r.shape[0] for r in row_t), default=1)
            d = max((r.shape[1] for r in row_t), default=d_model)
            pad = jnp.zeros((n, d), dtype=jnp.float32)
            for r in row_t:
                sh0, sh1 = min(r.shape[0], n), min(r.shape[1], d)
                pad = pad.at[:sh0, :sh1].set(r[:sh0, :sh1])
            rows.append(pad)
        out = jnp.stack(rows, axis=0)
        return lax.cond(out.size == 0, lambda: jnp.zeros((1, 1, d_model), dtype=jnp.float32), lambda: out)
    in_store = in_store or [[[]]]
    out_store = out_store or [[[]]]
    in_g = to_grid(in_store)
    out_g = to_grid(out_store)
    T = max(in_g.shape[0], out_g.shape[0])
    N_in = in_g.shape[1]
    N_out = out_g.shape[1]
    D = max(in_g.shape[2], out_g.shape[2], d_model)
    in_g = jnp.pad(in_g, ((0, T - in_g.shape[0]), (0, 0), (0, D - in_g.shape[2])))[:T, :N_in, :D]
    out_g = jnp.pad(out_g, ((0, T - out_g.shape[0]), (0, 0), (0, D - out_g.shape[2])))[:T, :N_out, :D]
    return (in_g, out_g)


class Iterator:
    def __init__(self, d_model=64):
        self.d_model = d_model

    def run_simulation(self, carry, num_steps, step_fn, *, unroll=1):
        return lax.cond(_ENGINE_COMPONENTS_AVAILABLE, lambda: run_simulation_scan(carry, num_steps, step_fn, unroll=unroll), lambda: carry)

    def run_training_step(self, params, batch, loss_fn, *, opt_state=None, opt_update=None):
        return lax.cond(_ENGINE_COMPONENTS_AVAILABLE, lambda: _run_training_step_component(params, batch, loss_fn, opt_state=opt_state, opt_update=opt_update), lambda: (params, opt_state))

    def run_query_scan(self, init_carry, inputs, query_fn, *, unroll=1):
        return lax.cond(_ENGINE_COMPONENTS_AVAILABLE, lambda: run_query_scan(init_carry, inputs, query_fn, unroll=unroll), lambda: (init_carry, None))

    # --- All functions on time ctlr (in_grid, out_grid). No str, minimal branch. ---
    def locate_feature(self, feature, ctlr):
        """Return (time_idx, eq_idx, abs_variation_idx) for best match. ctlr = (in_grid, out_grid)."""
        feat = jnp.ravel(jnp.asarray(feature, dtype=jnp.float32))
        in_g, out_g = ctlr[0], ctlr[1]
        flat_in = jnp.reshape(in_g, (-1, in_g.shape[-1]))
        flat_out = jnp.reshape(out_g, (-1, out_g.shape[-1]))
        all_ = jnp.concatenate([flat_in, flat_out], axis=0)
        dists = jnp.linalg.norm(all_ - feat, axis=-1)
        idx = jnp.argmin(dists)
        n_in = flat_in.shape[0]
        t_in, n_per_t = in_g.shape[0], in_g.shape[1]
        t_out, m_per_t = out_g.shape[0], out_g.shape[1]
        in_offset = idx < n_in
        flat_idx = lax.cond(in_offset, lambda: idx, lambda: idx - n_in)
        t_idx = lax.cond(in_offset,
                         lambda: flat_idx // n_per_t,
                         lambda: flat_idx // m_per_t)
        var_idx = lax.cond(in_offset,
                          lambda: flat_idx % n_per_t,
                          lambda: flat_idx % m_per_t)
        eq_idx = jnp.int32(0)
        abs_var = flat_idx
        return (t_idx, eq_idx, abs_var)

    def inject_time_loop(self, feature, ctlr, loop_score):
        """If any past feature matches >= loop_score return that past feature else current."""
        feat = jnp.ravel(jnp.asarray(feature, dtype=jnp.float32))
        in_g, out_g = ctlr[0], ctlr[1]
        hist = jnp.concatenate([jnp.reshape(in_g, (-1, in_g.shape[-1])),
                                jnp.reshape(out_g, (-1, out_g.shape[-1]))], axis=0)
        empty = hist.size == 0
        hist = lax.cond(empty, lambda: jnp.reshape(feat, (1, -1)), lambda: hist)
        hist = jnp.reshape(hist, (-1, feat.shape[0]))
        dists = jnp.linalg.norm(hist - feat, axis=-1)
        scores = 1.0 / (1.0 + dists)
        max_idx = jnp.argmax(scores)
        max_score = scores[max_idx]
        return lax.cond(
            max_score >= loop_score,
            lambda _: hist[max_idx],
            lambda _: feat,
            operand=None,
        )

    def scan_in_out_features(self, ctlr):
        """Return (array_scores, array_idx_of_entry_within_corresponding_time_step)."""
        in_g, out_g = ctlr[0], ctlr[1]
        T = in_g.shape[0]
        curr = jnp.concatenate([in_g, out_g], axis=1)
        prev = jnp.roll(curr, 1, axis=0).at[0].set(0.0)
        def score_one(t):
            c = curr[t]
            p = prev[t]
            d = jnp.linalg.norm(p - c[:, jnp.newaxis, :], axis=-1)
            min_d = jnp.min(d, axis=1)
            return 1.0 / (1.0 + min_d)
        scores = vmap(score_one)(jnp.arange(T))
        scores = jnp.reshape(scores, (-1,))
        idx_within_t = jnp.arange(T * curr.shape[1], dtype=jnp.int32) % curr.shape[1]
        return (jnp.asarray(scores, dtype=jnp.float32),
                jnp.asarray(idx_within_t, dtype=jnp.int32))

    def pattern_recall(self, param_grid, time_map):
        """Most similar past index per param row."""
        def dist_row(a, b):
            return jnp.linalg.norm(jnp.ravel(a) - jnp.ravel(b))
        def closest_one(row):
            dists = vmap(lambda t: dist_row(row, time_map[t]))(jnp.arange(time_map.shape[0]))
            return jnp.argmin(dists)
        return vmap(closest_one)(param_grid)
