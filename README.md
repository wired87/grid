# JAX GNN Simulation Engine

JAX/Flax simulation for field dynamics ($t \to t+1$). Uses **Inject → Filter → Compute → Shift** per step.

## Workflows

**GNN run** (`.agent/workflows/gnn_wf.md`): Load `DEMO_INPUT` from `test.py` → parse JSON into env vars → init `Guard` → `Guard.main()` runs simulation → save `model.safetensors` → verify file exists.

## Key components

- **GNN**: Global state, simulation loop, injections.
- **GnnModuleChain**: Sequence of `Node` modules (one JAX block).
- **Node**: Physics unit; equations + state via `nnx.Module`.

## TODOs

- Track energetic time distribution over time
- Implement a blur to pre fill results based on in feature line (to not require calcualtion) -> Benedikt is on this
- Time Iterator: The Model builds on time (There is no Room -> just time is what matters)
- Implement total tiem step feature

## Time engine (todo / ideas)

**Current plan**

- **Alternative reality** – When a feature matches a past one (e.g. via blur), mark the equation branch as “alternative reality” so the model can validate time steps and optionally create new ones (zero-shot prediction).

**Ideas to consider**

- **Reality tags / branch IDs** – Persist a `reality_id` or `branch_id` per (eq, t, param) when we reuse a precomputed result; use it later for validation and analytics.
- **Time-step consistency check** – After each step, compare “alternative reality” nodes to the branch they came from (e.g. same feature → same outcome); flag or log drift.
- **Zero-shot horizon** – For nodes marked alternative reality, try predicting one or more future steps without running the full equation (e.g. copy from reference branch or small head).
- **Rollback / replay** – Store enough state (or hashes) per branch so we can rollback to a given `t` or replay from a past “reality” for debugging or counterfactuals.
- **Interpolation between time steps** – Optional interpolation (e.g. linear or learned) between `t` and `t+1` for smoother trajectories or denser logging.
- **Confidence / uncertainty over time** – Attach a simple confidence or uncertainty (e.g. distance to nearest blur match, or variance over similar past steps) to decide when to trust “alternative” vs recompute.
- **Canonical vs alternative** – Define one “canonical” reality per run (e.g. first branch or main path) and treat others as alternatives; validate alternatives against canonical at same `t`.
- **Cross-step invariants** – Define cheap invariants (e.g. conservation, bounds) and check them at each step; invalidate or flag alternative realities that break invariants.
- **Minimal state for validation** – Store only the minimal state needed to re-validate a time step (e.g. inputs + equation id + blur ref); use for offline checks without full replay.

---

License:
All money earned with it must be collected within the POT


*Keep `test_out.json` at repo root. Use Cursor for nested todo rollup.*
