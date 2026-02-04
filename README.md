# JAX GNN Simulation Engine

JAX/Flax simulation for field dynamics ($t \to t+1$). Uses **Inject → Filter → Compute → Shift** per step.

## Workflows

**GNN run** (`.agent/workflows/gnn_wf.md`): Load `DEMO_INPUT` from `test.py` → parse JSON into env vars → init `Guard` → `Guard.main()` runs simulation → save `model.safetensors` → verify file exists.

## Key components

- **GNN**: Global state, simulation loop, injections.
- **GnnModuleChain**: Sequence of `Node` modules (one JAX block).
- **Node**: Physics unit; equations + state via `nnx.Module`.

## TODOs

- Emergent behaviour from time injections (e.g. blur inactive; use `FeatureEncoder.feature_history`).
- Debug the core.
- Classify params: **dynamic** (feature/time-dependent) vs **constant**.


---

*Keep `test_out.json` at repo root. Use Cursor for nested todo rollup.*
