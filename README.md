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
---

License:
All money earned with it must be collected within the POT


*Keep `test_out.json` at repo root. Use Cursor for nested todo rollup.*
