# JAX GNN Simulation Engine

A high-performance, JAX/Flax-based simulation framework for modeling complex field dynamics ($t \to t+1$) on GPUs. It leverages `jax.jit` and `jax.vmap` for massive parallelism.

## � Structured Workflow

The engine follows a strict **Inject → Filter → Compute → Shift** cycle for each time step.



## 🛠️ ToDos
* create a field layer which inlcudes its own specific db (entire grid) (pros: targeted changes can be written without 
touching/changing everything (jax array are immuatbale so when chngine jst 1 energy val, th entire grid needs to 
be replaced.)) 


## 🛠️ Key Components

*   **`GNN` (Engine)**: Manages global state (`Graph`), simulation loops, and data injections.
*   **`GnnModuleChain`**: Orchestrates a sequence of `Node` modules, executed as a single compiled JAX block.
*   **`Node`**: The fundamental physics unit. Encapsulates equations and state (weights) using `nnx.Module`.

## ⚡ Features

*   **Active Node Filtering**: Optimizes compute by isolating energy-rich nodes (`TimeMap`).
*   **Efficient Memory**: Uses double-buffering (Old/New Graph) to manage state transitions.
*   **Distributed**: Designed to run as a Ray actor (`Guard`) for scalable deployment.

## � Quick Start
Define physics in a `Node`, wrap in a `GnnModuleChain`, and pass to the `GNN` engine for execution on the requested logic grid.
