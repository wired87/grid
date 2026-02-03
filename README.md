# JAX GNN Simulation Engine

A high-performance, JAX/Flax-based simulation framework for modeling complex field dynamics ($t \to t+1$) on GPUs. 
Focus on time evolution of single paranmeters 

## � Structured Workflow

The engine follows a strict **Inject → Filter → Compute → Shift** cycle for each time step.

# keep test_out.json as root

## 🛠️ ToDos (please use Cursor for project nested todo sum)
* add emergent behaviour based on time injections (blur less active behaviour: track this with FeatureEncoder.feature_history[:5] -> e.g. merge from
*  debug the core 
* classify parameters to dynamic(feature score based on time) and constant(never change) 
* 


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
