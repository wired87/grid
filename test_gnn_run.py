import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from guard import Guard

def run_gnn_simulation():
    print("Loading test_out.json...")
    try:
        with open("test_out.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: test_out.json not found.")
        return

    os.environ["AMOUNT_NODES"] = int(data.get("AMOUNT_NODES", 10))
    os.environ["SIM_TIME"] = int(data.get("SIM_TIME", 10))
    
    print("Initializing Guard...")
    # Pass data directly to bypass Windows env var limits
    guard = Guard(
        db=data.get("DB"),
        updator_pattern=data.get("UPDATOR_PATTERN"),
        injection_pattern=data.get("INJECTION_PATTERN"),
        energy_map=data.get("ENERGY_MAP")
    )
    
    print("Running Simulation...")
    guard.main()
    
    print("Collecting generated data...")
    # Accessing the nodes from the GNN instance within guard
    if hasattr(guard.gnn, 'history'):
        history = guard.gnn.history
        print(f"History collected: {len(history)} steps.")
        
        # Save history
        import numpy as np
        try:
            # Convert JAX arrays to Numpy arrays for saving
            history_np = [np.array(h) for h in history]
            np.save("simulation_history.npy", history_np)
            print("Success: simulation_history.npy created.")
        except Exception as e:
            print(f"Error saving history: {e}")
            
        if history:
            final_nodes = history[-1]
            print(f"Final Nodes Shape: {final_nodes.shape}")
    else:
        print("Warning: No history found in guard.gnn.")
        final_nodes = guard.gnn.graph.nodes
        print(f"Final Nodes Shape (from graph): {final_nodes.shape}")
    
    # Verify model file
    if os.path.exists("model.safetensors"):
        print("Success: model.safetensors created.")
    else:
        print("Warning: model.safetensors not found.")



