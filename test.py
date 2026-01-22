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

    print("Initializing Guard...")
    guard = Guard(config=data)
    
    print("Running Simulation...")
    guard.main()
    print("run_gnn_simulation... done")


if __name__ == "__main__":
    run_gnn_simulation()


