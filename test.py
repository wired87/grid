import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from guard import Guard


def set_env_variables(data, prefix=""):
    """
    Durchläuft rekursiv das Dictionary und setzt Umgebungsvariablen.
    Beispiel: {"db": {"port": 5432}} -> os.environ["DB_PORT"] = "5432"
    """
    for key, value in data.items():
        # Erstelle einen sauberen Schlüsselnamen (Großbuchstaben, Pfade mit _ verbunden)
        env_key = f"{prefix}{key}".upper()

        if isinstance(value, dict):
            # Rekursiver Aufruf für verschachtelte Dicts
            set_env_variables(value, prefix=f"{env_key}_")
        else:
            # Wert als String setzen (Umgebungsvariablen speichern nur Strings)
            os.environ[env_key] = str(value)
            # Optional: Print zur Kontrolle (Vorsicht bei Passwörtern!)
            # print(f"SET ENV: {env_key} = {value}")


def run_gnn_simulation():
    print("Loading test_out.json...")
    try:
        with open("test_out.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: test_out.json not found.")
        return

    print("Setting Environment Variables from JSON...")
    set_env_variables(data)

    print("Initializing Guard...")
    # Wir übergeben data weiterhin, falls Guard es intern benötigt,
    # aber die Werte sind nun auch via os.environ abrufbar.
    guard = Guard(config=data)

    print("Running Simulation...")
    guard.main()
    print("run_gnn_simulation... done")


if __name__ == "__main__":
    run_gnn_simulation()