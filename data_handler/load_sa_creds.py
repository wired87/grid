import os
from google.oauth2 import service_account


def load_service_account_credentials(file_path: str = None, scopes=None):
    """Loads a Service Account file into the universal Credentials object."""

    if file_path is None:
        file_path = r"C:\Users\bestb\PycharmProjects\BestBrain\auth\credentials.json" if os.name == "nt" else "auth/credentials.json"
    if os.path.exists(file_path):
        return service_account.Credentials.from_service_account_file(
            file_path,
            scopes=scopes
        )
    else:
        print("no creds found under", file_path)
