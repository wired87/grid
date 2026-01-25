import json
import os

import dotenv

dotenv.load_dotenv()


def load_data():
    if os.getenv('TESTING').lower() == "true":
        return json.loads(open("test_out.json", "r").read())
    else:
        # load data bq
        pass