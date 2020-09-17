import json
from typing import Dict


def read_data(data_path: str) -> Dict:
    """

    :param data_path: takes a path to the json data, and returns the json as dictionary
    :return: data
    """
    with open(data_path, "r") as f:
        data = json.loads(f.read())
    f.close()
    return data
