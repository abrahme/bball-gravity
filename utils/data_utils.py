import json
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def read_data(data_path: str) -> Dict:
    """

    :param data_path: takes a path to the json data, and returns the json as dictionary
    :return: data
    """
    with open(data_path, "r") as f:
        data = json.loads(f.read())
    f.close()
    return data


def write_data(data_path: str, data: Dict) -> None:
    """
    :param data: data to dump
    :param data_path: a path to where to write data
    :return:
    """
    with open(data_path, "w") as f:
        json.dump(data, f)
    f.close()


def check_valid(frame: np.ndarray) -> bool:
    """
    checks if array is a half court set
    frame is (11,2)
    """
    correct_shape = frame.shape[0] == 11
    nan_present = np.mean(np.isnan(frame)) == 0
    return correct_shape and nan_present
