import json
from typing import Dict
import numpy as np
import pandas as pd
from typing import List


def read_data(data_path: str) -> Dict:
    """

    :param data_path: takes a path to the json data, and returns the json as dictionary
    :return: data
    """

    return pd.read_csv(data_path)


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
    checks if array is clean
    frame is (11,2)
    """
    correct_shape = frame.shape[0] == 11
    nan_present = np.mean(np.isnan(frame)) == 0
    return correct_shape and nan_present


def check_halfcourt(frame: np.ndarray) -> bool:
    """
    checks if array is in half court set
    :param frame: n x 2 array
    :return: bool
    """
    half_court = (np.mean(frame[:, 0] >= 47) == 0) or (np.mean(frame[:, 0] <= 47) == 0)
    return half_court


def flip_court(frame: np.ndarray) -> np.ndarray:
    """
    orient court so all x <= 47 , assumes everyone on the court is on one half or the other
    :param frame: n x 2 array
    :return:
    """
    copy_frame = np.zeros_like(frame)
    copy_frame[:, 1] = frame[:, 1]
    copy_frame[:, 0] = np.min(frame[:, 0], 94 - frame[:, 0])  ### flip over half court line
    return copy_frame


def make_tensor(frame: np.ndarray, player_index: List[int], n_players: int, values: np.ndarray,
                resolution: tuple) -> np.ndarray:
    """

    :param frame: n x 2 array of locations
    :param player_index: n locating where in tensor to store values
    :param n_players: total number of players in tensor
    :param values: value at tensor location
    :param resolution: resolution of grid to plot location
    :return: matrix of ( n_players x resolution)
    """
    x, y = resolution
    tensor = np.zeros((n_players, x, y))
    resolved_locations = np.round(frame)
    tensor[player_index, resolved_locations[:, 0], resolved_locations[:, 1]] = values

    return tensor


