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


def plot_frame(moment_location: List[List]) -> None:
    """
    plot a frame of data
    :param moment_location: information of one frame of data
    :return:
    """
    frame = np.array(moment_location)
    teams = set(frame[:, 0])
    color_map = {val: key for key, val in enumerate(teams)}
    plt.scatter(frame[:, 2], frame[:, 3], c=[color_map[key] for key in frame[:, 0]])
    plt.show()


def standardize_frame(frame: np.ndarray) -> np.ndarray:
    """
    given 2-D array of x, y standard frame of reference
    with all members on one side of the court
    return standardized frame
    """
    if np.mean(frame[:, 0] > 47) == 1:
        # if the play is on one side of the half court, flip it over
        frame[:, 0] = 94 - frame[:, 0]  # moves over the y axis
        frame[:, 1] = 50 - frame[:, 1]  # moves over x axis

    return frame


def check_valid(frame: np.ndarray) -> bool:
    """
    checks if array is a half court set
    frame is (11,2)
    """
    return (np.mean(frame[:, 2] > 47) == 1) or (np.mean(frame[:, 2] < 47) == 1)
