import matplotlib.pyplot as plt
from typing import List
import numpy as np


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
