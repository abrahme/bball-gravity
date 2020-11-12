import json
import ast
import pickle
from typing import Dict
import numpy as np
import pandas as pd


def read_data(data_path: str) -> pd.DataFrame:
    """

    :param data_path: takes a path to the json data, and returns the json as dictionary
    :return: data
    """

    return pd.read_csv(data_path)

def read_json(data_path:str) -> Dict:
    """

    :param data_path: self explanatory
    :return:
    """
    with open(data_path,"r") as f:
        data = json.load(f)
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


def pickle_data(data_path: str, data) -> None:
    """

    :param data_path: string where to pickle
    :param data: python object
    :return:
    """
    with open(data_path, "wb") as f:
        pickle.dump(data, f)
    f.close()


def unpickle_data(data_path: str):
    """

    :param data_path: string where to unpickle
    :return:
    """
    with open(data_path, "rb") as f:
        unpickled = pickle.load(f)
    f.close()
    return unpickled


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
    copy_frame[:, 0] = np.minimum(frame[:, 0], 94 - frame[:, 0])  ### flip over half court line
    return copy_frame


def string2array(x: str) -> np.ndarray:
    """

    :param x: string representation of moment array
    :return: numpy array of moment
    """
    return np.array([ast.literal_eval(val) for val in x.split(";")])
