import os
from utils.data_utils import *
from gravity.gravity import *

"""
Input data should be in the /data folder.
Data will be outputted to /processed_data folder
"""


def conditional_gravity(is_half: bool, moment: str) -> List:
    """

    :param is_half:
    :param moment: str representation of moment
    :return: empty list if its half court, else list of gravity values in list
    """
    if is_half:
        return calculate_gravity(flip_court(string2array(moment)[:, 2:4])).tolist()
    else:
        return []


def map_gravity(is_valid: bool, gravity: list, moment: str) -> Dict:
    if is_valid:
        moment_array = string2array(moment)
        return {int(key): val for key, val in zip(moment_array[:, 1].tolist() + [-2], gravity)}
    else:
        return {}


if __name__ == "__main__":
    for data_file in [os.path.join("data", f) for f in os.listdir("data")]:
        raw_data = read_data(data_file)
        raw_data["is_valid"] = raw_data["moment_info"].apply(lambda x: check_valid(string2array(x)))
        raw_data["is_halfcourt"] = raw_data["moment_info"].apply(lambda x: check_halfcourt(string2array(x)[:, 2:4]))
        raw_data["process_frame"] = raw_data["is_halfcourt"] & raw_data["is_valid"]
        raw_data["gravity"] = raw_data[["process_frame", "moment_info"]].apply(lambda x: conditional_gravity(*x),
                                                                               axis=1)
        raw_data["mapped_gravity"] = raw_data[["process_frame", "gravity", "moment_info"]].apply(
            lambda x: map_gravity(*x), axis=1)
        raw_data.to_csv("processed_" + data_file, index=False)
