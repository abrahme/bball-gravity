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


def find_offense_defense(is_valid: bool, offense_team_id: str, moment: str) -> str:
    """

    :param is_valid: check frame
    :param offense_team_id: team id of offense
    :param moment: str rep of moment
    :return:
    """
    if is_valid and offense_team_id != '(null)':
        frame = string2array(moment)
        offense = frame[:, 1][(frame[:, 0] == float(offense_team_id))].tolist()
        defense = frame[:, 1][(frame[:, 0] != float(offense_team_id)) & (frame[:, 0] != -1)].tolist()
        return json.dumps({"offense": offense, "defense": defense})
    else:
        return ""


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
        raw_data["offense_defense"] = raw_data[["process_frame", "TEAM_ID", "moment_info"]].apply(
            lambda x: find_offense_defense(*x), axis=1
        )
        raw_data.to_csv("processed_" + data_file, index=False)

        ### now we upload possessions as json
        pos_data = raw_data[(raw_data["process_frame"] == True) & (raw_data["TEAM_ID"] != '(null)')]
        possessions = set(pos_data["POSSESSION_ID"])
        for pos in possessions:
            sub_pos = pos_data[pos_data["POSSESSION_ID"] == pos]
            data_dump = {"pos_moments": np.concatenate([string2array(x) for x in sub_pos["moment_info"]]),
                         "outcome": max(sub_pos["PTS"]),
                         "gravity": [val for val in sub_pos["mapped_gravity"]],
                         "offense_defense": json.loads(list(set(sub_pos["offense_defense"]))[0]),
                         "n_moments": len(sub_pos)
                         }
            file_name_pos = f"processed_data/possession_data/{max(sub_pos['game_id'])}_{pos}.pkl"
            pickle_data(file_name_pos, data_dump)
            print(f"successfully wrote pos {pos} for {data_file}")
