import os
from utils.data_utils import *
from gravity import *

"""
Input data should be in the /data folder.
Data will be outputted to /processed_data folder
"""

if __name__ == "__main__":
    for data_file in [os.path.join("data", f) for f in os.listdir("data")]:
        raw_data = read_data(data_file)
        _, file_name = os.path.split(data_file)
        output_data = raw_data.copy()
        for event in output_data['events']:
            valid_moments = []
            gravities = []
            for moment in event['moments']:
                if check_valid(np.array(moment[-1])):
                    valid_moments.append(moment)
                    gravity = calculate_gravity(np.array(moment[-1]))
                    gravities.append(gravity)
            event['moments'] = valid_moments
            event['gravities'] = np.array(gravities).tolist()

        write_data(os.path.join("processed_data", file_name), output_data)
