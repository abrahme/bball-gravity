import glob
import torch
import numpy as np
from typing import Dict
from torch.utils.data import Dataset
from .data_utils import unpickle_data


class GravityDataSet(Dataset):
    def __init__(self, T: int, P: int, Q: int, R: int, max_x: int, max_y: int, data_dir_globbed: str,
                 player_encoding_map: Dict, transform=None) -> None:
        """
        :param transform: torchvision transform function
        :param player_encoding_map:
        :param data_dir_globbed: ex "data/*.json"
        :param T: max number of time steps
        :param max_x: max value location x can take
        :param max_y: max value location y can take
        :param P: number of players
        :param Q: x resolution
        :param R: y resolution
        """
        self.transform = transform
        self.max_y = max_y
        self.max_x = max_x
        self.player_encoding_map = player_encoding_map
        self.data_dir_globbed = data_dir_globbed
        self.T = T
        self.Q = Q
        self.R = R
        self.P = P
        self.data_files = glob.glob(self.data_dir_globbed)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx) -> Dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.data_files[idx]
        data = unpickle_data(file_name)
        return self.data_to_array(data)

    def location_to_resolution(self, frame: np.ndarray, max_x: int, max_y: int) -> np.ndarray:
        """
        maps location coordinates to appropriate indexes 
        :param max_y: max possible y coord
        :param max_x: max possible x coord
        :param frame: n x 2 array of locations
        :return: n x 2 array of mapped locations
        """

        div_x = max_x / self.Q
        div_y = max_y / self.R
        new_frame = np.zeros_like(frame)
        new_frame[:, 0] = np.floor(frame[:, 0] / div_x)
        new_frame[:, 1] = np.floor(frame[:, 1] / div_y)
        return new_frame

    def data_to_array(self, data) -> Dict:
        """

        :param data: dict with keys
        :return:
        """
        gravity_tensor = np.zeros((self.T, 2 * self.P, self.Q, self.R))
        all_moments = data["pos_moments"]
        all_gravity = data["gravity"]
        num_timesteps = data["n_moments"]
        target = data["outcome"]
        offense_id = data["offense_defense"]["offense"]

        for t in range(num_timesteps):
            moment = all_moments[t, :, :]
            resolved_location = self.location_to_resolution(moment[1:, 2:4], self.max_x,
                                                            self.max_y).astype(int)  ### don't want ball
            resolved_location[:, 0] = np.minimum(resolved_location[:, 0], self.Q - 1)
            resolved_location[:, 1] = np.minimum(resolved_location[:, 1], self.R - 1)
            gravity = all_gravity[t]
            player_indices = []
            grav_vals = []
            for player_id in moment[1:, 1].astype(int):
                if player_id in offense_id:
                    player_index = self.player_encoding_map[str(player_id)]
                else:
                    player_index = self.player_encoding_map[str(player_id)] + self.P
                grav_vals.append(gravity[player_id])
                player_indices.append(player_index)
            gravity_tensor[t, player_indices, resolved_location[:, 0], resolved_location[:, 1]] = np.array(grav_vals)

        sample = (gravity_tensor, np.array([float(target)]))
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """
    convert numpy array to tensor object
    """

    def __call__(self, sample):
        X, y = sample
        return torch.from_numpy(X), torch.from_numpy(y)
