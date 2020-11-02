import matplotlib.pyplot as plt
import torch


def save_model(model, path: str) -> None:
    """

    :param model: model object to save
    :param path: path where to save
    :return:
    """
    torch.save(model, path)


def load_model(path: str) -> None:
    """

    :param path: path where to load
    :return:
    """
    torch.load(path)


def visualize_player_gravity(model_weights: torch.Tensor, player_index: int):
    """

    :param model_weights: assumes full tensor form (num_poss,player,x_dim,y_dim)
    :param player_index: integer indicating player
    :return:
    """
    grav_vals = torch.squeeze(torch.mean(model_weights[:, player_index, :, :], dim=0)).numpy()
    plt.imshow(grav_vals, interpolation="nearest")
    plt.show()
