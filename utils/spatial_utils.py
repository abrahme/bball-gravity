import numpy as np
from scipy.spatial.distance import cdist


def create_rotation_matrix(a: float) -> np.ndarray:
    """
    create ccw rotation matrix in 2D
    :param a: angle of rotation
    :return:
    """

    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def apply_rotation_matrix(rot_mat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """

    :param rot_mat: 2 x 2 matrix
    :param x: n x 2 (x,y) locations
    :return: n x 2 rotated (x,y) locations
    """
    return np.einsum("ij,jk->ik", rot_mat, x.T).T


def angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    finds elementwise angle between
    :param a: array
    :param b: array
    :return: n x m array of angles between a_i and b_j
    """

    norm_a = a / np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.arccos(np.einsum("ij,jk->ik", norm_a, norm_b.T))


def squared_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """

    :param a: n x 2 array of (x,y) coordinates
    :param b: m x 2 array of (x,y) coordinates
    :return: n x m array of squared distance between a_i and b_j
    """
    return np.square(cdist(a, b, 'euclidean'))


def center(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    centers locations a around b
    :param a: n x 2 array of (x,y) coordinates
    :param b: either 1 x 2 array of (x,y) coordinates
    :return: n x 2 array
    """

    return a - b


