from utils.spatial_utils import *
from scipy.optimize import lsq_linear


def equilibrium_matrix(frame: np.ndarray, basket: np.ndarray = np.array([[4.0, 25.0], [90.0, 25.0]])) -> np.ndarray:
    """
    sets up equilibrium equation for x and y
    :param frame: (11 x 5) array of one frame already corrected for location,
    assumes first entry is the ball
    :param basket: location of basket (1 x 2) array
    :return:
    """
    A = np.zeros((26, 13))

    # append basket to frame
    locations = np.vstack((frame[:, 2:4], basket))
    # calculate distance matrix
    distance_weight = squared_distance(locations, locations)

    # begin jackknife
    for i in range(13):
        jackknifed_position = center(locations, locations[i, :])
        # jackknifed position
        if i != 0:  # ball
            # rotate to align ball with + x-axis
            rotation_angle = np.arctan2(jackknifed_position[0, 1], jackknifed_position[0, 0])
            if rotation_angle < 0:
                rotation_angle *= -1
                rotation_angle += np.pi / 2
            rot_mat = create_rotation_matrix(rotation_angle)
        else:
            rot_mat = np.eye(2)
        rotated_jackknife = apply_rotation_matrix(rot_mat, jackknifed_position)
        norms = np.linalg.norm(rotated_jackknife, axis=1)
        cos_theta = rotated_jackknife[:, 0] / norms
        sin_theta = rotated_jackknife[:, 1] / norms
        x_row = 2 * i
        y_row = 2 * i + 1
        for j in range(13):
            if i != j:
                A[x_row, j] = (1 / distance_weight[i, j]) * cos_theta[j]
                A[y_row, j] = (1 / distance_weight[i, j]) * sin_theta[j]

    return np.nan_to_num(A)


def calculate_gravity(frame: np.ndarray) -> np.ndarray:
    """
    solves the equilibrium equation using least squares with constraints
    :param frame: (11 x 5) array
    :return:
    """
    A = equilibrium_matrix(frame)
    lb = -np.inf * np.ones((13,))
    lb[0] = 1  # sets the ball to be 1
    ub = np.inf * np.ones((13,))
    ub[0] = 1.01
    result = lsq_linear(A, np.zeros((26,)), bounds=(lb, ub))
    return result.x
