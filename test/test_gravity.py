import unittest
from numpy.testing import assert_array_almost_equal
from utils.spatial_utils import *


class GravityTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.locations = np.array([[0, 0], [5, 5], [-5, 5], [0, -5]])
        self.rotation_angle = np.pi / 2
        self.neg_y_axis = np.array([[0, -1]])

    def test_distance(self) -> None:
        assert_array_almost_equal(squared_distance(self.locations, self.locations), np.array([[0., 50., 50., 25.],
                                                                                              [50., 0., 100., 125.],
                                                                                              [50., 100., 0., 125.],
                                                                                              [25., 125., 125., 0.]]))

    def test_rotation_matrix(self) -> None:
        assert_array_almost_equal(create_rotation_matrix(self.rotation_angle), np.array([[0, -1], [1, 0]]))

    def test_rotation(self) -> None:
        rot_mat = create_rotation_matrix(np.pi / 2)
        assert_array_almost_equal(apply_rotation_matrix(rot_mat, self.locations),
                                  np.array([[0, 0],
                                            [-5, 5],
                                            [-5, -5],
                                            [5, 0]]))

    def test_alignment(self) -> None:
        ### wish to align ball (5,5) with the negative y axis
        ### calculate angle of rotation
        angles = angle(self.locations, self.neg_y_axis)
        angles[self.locations[:, 0] >= 0] *= -1
        angles[self.locations[:, 0] >= 0] += np.pi * 2
        ball_index = 1

        rotation_angle = angles[ball_index, 0]
        rot_mat = create_rotation_matrix(rotation_angle)
        rot_vals = apply_rotation_matrix(rot_mat, self.locations)
        assert_array_almost_equal(rot_vals[ball_index, :], np.array([0, -np.sqrt(50)]))


