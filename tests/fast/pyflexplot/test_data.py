# -*- coding: utf-8 -*-
# flake8: noqa
"""
Tests for module ``pyflexplot.data``.
"""
# Third-party
import numpy as np
import pytest  # type: ignore

# First-party
from pyflexplot.data import EnsembleCloud
from pyflexplot.data import threshold_agreement


class TestThrAgrmt2D:
    """Count number threshold exceedences along a 2D-array axis."""

    arr = np.array(
        [
            [3, 5, 6, 0, 1, 0, 3, 5],
            [6, 3, 4, 4, 4, 1, 6, 9],
            [2, 1, 4, 2, 6, 8, 4, 2],
            [9, 6, 7, 5, 7, 3, 1, 3],
        ]
    )

    thr = 5

    def test_default(self):
        sol = threshold_agreement(self.arr, self.thr)
        ref = 10
        assert sol == ref

    def test_ax0(self):
        sol = threshold_agreement(self.arr, self.thr, axis=0)
        ref = np.array([2, 1, 2, 0, 2, 1, 1, 1])
        assert (sol == ref).all()

    def test_ax1(self):
        sol = threshold_agreement(self.arr, self.thr, axis=1)
        ref = np.array([1, 3, 2, 4])
        assert (sol == ref).all()


class TestThrAgrmt3D:
    """Count number threshold exceedences along a 3D-array axis."""

    arr = np.array(
        [
            [[5, 2, 8], [0, 5, 0], [2, 1, 2], [9, 5, 2], [2, 0, 2]],
            [[5, 6, 7], [0, 0, 1], [2, 1, 9], [4, 5, 2], [0, 3, 9]],
            [[5, 6, 9], [1, 3, 8], [5, 3, 1], [3, 3, 3], [5, 3, 7]],
        ]
    )

    thr = 3

    def test_default(self):
        sol = threshold_agreement(self.arr, self.thr)
        ref = 19
        assert ref == sol

    def test_ax0(self):
        sol = threshold_agreement(self.arr, self.thr, axis=0)
        ref = np.array([[3, 2, 3], [0, 1, 1], [1, 0, 1], [2, 2, 0], [1, 0, 2]])
        assert (sol == ref).all()

    def test_ax1(self):
        sol = threshold_agreement(self.arr, self.thr, axis=1)
        ref = np.array([[2, 2, 1], [2, 2, 3], [3, 1, 3]])
        assert (sol == ref).all()

    def test_ax2(self):
        sol = threshold_agreement(self.arr, self.thr, axis=2)
        ref = np.array([[2, 1, 0, 2, 0], [3, 0, 1, 2, 1], [3, 1, 1, 0, 2]])
        assert (sol == ref).all()


N = np.nan
I = np.inf
J = -np.inf


class TestEnsembleCloud:
    """Test ensemble cloud fields like arrival and departure time."""

    arr = np.array(
        [
            [  # 0  1  2  3  4  5  6
                [1, 0, 0, 0, 0, 0, 1],  # mem: 0, time: 0, x: 0-6
                [2, 1, 0, 0, 0, 1, 1],  # mem: 0, time: 1, x: 0-6
                [1, 0, 0, 0, 0, 1, 2],  # mem: 0, time: 2, x: 0-6
                [0, 0, 1, 2, 3, 2, 2],  # mem: 0, time: 3, x: 0-6
                [0, 0, 0, 2, 3, 3, 1],  # mem: 0, time: 4, x: 0-6
            ],
            [  # 0  1  2  3  4  5  6
                [0, 0, 0, 0, 1, 3, 2],  # mem: 1, time: 0, x: 0-6
                [1, 0, 0, 0, 2, 2, 1],  # mem: 1, time: 1, x: 0-6
                [0, 0, 0, 1, 3, 3, 0],  # mem: 1, time: 2, x: 0-6
                [0, 0, 1, 2, 2, 1, 0],  # mem: 1, time: 3, x: 0-6
                [0, 1, 2, 3, 1, 0, 0],  # mem: 1, time: 4, x: 0-6
            ],
            [  # 0  1  2  3  4  5  6
                [0, 0, 0, 0, 0, 0, 0],  # mem: 2, time: 0, x: 0-6
                [0, 0, 0, 0, 2, 2, 2],  # mem: 2, time: 1, x: 0-6
                [0, 0, 0, 3, 3, 0, 0],  # mem: 2, time: 2, x: 0-6
                [1, 0, 2, 3, 2, 0, 0],  # mem: 2, time: 3, x: 0-6
                [0, 0, 0, 2, 1, 0, 0],  # mem: 2, time: 4, x: 0-6
            ],
            [  # 0  1  2  3  4  5  6
                [0, 0, 1, 3, 1, 0, 0],  # mem: 3, time: 0, x: 0-6
                [0, 0, 0, 2, 1, 1, 0],  # mem: 3, time: 1, x: 0-6
                [0, 1, 3, 3, 2, 0, 0],  # mem: 3, time: 2, x: 0-6
                [0, 0, 1, 2, 1, 0, 0],  # mem: 3, time: 3, x: 0-6
                [0, 1, 3, 1, 0, 0, 0],  # mem: 3, time: 4, x: 0-6
            ],
        ]
    ).astype(float)
    """Number of members above certain thresholds.

    >>> (arr > 0.5).sum(axis=0)

    #       0  1  2  3  4  5  6
    array([[1, 0, 1, 1, 2, 1, 2],  # time: 0
           [2, 1, 0, 1, 3, 4, 3],  # time: 1
           [1, 1, 1, 3, 3, 2, 1],  # time: 2
           [1, 0, 4, 4, 4, 2, 1],  # time: 3
           [0, 2, 2, 4, 3, 1, 1]]) # time: 4

    >>> (arr > 1.5).sum(axis=0)

    #       0  1  2  3  4  5  6
    array([[0, 0, 0, 1, 0, 1, 1],  # time: 0
           [1, 0, 0, 1, 2, 2, 1],  # time: 1
           [0, 0, 1, 2, 3, 1, 1],  # time: 2
           [0, 0, 1, 4, 3, 1, 1],  # time: 3
           [0, 0, 2, 3, 1, 1, 0]]) # time: 4

    >>> (arr > 2.5).sum(axis=0)

    #       0  1  2  3  4  5  6
    array([[0, 0, 0, 1, 0, 1, 0],  # time: 0
           [0, 0, 0, 0, 0, 0, 0],  # time: 1
           [0, 0, 1, 2, 2, 1, 0],  # time: 2
           [0, 0, 0, 1, 1, 0, 0],  # time: 3
           [0, 0, 1, 1, 1, 1, 0]]) # time: 4)

    >>> (arr > 3.5).sum(axis=0)

    #       0  1  2  3  4  5  6
    array([[0, 0, 0, 0, 0, 0, 0],  # time: 0
           [0, 0, 0, 0, 0, 0, 0],  # time: 1
           [0, 0, 0, 0, 0, 0, 0],  # time: 2
           [0, 0, 0, 0, 0, 0, 0],  # time: 3
           [0, 0, 0, 0, 0, 0, 0]]) # time: 4)

    """

    def create_cloud(self, thr, n_mem_min):
        return EnsembleCloud(
            arr=self.arr, time=np.arange(5), thr=thr, n_mem_min=n_mem_min,
        )

    # test_arrival_time
    # fmt: off
    @pytest.mark.parametrize(
        "thr, n_mem_min, sol",
        [
            (
                0.5,
                1,
                [  #  0  1  2  3  4  5  6
                    [ J, 1, J, J, J, J, J],  # time: 0
                    [ J, 0, J, J, J, J, J],  # time: 1
                    [ J,-1, J, J, J, J, J],  # time: 2
                    [ J,-2, J, J, J, J, J],  # time: 3
                    [ J,-3, J, J, J, J, J],  # time: 4
                ],
            ),
            (
                0.5,
                2,
                [  #  0  1  2  3  4  5  6
                    [ 1, 4, 3, 2, J, 1, J],  # time: 0
                    [ 0, 3, 2, 1, J, 0, J],  # time: 1
                    [-1, 2, 1, 0, J,-1, J],  # time: 2
                    [-2, 1, 0,-1, J,-2, J],  # time: 3
                    [-3, 0,-1,-2, J,-3, J],  # time: 4
                ],
            ),
            (
                0.5,
                3,
                [  #  0  1  2  3  4  5  6
                    [ N, N, 3, 2, 1, 1, 1],  # time: 0
                    [ N, N, 2, 1, 0, 0, 0],  # time: 1
                    [ N, N, 1, 0,-1,-1,-1],  # time: 2
                    [ N, N, 0,-1,-2,-2,-2],  # time: 3
                    [ N, N,-1,-2,-3,-3,-3],  # time: 4
                ],
            ),
            (
                0.5,
                4,
                [  #  0  1  2  3  4  5  6
                    [ N, N, 3, 3, 3, 1, N],  # time: 0
                    [ N, N, 2, 2, 2, 0, N],  # time: 1
                    [ N, N, 1, 1, 1,-1, N],  # time: 2
                    [ N, N, 0, 0, 0,-2, N],  # time: 3
                    [ N, N,-1,-1,-1,-3, N],  # time: 4
                ],
            ),
            (
                0.5,
                5,
                [  #  0  1  2  3  4  5  6
                    [ N, N, N, N, N, N, N],  # time: 0
                    [ N, N, N, N, N, N, N],  # time: 1
                    [ N, N, N, N, N, N, N],  # time: 2
                    [ N, N, N, N, N, N, N],  # time: 3
                    [ N, N, N, N, N, N, N],  # time: 4
                ],
            ),
            (
                1.5,
                1,
                [  #  0  1  2  3  4  5  6
                    [ 1, N, 2, J, 1, J, J],  # time: 0
                    [ 0, N, 1, J, 0, J, J],  # time: 1
                    [-1, N, 0, J,-1, J, J],  # time: 2
                    [-2, N,-1, J,-2, J, J],  # time: 3
                    [-3, N,-2, J,-3, J, J],  # time: 4
                ],
            ),
            (
                1.5,
                2,
                [  #  0  1  2  3  4  5  6
                    [ N, N, 4, 2, 1, 1, N],  # time: 0
                    [ N, N, 3, 1, 0, 0, N],  # time: 1
                    [ N, N, 2, 0,-1,-1, N],  # time: 2
                    [ N, N, 1,-1,-2,-2, N],  # time: 3
                    [ N, N, 0,-2,-3,-3, N],  # time: 4
                ],
            ),
            (
                1.5,
                3,
                [  #  0  1  2  3  4  5  6
                    [ N, N, N, 3, 2, N, N],  # time: 0
                    [ N, N, N, 2, 1, N, N],  # time: 1
                    [ N, N, N, 1, 0, N, N],  # time: 2
                    [ N, N, N, 0,-1, N, N],  # time: 3
                    [ N, N, N,-1,-2, N, N],  # time: 4
                ],
            ),
            (
                1.5,
                4,
                [  #  0  1  2  3  4  5  6
                    [ N, N, N, 3, N, N, N],  # time: 0
                    [ N, N, N, 2, N, N, N],  # time: 1
                    [ N, N, N, 1, N, N, N],  # time: 2
                    [ N, N, N, 0, N, N, N],  # time: 3
                    [ N, N, N,-1, N, N, N],  # time: 4
                ],
            ),
            (
                2.5,
                1,
                [  #  0  1  2  3  4  5  6
                    [ N, N, 2, J, 2, J, N],  # time: 0
                    [ N, N, 1, J, 1, J, N],  # time: 1
                    [ N, N, 0, J, 0, J, N],  # time: 2
                    [ N, N,-1, J,-1, J, N],  # time: 3
                    [ N, N,-2, J,-2, J, N],  # time: 4
                ],
            ),
            (
                2.5,
                2,
                [  #  0  1  2  3  4  5  6
                    [ N, N, N, 2, 2, N, N],  # time: 0
                    [ N, N, N, 1, 1, N, N],  # time: 1
                    [ N, N, N, 0, 0, N, N],  # time: 2
                    [ N, N, N,-1,-1, N, N],  # time: 3
                    [ N, N, N,-2,-2, N, N],  # time: 4
                ],
            ),
            (
                3.5,
                1,
                [  #  0  1  2  3  4  5  6
                    [ N, N, N, N, N, N, N],  # time: 0
                    [ N, N, N, N, N, N, N],  # time: 1
                    [ N, N, N, N, N, N, N],  # time: 2
                    [ N, N, N, N, N, N, N],  # time: 3
                    [ N, N, N, N, N, N, N],  # time: 4
                ],
            ),
        ],
    )
    # fmt: on
    def test_arrival_time(self, thr, n_mem_min, sol):
        res = self.create_cloud(thr, n_mem_min).arrival_time()
        np.testing.assert_array_equal(res, sol)

    # test_departure_time
    # fmt: off
    @pytest.mark.parametrize(
        "thr, n_mem_min, sol",
        [
            (
                0.5,
                1,
                [  #  0  1  2  3  4  5  6
                    [ 4, I, I, I, I, I, I],  # time: 0
                    [ 3, I, I, I, I, I, I],  # time: 1
                    [ 2, I, I, I, I, I, I],  # time: 2
                    [ 1, I, I, I, I, I, I],  # time: 3
                    [ 0, I, I, I, I, I, I],  # time: 4
                ],
            ),
            (
                0.5,
                2,
                [  #  0  1  2  3  4  5  6
                    [ 2, I, I, I, I, 4, 2],  # time: 0
                    [ 1, I, I, I, I, 3, 1],  # time: 1
                    [ 0, I, I, I, I, 2, 0],  # time: 2
                    [-1, I, I, I, I, 1,-1],  # time: 3
                    [-2, I, I, I, I, 0,-2],  # time: 4
                ],
            ),
            (
                0.5,
                3,
                [  #  0  1  2  3  4  5  6
                    [ N, N, 4, I, I, 2, 2],  # time: 0
                    [ N, N, 3, I, I, 1, 1],  # time: 1
                    [ N, N, 2, I, I, 0, 0],  # time: 2
                    [ N, N, 1, I, I,-1,-1],  # time: 3
                    [ N, N, 0, I, I,-2,-2],  # time: 4
                ],
            ),
            (
                0.5,
                4,
                [  #  0  1  2  3  4  5  6
                    [ N, N, 4, I, 4, 2, N],  # time: 0
                    [ N, N, 3, I, 3, 1, N],  # time: 1
                    [ N, N, 2, I, 2, 0, N],  # time: 2
                    [ N, N, 1, I, 1,-1, N],  # time: 3
                    [ N, N, 0, I, 0,-2, N],  # time: 4
                ],
            ),
            (
                0.5,
                5,
                [  #  0  1  2  3  4  5  6
                    [ N, N, N, N, N, N, N],  # time: 0
                    [ N, N, N, N, N, N, N],  # time: 1
                    [ N, N, N, N, N, N, N],  # time: 2
                    [ N, N, N, N, N, N, N],  # time: 3
                    [ N, N, N, N, N, N, N],  # time: 4
                ],
            ),
            (
                1.5,
                1,
                [  #  0  1  2  3  4  5  6
                    [ 2, N, I, I, I, I, 4],  # time: 0
                    [ 1, N, I, I, I, I, 3],  # time: 1
                    [ 0, N, I, I, I, I, 2],  # time: 2
                    [-1, N, I, I, I, I, 1],  # time: 3
                    [-2, N, I, I, I, I, 0],  # time: 4
                ],
            ),
            (
                1.5,
                2,
                [  # 0  1  2  3  4  5  6
                    [N, N, I, I, 4, 2, N],  # time: 0
                    [N, N, I, I, 3, 1, N],  # time: 1
                    [N, N, I, I, 2, 0, N],  # time: 2
                    [N, N, I, I, 1,-1, N],  # time: 3
                    [N, N, I, I, 0,-2, N],  # time: 4
                ],
            ),
            (
                1.5,
                3,
                [  # 0  1  2  3  4  5  6
                    [N, N, N, I, 4, N, N],  # time: 0
                    [N, N, N, I, 3, N, N],  # time: 1
                    [N, N, N, I, 2, N, N],  # time: 2
                    [N, N, N, I, 1, N, N],  # time: 3
                    [N, N, N, I, 0, N, N],  # time: 4
                ],
            ),
            (
                1.5,
                4,
                [  # 0  1  2  3  4  5  6
                    [N, N, N, 4, N, N, N],  # time: 0
                    [N, N, N, 3, N, N, N],  # time: 1
                    [N, N, N, 2, N, N, N],  # time: 2
                    [N, N, N, 1, N, N, N],  # time: 3
                    [N, N, N, 0, N, N, N],  # time: 4
                ],
            ),
            (
                2.5,
                1,
                [  # 0  1  2  3  4  5  6
                    [N, N, I, I, I, I, N],  # time: 0
                    [N, N, I, I, I, I, N],  # time: 1
                    [N, N, I, I, I, I, N],  # time: 2
                    [N, N, I, I, I, I, N],  # time: 3
                    [N, N, I, I, I, I, N],  # time: 4
                ],
            ),
            (
                2.5,
                2,
                [  # 0  1  2  3  4  5  6
                    [N, N, N, 3, 3, N, N],  # time: 0
                    [N, N, N, 2, 2, N, N],  # time: 1
                    [N, N, N, 1, 1, N, N],  # time: 2
                    [N, N, N, 0, 0, N, N],  # time: 3
                    [N, N, N,-1,-1, N, N],  # time: 4
                ],
            ),
            (
                3.5,
                1,
                [  # 0  1  2  3  4  5  6
                    [N, N, N, N, N, N, N],  # time: 0
                    [N, N, N, N, N, N, N],  # time: 1
                    [N, N, N, N, N, N, N],  # time: 2
                    [N, N, N, N, N, N, N],  # time: 3
                    [N, N, N, N, N, N, N],  # time: 4
                ],
            ),
        ],
    )
    # fmt: on
    def test_departure_time(self, thr, n_mem_min, sol):
        res = self.create_cloud(thr, n_mem_min).departure_time()
        np.testing.assert_array_equal(res, sol)
