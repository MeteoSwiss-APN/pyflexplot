# -*- coding: utf-8 -*-
"""
Tests for module ``pyflexplot.data``.
"""
# Third-party
import numpy as np
import pytest  # type: ignore

# First-party
from pyflexplot.data import cloud_arrival_time
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

    def test_eq_ok(self):
        sol = threshold_agreement(self.arr, self.thr, thr_eq_ok=True)
        ref = 13
        assert sol == ref

    def test_ax0(self):
        sol = threshold_agreement(self.arr, self.thr, axis=0)
        ref = np.array([2, 1, 2, 0, 2, 1, 1, 1])
        assert (sol == ref).all()

    def test_ax0_eq_ok(self):
        sol = threshold_agreement(self.arr, self.thr, axis=0, thr_eq_ok=True)
        ref = np.array([2, 2, 2, 1, 2, 1, 1, 2])
        assert (sol == ref).all()

    def test_ax1(self):
        sol = threshold_agreement(self.arr, self.thr, axis=1)
        ref = np.array([1, 3, 2, 4])
        assert (sol == ref).all()

    def test_ax1_eq_ok(self):
        sol = threshold_agreement(self.arr, self.thr, axis=1, thr_eq_ok=True)
        ref = np.array([3, 3, 2, 5])
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


class TestCloudArrivalTime:
    """Compute the time until enough members forecast a cloud."""

    arr = np.array(
        [
            [
                [1, 0, 0, 0, 0, 0, 1],  # mem: 0, time: 0, x: 0-6
                [2, 1, 0, 0, 0, 1, 1],  # mem: 0, time: 1, x: 0-6
                [1, 0, 0, 0, 0, 1, 2],  # mem: 0, time: 2, x: 0-6
                [0, 0, 1, 2, 3, 2, 2],  # mem: 0, time: 3, x: 0-6
                [0, 0, 0, 2, 3, 3, 1],  # mem: 0, time: 4, x: 0-6
            ],
            [
                [0, 0, 0, 0, 1, 3, 2],  # mem: 1, time: 0, x: 0-6
                [1, 0, 0, 0, 2, 2, 1],  # mem: 1, time: 1, x: 0-6
                [0, 0, 0, 1, 3, 3, 0],  # mem: 1, time: 2, x: 0-6
                [0, 0, 1, 2, 2, 1, 0],  # mem: 1, time: 3, x: 0-6
                [0, 1, 2, 3, 1, 0, 0],  # mem: 1, time: 4, x: 0-6
            ],
            [
                [0, 0, 0, 0, 0, 0, 0],  # mem: 2, time: 0, x: 0-6
                [0, 0, 0, 0, 2, 2, 2],  # mem: 2, time: 1, x: 0-6
                [0, 0, 0, 3, 3, 0, 0],  # mem: 2, time: 2, x: 0-6
                [1, 0, 2, 3, 2, 0, 0],  # mem: 2, time: 3, x: 0-6
                [0, 0, 0, 2, 1, 0, 0],  # mem: 2, time: 4, x: 0-6
            ],
            [
                [0, 0, 1, 3, 1, 0, 0],  # mem: 3, time: 0, x: 0-6
                [0, 0, 0, 2, 1, 1, 0],  # mem: 3, time: 1, x: 0-6
                [0, 1, 3, 3, 2, 0, 0],  # mem: 3, time: 2, x: 0-6
                [0, 0, 1, 2, 1, 0, 0],  # mem: 3, time: 3, x: 0-6
                [0, 1, 3, 1, 0, 0, 0],  # mem: 3, time: 4, x: 0-6
            ],
        ]
    ).astype(float)

    #        >>> (arr > 0.5).sum(axis=0)
    #
    #        array([[1, 0, 1, 1, 2, 1, 2],
    #               [2, 1, 0, 1, 3, 4, 3],
    #               [1, 1, 1, 3, 3, 2, 1],
    #               [1, 0, 4, 4, 4, 2, 1],
    #               [0, 2, 2, 4, 3, 1, 1]])
    #
    #        >>> (arr > 1.5).sum(axis=0)
    #
    #        array([[0, 0, 0, 1, 0, 1, 1],
    #               [1, 0, 0, 1, 2, 2, 1],
    #               [0, 0, 1, 2, 3, 1, 1],
    #               [0, 0, 1, 4, 3, 1, 1],
    #               [0, 0, 2, 3, 1, 1, 0]]
    #
    #        >>> (arr > 2.5).sum(axis=0)
    #
    #        array([[0, 0, 0, 1, 0, 1, 0],
    #               [0, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 1, 2, 2, 1, 0],
    #               [0, 0, 0, 1, 1, 0, 0],
    #               [0, 0, 1, 1, 1, 1, 0]])
    #
    #        >>> (arr > 3.5).sum(axis=0)
    #
    #        array([[0, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 0]])

    kwargs = {"mem_axis": 0, "time_axis": 1}

    @pytest.mark.parametrize(
        "thr, n_mem_min, sol",
        [
            (
                0.5,
                1,
                [
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [N, 0, 0, 0, 0, 0, 0],
                ],
            ),
            (
                0.5,
                2,
                [
                    [1, 4, 3, 2, 0, 1, 0],
                    [0, 3, 2, 1, 0, 0, 0],
                    [N, 2, 1, 0, 0, 0, N],
                    [N, 1, 0, 0, 0, 0, N],
                    [N, 0, 0, 0, 0, N, N],
                ],
            ),
            (
                0.5,
                3,
                [
                    [N, N, 3, 2, 1, 1, 1],
                    [N, N, 2, 1, 0, 0, 0],
                    [N, N, 1, 0, 0, N, N],
                    [N, N, 0, 0, 0, N, N],
                    [N, N, N, 0, 0, N, N],
                ],
            ),
            (
                0.5,
                4,
                [
                    [N, N, 3, 3, 3, 1, N],
                    [N, N, 2, 2, 2, 0, N],
                    [N, N, 1, 1, 1, N, N],
                    [N, N, 0, 0, 0, N, N],
                    [N, N, N, 0, N, N, N],
                ],
            ),
            (
                0.5,
                5,
                [
                    [N, N, N, N, N, N, N],
                    [N, N, N, N, N, N, N],
                    [N, N, N, N, N, N, N],
                    [N, N, N, N, N, N, N],
                    [N, N, N, N, N, N, N],
                ],
            ),
            (
                1.5,
                1,
                [
                    [1, N, 2, 0, 1, 0, 0],
                    [0, N, 1, 0, 0, 0, 0],
                    [N, N, 0, 0, 0, 0, 0],
                    [N, N, 0, 0, 0, 0, 0],
                    [N, N, 0, 0, 0, 0, N],
                ],
            ),
            (
                1.5,
                2,
                [
                    [N, N, 4, 2, 1, 1, N],
                    [N, N, 3, 1, 0, 0, N],
                    [N, N, 2, 0, 0, N, N],
                    [N, N, 1, 0, 0, N, N],
                    [N, N, 0, 0, N, N, N],
                ],
            ),
            (
                1.5,
                3,
                [
                    [N, N, N, 3, 2, N, N],
                    [N, N, N, 2, 1, N, N],
                    [N, N, N, 1, 0, N, N],
                    [N, N, N, 0, 0, N, N],
                    [N, N, N, 0, N, N, N],
                ],
            ),
            (
                1.5,
                4,
                [
                    [N, N, N, 3, N, N, N],
                    [N, N, N, 2, N, N, N],
                    [N, N, N, 1, N, N, N],
                    [N, N, N, 0, N, N, N],
                    [N, N, N, N, N, N, N],
                ],
            ),
            (
                2.5,
                1,
                [
                    [N, N, 2, 0, 2, 0, N],
                    [N, N, 1, 1, 1, 1, N],
                    [N, N, 0, 0, 0, 0, N],
                    [N, N, 1, 0, 0, 1, N],
                    [N, N, 0, 0, 0, 0, N],
                ],
            ),
            (
                2.5,
                2,
                [
                    [N, N, N, 2, 2, N, N],
                    [N, N, N, 1, 1, N, N],
                    [N, N, N, 0, 0, N, N],
                    [N, N, N, N, N, N, N],
                    [N, N, N, N, N, N, N],
                ],
            ),
            (
                3.5,
                1,
                [
                    [N, N, N, N, N, N, N],
                    [N, N, N, N, N, N, N],
                    [N, N, N, N, N, N, N],
                    [N, N, N, N, N, N, N],
                    [N, N, N, N, N, N, N],
                ],
            ),
        ],
    )
    def test(self, thr, n_mem_min, sol):
        time = np.arange(5)
        res = cloud_arrival_time(
            self.arr, time=time, thr=thr, n_mem_min=n_mem_min, **self.kwargs
        )
        np.testing.assert_array_equal(res, sol)
