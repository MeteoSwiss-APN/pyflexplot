#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for module ``pyflexplot.data``."""
# Third-party
import numpy as np
import pytest

# First-party
from pyflexplot.data import threshold_agreement


class TestThrAgrmt_2D:
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
        sol = threshold_agreement(self.arr, self.thr, eq_ok=True)
        ref = 13
        assert sol == ref

    def test_ax0(self):
        sol = threshold_agreement(self.arr, self.thr, axis=0)
        ref = np.array([2, 1, 2, 0, 2, 1, 1, 1])
        assert (sol == ref).all()

    def test_ax0_eq_ok(self):
        sol = threshold_agreement(self.arr, self.thr, axis=0, eq_ok=True)
        ref = np.array([2, 2, 2, 1, 2, 1, 1, 2])
        assert (sol == ref).all()

    def test_ax1(self):
        sol = threshold_agreement(self.arr, self.thr, axis=1)
        ref = np.array([1, 3, 2, 4])
        assert (sol == ref).all()

    def test_ax1_eq_ok(self):
        sol = threshold_agreement(self.arr, self.thr, axis=1, eq_ok=True)
        ref = np.array([3, 3, 2, 5])
        assert (sol == ref).all()


class TestThrAgrmt_3D:
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
