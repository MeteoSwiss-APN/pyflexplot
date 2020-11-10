"""Tests for class ``pyflexplot.data.EnsembleCloud``."""
# flake8: noqa  # complains about code in "fmt: off/on" blocks
# Standard library
from dataclasses import dataclass

# Third-party
import numpy as np
import pytest  # noqa: F401  # imported but unused

# First-party
from pyflexplot.data import EnsembleCloud

# Shorthands for special values
N = np.nan
I = np.inf  # noqa: E741  # ambiguous variable name


def a(*args, **kwargs):
    """Shorthand function to create array."""
    return np.array(*args, **kwargs)


@dataclass
class Config:
    thr: float
    mem_min: int
    sol_norm: np.ndarray


N_MEM = 4
D_TIME = 2
ARR = np.array(
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
# Number of members above certain thresholds:
#
# >>> (arr > 0.5).sum(axis=0)
#
# #       0  1  2  3  4  5  6
# array([[1, 0, 1, 1, 2, 1, 2],  # time: 0
#        [2, 1, 0, 1, 3, 4, 3],  # time: 1
#        [1, 1, 1, 3, 3, 2, 1],  # time: 2
#        [1, 0, 4, 4, 4, 2, 1],  # time: 3
#        [0, 2, 2, 4, 3, 1, 1]]) # time: 4
#
# >>> (arr > 1.5).sum(axis=0)
#
# #       0  1  2  3  4  5  6
# array([[0, 0, 0, 1, 0, 1, 1],  # time: 0
#        [1, 0, 0, 1, 2, 2, 1],  # time: 1
#        [0, 0, 1, 2, 3, 1, 1],  # time: 2
#        [0, 0, 1, 4, 3, 1, 1],  # time: 3
#        [0, 0, 2, 3, 1, 1, 0]]) # time: 4
#
# >>> (arr > 2.5).sum(axis=0)
#
# #       0  1  2  3  4  5  6
# array([[0, 0, 0, 1, 0, 1, 0],  # time: 0
#        [0, 0, 0, 0, 0, 0, 0],  # time: 1
#        [0, 0, 1, 2, 2, 1, 0],  # time: 2
#        [0, 0, 0, 1, 1, 0, 0],  # time: 3
#        [0, 0, 1, 1, 1, 1, 0]]) # time: 4)
#
# >>> (arr > 3.5).sum(axis=0)
#
# #       0  1  2  3  4  5  6
# array([[0, 0, 0, 0, 0, 0, 0],  # time: 0
#        [0, 0, 0, 0, 0, 0, 0],  # time: 1
#        [0, 0, 0, 0, 0, 0, 0],  # time: 2
#        [0, 0, 0, 0, 0, 0, 0],  # time: 3
#        [0, 0, 0, 0, 0, 0, 0]]) # time: 4)


# test_arrival_time
# fmt: off
@pytest.mark.parametrize(
    "config",
    [
        Config(
            thr=0.5,
            mem_min=1,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [-I, 1,-I,-I,-I,-I,-I],  # time: 0
                    [-I, 0,-I,-I,-I,-I,-I],  # time: 1
                    [-I,-1,-I,-I,-I,-I,-I],  # time: 2
                    [-I,-2,-I,-I,-I,-I,-I],  # time: 3
                    [-I,-3,-I,-I,-I,-I,-I],  # time: 4
                ],
            ),
        ),
        Config(
            thr=0.5,
            mem_min=2,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ 1, 4, 3, 2,-I, 1,-I],  # time: 0
                    [ 0, 3, 2, 1,-I, 0,-I],  # time: 1
                    [-1, 2, 1, 0,-I,-1,-I],  # time: 2
                    [-2, 1, 0,-1,-I,-2,-I],  # time: 3
                    [-3, 0,-1,-2,-I,-3,-I],  # time: 4
                ],
            ),
        ),
        Config(
            thr=0.5,
            mem_min=3,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ N, N, 3, 2, 1, 1, 1],  # time: 0
                    [ N, N, 2, 1, 0, 0, 0],  # time: 1
                    [ N, N, 1, 0,-1,-1,-1],  # time: 2
                    [ N, N, 0,-1,-2,-2,-2],  # time: 3
                    [ N, N,-1,-2,-3,-3,-3],  # time: 4
                ],
            ),
        ),
        Config(
            thr=0.5,
            mem_min=4,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ N, N, 3, 3, 3, 1, N],  # time: 0
                    [ N, N, 2, 2, 2, 0, N],  # time: 1
                    [ N, N, 1, 1, 1,-1, N],  # time: 2
                    [ N, N, 0, 0, 0,-2, N],  # time: 3
                    [ N, N,-1,-1,-1,-3, N],  # time: 4
                ],
            ),
        ),
        Config(
            thr=0.5,
            mem_min=5,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ N, N, N, N, N, N, N],  # time: 0
                    [ N, N, N, N, N, N, N],  # time: 1
                    [ N, N, N, N, N, N, N],  # time: 2
                    [ N, N, N, N, N, N, N],  # time: 3
                    [ N, N, N, N, N, N, N],  # time: 4
                ]
            ),
        ),
        Config(
            thr=1.5,
            mem_min=1,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ 1, N, 2,-I, 1,-I,-I],  # time: 0
                    [ 0, N, 1,-I, 0,-I,-I],  # time: 1
                    [-1, N, 0,-I,-1,-I,-I],  # time: 2
                    [-2, N,-1,-I,-2,-I,-I],  # time: 3
                    [-3, N,-2,-I,-3,-I,-I],  # time: 4
                ],
            ),
        ),
        Config(
            thr=1.5,
            mem_min=2,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ N, N, 4, 2, 1, 1, N],  # time: 0
                    [ N, N, 3, 1, 0, 0, N],  # time: 1
                    [ N, N, 2, 0,-1,-1, N],  # time: 2
                    [ N, N, 1,-1,-2,-2, N],  # time: 3
                    [ N, N, 0,-2,-3,-3, N],  # time: 4
                ],
            ),
        ),
        Config(
            thr=1.5,
            mem_min=3,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ N, N, N, 3, 2, N, N],  # time: 0
                    [ N, N, N, 2, 1, N, N],  # time: 1
                    [ N, N, N, 1, 0, N, N],  # time: 2
                    [ N, N, N, 0,-1, N, N],  # time: 3
                    [ N, N, N,-1,-2, N, N],  # time: 4
                ],
            ),
        ),
        Config(
            thr=1.5,
            mem_min=4,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ N, N, N, 3, N, N, N],  # time: 0
                    [ N, N, N, 2, N, N, N],  # time: 1
                    [ N, N, N, 1, N, N, N],  # time: 2
                    [ N, N, N, 0, N, N, N],  # time: 3
                    [ N, N, N,-1, N, N, N],  # time: 4
                ],
            ),
        ),
        Config(
            thr=2.5,
            mem_min=1,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ N, N, 2,-I, 2,-I, N],  # time: 0
                    [ N, N, 1,-I, 1,-I, N],  # time: 1
                    [ N, N, 0,-I, 0,-I, N],  # time: 2
                    [ N, N,-1,-I,-1,-I, N],  # time: 3
                    [ N, N,-2,-I,-2,-I, N],  # time: 4
                ],
            ),
        ),
        Config(
            thr=2.5,
            mem_min=2,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ N, N, N, 2, 2, N, N],  # time: 0
                    [ N, N, N, 1, 1, N, N],  # time: 1
                    [ N, N, N, 0, 0, N, N],  # time: 2
                    [ N, N, N,-1,-1, N, N],  # time: 3
                    [ N, N, N,-2,-2, N, N],  # time: 4
                ],
            ),
        ),
        Config(
            thr=3.5,
            mem_min=1,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ N, N, N, N, N, N, N],  # time: 0
                    [ N, N, N, N, N, N, N],  # time: 1
                    [ N, N, N, N, N, N, N],  # time: 2
                    [ N, N, N, N, N, N, N],  # time: 3
                    [ N, N, N, N, N, N, N],  # time: 4
                ],
            ),
        ),
    ],
)
# fmt: on
def test_arrival_time(config):
    res = EnsembleCloud(
        arr=ARR, time=np.arange(5) * D_TIME, thr=config.thr  # scale time step
    ).arrival_time(config.mem_min)
    sol = config.sol_norm * D_TIME  # scale time step
    np.testing.assert_array_equal(res, sol)


# test_departure_time
# fmt: off
@pytest.mark.parametrize(
    "config",
    [
        Config(
            thr=0.5,
            mem_min=1,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ 4, I, I, I, I, I, I],  # time: 0
                    [ 3, I, I, I, I, I, I],  # time: 1
                    [ 2, I, I, I, I, I, I],  # time: 2
                    [ 1, I, I, I, I, I, I],  # time: 3
                    [ 0, I, I, I, I, I, I],  # time: 4
                ],
            ),
        ),
        Config(
            thr=0.5,
            mem_min=2,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ 2, I, I, I, I, 4, 2],  # time: 0
                    [ 1, I, I, I, I, 3, 1],  # time: 1
                    [ 0, I, I, I, I, 2, 0],  # time: 2
                    [-1, I, I, I, I, 1,-1],  # time: 3
                    [-2, I, I, I, I, 0,-2],  # time: 4
                ],
            ),
        ),
        Config(
            thr=0.5,
            mem_min=3,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ N, N, 4, I, I, 2, 2],  # time: 0
                    [ N, N, 3, I, I, 1, 1],  # time: 1
                    [ N, N, 2, I, I, 0, 0],  # time: 2
                    [ N, N, 1, I, I,-1,-1],  # time: 3
                    [ N, N, 0, I, I,-2,-2],  # time: 4
                ],
            ),
        ),
        Config(
            thr=0.5,
            mem_min=4,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ N, N, 4, I, 4, 2, N],  # time: 0
                    [ N, N, 3, I, 3, 1, N],  # time: 1
                    [ N, N, 2, I, 2, 0, N],  # time: 2
                    [ N, N, 1, I, 1,-1, N],  # time: 3
                    [ N, N, 0, I, 0,-2, N],  # time: 4
                ],
            ),
        ),
        Config(
            thr=0.5,
            mem_min=5,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ N, N, N, N, N, N, N],  # time: 0
                    [ N, N, N, N, N, N, N],  # time: 1
                    [ N, N, N, N, N, N, N],  # time: 2
                    [ N, N, N, N, N, N, N],  # time: 3
                    [ N, N, N, N, N, N, N],  # time: 4
                ],
            ),
        ),
        Config(
            thr=1.5,
            mem_min=1,
            sol_norm=np.array(
                [  #  0  1  2  3  4  5  6
                    [ 2, N, I, I, I, I, 4],  # time: 0
                    [ 1, N, I, I, I, I, 3],  # time: 1
                    [ 0, N, I, I, I, I, 2],  # time: 2
                    [-1, N, I, I, I, I, 1],  # time: 3
                    [-2, N, I, I, I, I, 0],  # time: 4
                ],
            ),
        ),
        Config(
            thr=1.5,
            mem_min=2,
            sol_norm=np.array(
                [  # 0  1  2  3  4  5  6
                    [ N, N, I, I, 4, 2, N],  # time: 0
                    [ N, N, I, I, 3, 1, N],  # time: 1
                    [ N, N, I, I, 2, 0, N],  # time: 2
                    [ N, N, I, I, 1,-1, N],  # time: 3
                    [ N, N, I, I, 0,-2, N],  # time: 4
                ],
            ),
        ),
        Config(
            thr=1.5,
            mem_min=3,
            sol_norm=np.array(
                [  # 0  1  2  3  4  5  6
                    [ N, N, N, I, 4, N, N],  # time: 0
                    [ N, N, N, I, 3, N, N],  # time: 1
                    [ N, N, N, I, 2, N, N],  # time: 2
                    [ N, N, N, I, 1, N, N],  # time: 3
                    [ N, N, N, I, 0, N, N],  # time: 4
                ],
            ),
        ),
        Config(
            thr=1.5,
            mem_min=4,
            sol_norm=np.array(
                [  # 0  1  2  3  4  5  6
                    [ N, N, N, 4, N, N, N],  # time: 0
                    [ N, N, N, 3, N, N, N],  # time: 1
                    [ N, N, N, 2, N, N, N],  # time: 2
                    [ N, N, N, 1, N, N, N],  # time: 3
                    [ N, N, N, 0, N, N, N],  # time: 4
                ],
            ),
        ),
        Config(
            thr=2.5,
            mem_min=1,
            sol_norm=np.array(
                [  # 0  1  2  3  4  5  6
                    [ N, N, I, I, I, I, N],  # time: 0
                    [ N, N, I, I, I, I, N],  # time: 1
                    [ N, N, I, I, I, I, N],  # time: 2
                    [ N, N, I, I, I, I, N],  # time: 3
                    [ N, N, I, I, I, I, N],  # time: 4
                ],
            ),
        ),
        Config(
            thr=2.5,
            mem_min=2,
            sol_norm=np.array(
                [  # 0  1  2  3  4  5  6
                    [ N, N, N, 3, 3, N, N],  # time: 0
                    [ N, N, N, 2, 2, N, N],  # time: 1
                    [ N, N, N, 1, 1, N, N],  # time: 2
                    [ N, N, N, 0, 0, N, N],  # time: 3
                    [ N, N, N,-1,-1, N, N],  # time: 4
                ],
            ),
        ),
        Config(
            thr=3.5,
            mem_min=1,
            sol_norm=np.array(
                [  # 0  1  2  3  4  5  6
                    [ N, N, N, N, N, N, N],  # time: 0
                    [ N, N, N, N, N, N, N],  # time: 1
                    [ N, N, N, N, N, N, N],  # time: 2
                    [ N, N, N, N, N, N, N],  # time: 3
                    [ N, N, N, N, N, N, N],  # time: 4
                ],
            ),
        ),
    ],
)
# fmt: on
def test_departure_time(config):
    res = EnsembleCloud(
        arr=ARR, time=np.arange(5) * D_TIME, thr=config.thr  # scale time step
    ).departure_time(config.mem_min)
    sol = config.sol_norm * D_TIME  # scale time step
    np.testing.assert_array_equal(res, sol)
