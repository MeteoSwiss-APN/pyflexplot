# flake8: noqa  # complains about code in "fmt: off/on" blocks
"""Tests for class ``pyflexplot.data.Cloud``."""
# Standard library
from dataclasses import dataclass

# Third-party
import numpy as np
import pytest  # noqa: F401  # imported but unused

# First-party
from pyflexplot.data import Cloud

# Shorthands for special values
I = np.inf  # noqa: E741  # ambiguous variable name


def a(*args, **kwargs):
    """Shorthand function to create array."""
    return np.array(*args, **kwargs)


@dataclass
class Config:
    thr: float
    dt: float
    time_idx: int
    sol_norm: np.ndarray


DTS = [1.0, 2.0, 0.5] + np.linspace(0.1, 10, 7).tolist()
N_MEM = 4
ARR = (
    np.array(
        [
            [  # 0  1  2  3  4  5  6
                [1, 0, 0, 0, 0, 0, 1],  # time: 0, x: 0, y: 0-6
                [0, 0, 0, 0, 1, 3, 2],  # time: 1, x: 0, y: 0-6
                [0, 0, 0, 0, 0, 0, 0],  # time: 2, x: 0, y: 0-6
                [0, 0, 1, 3, 1, 0, 0],  # time: 3, x: 0, y: 0-6
                [0, 0, 0, 2, 1, 0, 0],  # time: 4, x: 0, y: 0-6
            ],
            [  # 0  1  2  3  4  5  6
                [2, 1, 0, 0, 0, 1, 1],  # time: 0, x: 1, y: 0-6
                [1, 0, 0, 0, 2, 2, 1],  # time: 1, x: 1, y: 0-6
                [0, 0, 0, 0, 2, 2, 2],  # time: 2, x: 1, y: 0-6
                [0, 0, 0, 2, 1, 1, 0],  # time: 3, x: 1, y: 0-6
                [0, 0, 0, 1, 0, 0, 0],  # time: 4, x: 1, y: 0-6
            ],
            [  # 0  1  2  3  4  5  6
                [1, 0, 0, 0, 0, 1, 2],  # time: 0, x: 2, y: 0-6
                [0, 0, 0, 1, 3, 3, 0],  # time: 1, x: 2, y: 0-6
                [0, 0, 0, 3, 3, 0, 0],  # time: 2, x: 2, y: 0-6
                [0, 1, 3, 3, 2, 0, 0],  # time: 3, x: 2, y: 0-6
                [0, 2, 3, 2, 0, 0, 0],  # time: 4, x: 2, y: 0-6
            ],
            [  # 0  1  2  3  4  5  6
                [0, 0, 1, 2, 3, 2, 2],  # time: 0, x: 3, y: 0-6
                [0, 0, 1, 2, 2, 1, 0],  # time: 1, x: 3, y: 0-6
                [1, 0, 2, 3, 2, 0, 0],  # time: 2, x: 3, y: 0-6
                [0, 0, 1, 2, 1, 0, 0],  # time: 3, x: 3, y: 0-6
                [0, 0, 0, 1, 0, 0, 0],  # time: 4, x: 3, y: 0-6
            ],
            [  # 0  1  2  3  4  5  6
                [0, 0, 0, 2, 3, 3, 1],  # time: 0, x: 4, y: 0-6
                [0, 1, 2, 3, 1, 0, 0],  # time: 1, x: 4, y: 0-6
                [0, 0, 0, 2, 1, 0, 0],  # time: 2, x: 4, y: 0-6
                [0, 1, 3, 1, 0, 0, 0],  # time: 3, x: 4, y: 0-6
                [0, 0, 0, 0, 0, 0, 0],  # time: 4, x: 4, y: 0-6
            ],
        ]
    )
    .swapaxes(0, 1)
    .astype(float)
)


# test_arrival_time
# fmt: off
@pytest.mark.parametrize(
    "config",
    [
        config
        for dt in DTS
        for config in [
            Config(  # [config{0,7,14,21,28,35,42,49,56,63}]
                thr=0.5,
                dt=dt,
                time_idx=0,
                sol_norm=np.array(
                    [  #  0  1  2  3  4  5  6
                        [-I, I, 3, 3, 1, 1,-I],  # x: 0, y: 0-6
                        [-I,-I, I, 3, 1,-I,-I],  # x: 1, y: 0-6
                        [-I, 3, 3, 1, 1,-I,-I],  # x: 2, y: 0-6
                        [ 2, I,-I,-I,-I,-I,-I],  # x: 3, y: 0-6
                        [ I, 1, 1,-I,-I,-I,-I],  # x: 4, y: 0-6
                    ],
                ),
            ),
            Config(  # [config{1,8,15,22,29,36,43,50,57,64}]
                thr=0.5,
                dt=dt,
                time_idx=2,
                sol_norm=np.array(
                    [  #  0  1  2  3  4  5  6
                        [ I, I, 1, 1,-2, I, I],  # x: 0, y: 0-6
                        [ I, I, I, 1,-2,-I,-I],  # x: 1, y: 0-6
                        [ I, 1, 1,-2,-2, I, I],  # x: 2, y: 0-6
                        [-1, I,-I,-I,-I, I, I],  # x: 3, y: 0-6
                        [ I,-2,-2,-I,-I, I, I],  # x: 4, y: 0-6
                    ],
                ),
            ),
            Config(  # [config{2,9,16,23,30,37,44,51,58,65}]
                thr=1.5,
                dt=dt,
                time_idx=1,
                sol_norm=np.array(
                    [  #  0  1  2  3  4  5  6
                        [ I, I, I, 2, I,-1,-1],  # x: 0, y: 0-6
                        [ I, I, I, 2,-1,-1, 1],  # x: 1, y: 0-6
                        [ I, 3, 2, 1,-1,-1, I],  # x: 2, y: 0-6
                        [ I, I, 1,-I,-I, I, I],  # x: 3, y: 0-6
                        [ I, I,-1,-I, I, I, I],  # x: 4, y: 0-6
                    ],
                ),
            ),
            Config(  # [config{3,10,17,24,31,38,45,52,59,66}]
                thr=1.5,
                dt=dt,
                time_idx=3,
                sol_norm=np.array(
                    [  #  0  1  2  3  4  5  6
                        [ I, I, I,-1, I, I, I],  # x: 0, y: 0-6
                        [ I, I, I,-1, I, I, I],  # x: 1, y: 0-6
                        [ I, 1,-1,-2,-3, I, I],  # x: 2, y: 0-6
                        [ I, I, I,-I, I, I, I],  # x: 3, y: 0-6
                        [ I, I,-3, I, I, I, I],  # x: 4, y: 0-6
                    ],
                ),
            ),
            Config(  # [config{4,11,18,25,32,39,46,53,60,67}]
                thr=2.5,
                dt=dt,
                time_idx=0,
                sol_norm=np.array(
                    [  #  0  1  2  3  4  5  6
                        [ I, I, I, 3, I, 1, I],  # x: 0, y: 0-6
                        [ I, I, I, I, I, I, I],  # x: 1, y: 0-6
                        [ I, I, 3, 2, 1, 1, I],  # x: 2, y: 0-6
                        [ I, I, I, 2,-I, I, I],  # x: 3, y: 0-6
                        [ I, I, 3, 1,-I,-I, I],  # x: 4, y: 0-6
                    ],
                ),
            ),
            Config(  # [config{5,12,19,26,33,40,47,54,61,68}]
                thr=2.5,
                dt=dt,
                time_idx=2,
                sol_norm=np.array(
                    [  #  0  1  2  3  4  5  6
                        [ I, I, I, 1, I, I, I],  # x: 0, y: 0-6
                        [ I, I, I, I, I, I, I],  # x: 1, y: 0-6
                        [ I, I, 1,-1,-2, I, I],  # x: 2, y: 0-6
                        [ I, I, I,-1, I, I, I],  # x: 3, y: 0-6
                        [ I, I, 1, I, I, I, I],  # x: 4, y: 0-6
                    ],
                ),
            ),
            Config(  # [config{6,13,20,27,34,41,48,55,62,69}]
                thr=3.5,
                dt=dt,
                time_idx=0,
                sol_norm=np.array(
                    [  #  0  1  2  3  4  5  6
                        [ I, I, I, I, I, I, I],  # x: 0, y: 0-6
                        [ I, I, I, I, I, I, I],  # x: 1, y: 0-6
                        [ I, I, I, I, I, I, I],  # x: 2, y: 0-6
                        [ I, I, I, I, I, I, I],  # x: 3, y: 0-6
                        [ I, I, I, I, I, I, I],  # x: 4, y: 0-6
                    ],
                ),
            ),
        ]
    ],
)
# fmt: on
def test_arrival_time(config):
    res = Cloud(mask=ARR > config.thr, ts=config.dt).arrival_time()[config.time_idx]
    sol = config.sol_norm * config.dt  # scale time step
    msg = f"test_arrival_time:\n{config}\n"
    np.testing.assert_array_almost_equal(res, sol, decimal=5, err_msg=msg)


# test_departure_time
# fmt: off
@pytest.mark.parametrize(
    "config",
    [
        config
        for dt in DTS
        for config in [
            Config(  # [config{0,7,14,21,28,35,42,49,56,63}]
                thr=0.5,
                dt=dt,
                time_idx=0,
                sol_norm=np.array(  # [config0]
                    [  #  0  1  2  3  4  5  6
                        [ 1,-I, 4, I, I, 2, 2],  # x: 0, y: 0-6
                        [ 2, 1,-I, I, 4, 4, 3],  # x: 1, y: 0-6
                        [ 1, I, I, I, 4, 2, 1],  # x: 2, y: 0-6
                        [ 3,-I, 4, I, 4, 2, 1],  # x: 3, y: 0-6
                        [-I, 4, 4, 4, 3, 1, 1],  # x: 4, y: 0-6
                    ],
                ),
            ),
            Config(  # [config{1,8,15,22,29,36,43,50,57,64}]
                thr=0.5,
                dt=dt,
                time_idx=2,
                sol_norm=np.array(  # [config1]
                    [  #  0  1  2  3  4  5  6
                        [-I,-I, 2, I, I,-I,-I],  # x: 0, y: 0-6
                        [-I,-I,-I, I, 2, 2, 1],  # x: 1, y: 0-6
                        [-I, I, I, I, 2,-I,-I],  # x: 2, y: 0-6
                        [ 1,-I, 2, I, 2,-I,-I],  # x: 3, y: 0-6
                        [-I, 2, 2, 2, 1,-I,-I],  # x: 4, y: 0-6
                    ],
                ),
            ),
            Config(  # [config{2,9,16,23,30,37,44,51,58,65}]
                thr=1.5,
                dt=dt,
                time_idx=1,
                sol_norm=np.array(  # [config2]
                    [  #  0  1  2  3  4  5  6
                        [-I,-I,-I, I,-I, 1, 1],  # x: 0, y: 0-6
                        [-I,-I,-I, 3, 2, 2, 2],  # x: 1, y: 0-6
                        [-I, I, I, I, 3, 1,-I],  # x: 2, y: 0-6
                        [-I,-I, 2, 3, 2,-I,-I],  # x: 3, y: 0-6
                        [-I,-I, 3, 2,-I,-I,-I],  # x: 4, y: 0-6
                    ],
                ),
            ),
            Config(  # [config{3,10,17,24,31,38,45,52,59,66}]
                thr=1.5,
                dt=dt,
                time_idx=3,
                sol_norm=np.array(  # [config3]
                    [  #  0  1  2  3  4  5  6
                        [-I,-I,-I, I,-I,-I,-I],  # x: 0, y: 0-6
                        [-I,-I,-I, 1,-I,-I,-I],  # x: 1, y: 0-6
                        [-I, I, I, I, 1,-I,-I],  # x: 2, y: 0-6
                        [-I,-I,-I, 1,-I,-I,-I],  # x: 3, y: 0-6
                        [-I,-I, 1,-I,-I,-I,-I],  # x: 4, y: 0-6
                    ],
                ),
            ),
            Config(  # [config{4,11,18,25,32,39,46,53,60,67}]
                thr=2.5,
                dt=dt,
                time_idx=0,
                sol_norm=np.array(  # [config4]
                    [  #  0  1  2  3  4  5  6
                        [-I,-I,-I, 4,-I, 2,-I],  # x: 0, y: 0-6
                        [-I,-I,-I,-I,-I,-I,-I],  # x: 1, y: 0-6
                        [-I,-I, I, 4, 3, 2,-I],  # x: 2, y: 0-6
                        [-I,-I,-I, 3, 1,-I,-I],  # x: 3, y: 0-6
                        [-I,-I, 4, 2, 1, 1,-I],  # x: 4, y: 0-6
                    ],
                ),
            ),
            Config(  # [config{5,12,19,26,33,40,47,54,61,68}]
                thr=2.5,
                dt=dt,
                time_idx=2,
                sol_norm=np.array(  # [config5]
                    [  #  0  1  2  3  4  5  6
                        [-I,-I,-I, 2,-I,-I,-I],  # x: 0, y: 0-6
                        [-I,-I,-I,-I,-I,-I,-I],  # x: 1, y: 0-6
                        [-I,-I, I, 2, 1,-I,-I],  # x: 2, y: 0-6
                        [-I,-I,-I, 1,-I,-I,-I],  # x: 3, y: 0-6
                        [-I,-I, 2,-I,-I,-I,-I],  # x: 4, y: 0-6
                    ],
                ),
            ),
            Config(  # [config{6,13,20,27,34,41,48,55,62,69}]
                thr=3.5,
                dt=dt,
                time_idx=0,
                sol_norm=np.array(  # [config6]
                    [  #  0  1  2  3  4  5  6
                        [-I,-I,-I,-I,-I,-I,-I],  # x: 0, y: 0-6
                        [-I,-I,-I,-I,-I,-I,-I],  # x: 1, y: 0-6
                        [-I,-I,-I,-I,-I,-I,-I],  # x: 2, y: 0-6
                        [-I,-I,-I,-I,-I,-I,-I],  # x: 3, y: 0-6
                        [-I,-I,-I,-I,-I,-I,-I],  # x: 4, y: 0-6
                    ],
                ),
            ),
        ]
    ],
)
# fmt: on
def test_departure_time(config):
    res = Cloud(mask=ARR > config.thr, ts=config.dt).departure_time()[config.time_idx]
    sol = config.sol_norm * config.dt  # scale time step
    msg = f"test_departure_time:\n{config}\n"
    np.testing.assert_array_almost_equal(res, sol, decimal=5, err_msg=msg)
