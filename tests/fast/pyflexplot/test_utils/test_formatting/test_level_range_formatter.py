"""Tests for classes pyflexplot.utils.formatting.LevelRangeFormatter*``."""
# Standard library
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Type

# Third-party
import numpy as np
import pytest

# First-party
from pyflexplot.utils.formatting import LevelRangeFormatter
from pyflexplot.utils.formatting import LevelRangeFormatterAnd
from pyflexplot.utils.formatting import LevelRangeFormatterDown
from pyflexplot.utils.formatting import LevelRangeFormatterInt
from pyflexplot.utils.formatting import LevelRangeFormatterMath
from pyflexplot.utils.formatting import LevelRangeFormatterUp
from pyflexplot.utils.formatting import LevelRangeFormatterVar


@dataclass
class Config:
    cls: Type[LevelRangeFormatter]
    sol: List[str]
    kwargs: Dict[str, Any] = field(default_factory=dict)


@pytest.mark.parametrize(
    "config",
    [
        Config(  # [config0]
            cls=LevelRangeFormatter,
            sol=[
                "3.000 $\\tt -$ 6.000",
                "6.000 $\\tt -$ 9.000",
                "9.000 $\\tt -$ 12.00",
                "12.00 $\\tt -$ 15.00",
                "15.00 $\\tt -$ 18.00",
                "18.00 $\\tt -$ 21.00",
            ],
        ),
        Config(  # [config1]
            cls=LevelRangeFormatterAnd,
            sol=[
                "$\\tt \\geq$ 3.000  $\\tt &$   $\\tt <$ 6.000",
                "$\\tt \\geq$ 6.000  $\\tt &$   $\\tt <$ 9.000",
                "$\\tt \\geq$ 9.000  $\\tt &$   $\\tt <$ 12.00",
                "$\\tt \\geq$ 12.00  $\\tt &$   $\\tt <$ 15.00",
                "$\\tt \\geq$ 15.00  $\\tt &$   $\\tt <$ 18.00",
                "$\\tt \\geq$ 18.00  $\\tt &$   $\\tt <$ 21.00",
            ],
        ),
        Config(  # [config2]
            cls=LevelRangeFormatterDown,
            sol=[
                "$\\tt <$ 6.000",
                "$\\tt <$ 9.000",
                "$\\tt <$ 12.00",
                "$\\tt <$ 15.00",
                "$\\tt <$ 18.00",
                "$\\tt <$ 21.00",
            ],
        ),
        Config(  # [config3]
            cls=LevelRangeFormatterInt,
            sol=[
                " 3 $\\tt -$  5",
                " 6 $\\tt -$  8",
                " 9 $\\tt -$ 11",
                "12 $\\tt -$ 14",
                "15 $\\tt -$ 17",
                "18 $\\tt -$ 20",
            ],
        ),
        Config(  # [config4]
            cls=LevelRangeFormatterMath,
            sol=[
                "[3.000, 6.000)",
                "[6.000, 9.000)",
                "[9.000, 12.00)",
                "[12.00, 15.00)",
                "[15.00, 18.00)",
                "[18.00, 21.00)",
            ],
        ),
        Config(  # [config5]
            cls=LevelRangeFormatterUp,
            sol=[
                "$\\tt \\geq$ 3.000",
                "$\\tt \\geq$ 6.000",
                "$\\tt \\geq$ 9.000",
                "$\\tt \\geq$ 12.00",
                "$\\tt \\geq$ 15.00",
                "$\\tt \\geq$ 18.00",
            ],
        ),
        Config(  # [config6]
            cls=LevelRangeFormatterVar,
            sol=[
                "3.000  $\\tt \\leq$ v $\\tt <$  6.000",
                "6.000  $\\tt \\leq$ v $\\tt <$  9.000",
                "9.000  $\\tt \\leq$ v $\\tt <$  12.00",
                "12.00  $\\tt \\leq$ v $\\tt <$  15.00",
                "15.00  $\\tt \\leq$ v $\\tt <$  18.00",
                "18.00  $\\tt \\leq$ v $\\tt <$  21.00",
            ],
        ),
        Config(  # [config7]
            cls=LevelRangeFormatterInt,
            sol=[
                "   $\\tt <$  3",
                " 3 $\\tt -$  5",
                " 6 $\\tt -$  8",
                " 9 $\\tt -$ 11",
                "12 $\\tt -$ 14",
                "15 $\\tt -$ 17",
                "18 $\\tt -$ 20",
                "   $\\tt \\geq$ 21",
            ],
            kwargs={
                "extend": "both",
                "include": "lower",
            },
        ),
        Config(  # [config8]
            cls=LevelRangeFormatterInt,
            sol=[
                "   $\\tt \\leq$  3",
                " 4 $\\tt -$  6",
                " 7 $\\tt -$  9",
                "10 $\\tt -$ 12",
                "13 $\\tt -$ 15",
                "16 $\\tt -$ 18",
                "19 $\\tt -$ 21",
                "   $\\tt >$ 21",
            ],
            kwargs={
                "extend": "both",
                "include": "upper",
            },
        ),
    ],
)
def test_base(config):
    levels = np.arange(3, 22, 3)
    formatter = config.cls(**config.kwargs)
    res = formatter.format_multiple(levels)
    assert res == config.sol
