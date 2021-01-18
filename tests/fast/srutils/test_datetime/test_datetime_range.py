"""Tests for function ``srutils.datetime.datetime_range``."""
# Standard library
import dataclasses as dc
import datetime as dt
from typing import Any
from typing import Dict
from typing import List
from typing import Union

# Third-party
import pytest

# First-party
from srutils.datetime import datetime_range


@dc.dataclass
class Config:
    start: Union[dt.datetime, str, int]
    end: Union[dt.datetime, str, int]
    step: Union[dt.timedelta, str, int]
    sol: List[Union[dt.datetime, str, int]]
    kwargs: Dict[str, Any] = dc.field(default_factory=dict)


UTC = dt.timezone.utc


@pytest.mark.parametrize(
    "config",
    [
        Config(  # [config0]
            start=20210103,
            end=20210104,
            step=6 * 3600,
            sol=[
                dt.datetime(2021, 1, 3, 0, tzinfo=UTC),
                dt.datetime(2021, 1, 3, 6, tzinfo=UTC),
                dt.datetime(2021, 1, 3, 12, tzinfo=UTC),
                dt.datetime(2021, 1, 3, 18, tzinfo=UTC),
                dt.datetime(2021, 1, 4, 0, tzinfo=UTC),
            ],
        ),
        Config(  # [config1]
            start="20210103",
            end="20210104",
            step=6 * 3600,
            kwargs={"convert": int},
            sol=[
                20210103000000,
                20210103060000,
                20210103120000,
                20210103180000,
                20210104000000,
            ],
        ),
        Config(  # [config2]
            start="20210103",
            end="20210104",
            step=6 * 3600,
            kwargs={"convert": str, "fmt": "%Y-%m-%d_%H:%M"},
            sol=[
                "2021-01-03_00:00",
                "2021-01-03_06:00",
                "2021-01-03_12:00",
                "2021-01-03_18:00",
                "2021-01-04_00:00",
            ],
        ),
        Config(  # [config3]
            start="2021010300",
            end="2021010400",
            step=6 * 3600,
            kwargs={"convert": int},
            sol=[
                20210103000000,
                20210103060000,
                20210103120000,
                20210103180000,
                20210104000000,
            ],
        ),
        Config(  # [config4]
            start="2021010300",
            end="2021010403",
            step=6 * 3600,
            kwargs={"convert": int},
            sol=[
                20210103000000,
                20210103060000,
                20210103120000,
                20210103180000,
                20210104000000,
            ],
        ),
        Config(  # [config5]
            start="2021010300",
            end="2021010406",
            step=6 * 3600,
            kwargs={"convert": int},
            sol=[
                20210103000000,
                20210103060000,
                20210103120000,
                20210103180000,
                20210104000000,
                20210104060000,
            ],
        ),
    ],
)
def test(config):
    res = datetime_range(config.start, config.end, config.step, **config.kwargs)
    assert res == config.sol
