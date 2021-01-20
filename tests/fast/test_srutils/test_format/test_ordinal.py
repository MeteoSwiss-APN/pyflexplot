"""Tests for function ``srutils.format.ordinal``."""
# Standard library
import dataclasses as dc

# Third-party
import pytest

# First-party
from srutils.format import ordinal


@dc.dataclass
class Config:
    in_: int
    out: str


@pytest.mark.parametrize(
    "config",
    [
        Config(0, "0th"),
        Config(1, "1st"),
        Config(2, "2nd"),
        Config(3, "3rd"),
        Config(4, "4th"),
        Config(5, "5th"),
        Config(6, "6th"),
        Config(7, "7th"),
        Config(8, "8th"),
        Config(9, "9th"),
        Config(10, "10th"),
        Config(11, "11th"),
        Config(12, "12th"),
        Config(13, "13th"),
        Config(14, "14th"),
        Config(15, "15th"),
        Config(16, "16th"),
        Config(17, "17th"),
        Config(18, "18th"),
        Config(19, "19th"),
        Config(20, "20th"),
        Config(21, "21st"),
        Config(22, "22nd"),
        Config(23, "23rd"),
        Config(24, "24th"),
        Config(25, "25th"),
        Config(26, "26th"),
        Config(27, "27th"),
        Config(28, "28th"),
        Config(29, "29th"),
        Config(30, "30th"),
        Config(31, "31st"),
        Config(32, "32nd"),
        Config(33, "33rd"),
        Config(34, "34th"),
        Config(35, "35th"),
        Config(36, "36th"),
        Config(37, "37th"),
        Config(38, "38th"),
        Config(39, "39th"),
        Config(100, "100th"),
        Config(101, "101st"),
        Config(102, "102nd"),
        Config(103, "103rd"),
        Config(2_210, "2210th"),
        Config(2_211, "2211th"),
        Config(2_212, "2212th"),
        Config(2_213, "2213th"),
        Config(2_214, "2214th"),
        Config(3_4520, "34520th"),
        Config(3_4521, "34521st"),
        Config(3_4522, "34522nd"),
        Config(3_4523, "34523rd"),
        Config(3_4524, "34524th"),
        Config(1_999_930, "1999930th"),
        Config(1_999_931, "1999931st"),
        Config(1_999_932, "1999932nd"),
        Config(1_999_933, "1999933rd"),
        Config(1_999_934, "1999934th"),
    ],
)
def test_ordinal(config):
    assert ordinal(config.in_) == config.out
