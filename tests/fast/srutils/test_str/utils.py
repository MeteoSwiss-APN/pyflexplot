"""Testing utilities."""
# Standard library
from contextlib import contextmanager
from typing import Generator
from typing import Tuple
from typing import Type
from typing import Union

# Third-party
import pytest


@contextmanager
def not_raises(
    unexpected_exception: Union[Type[Exception], Tuple[Type[Exception]]]
) -> Generator[None, None, None]:
    """Test that an exception is not raised.

    Based on: https://stackoverflow.com/a/42327075/4419816

    """
    try:
        yield
    except unexpected_exception as e:
        raise pytest.fail(f"DID RAISE {unexpected_exception}") from e
