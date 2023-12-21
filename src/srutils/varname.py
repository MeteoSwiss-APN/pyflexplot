"""Represent a string as a Python variable name."""
# Standard library
import re
from typing import Any
from typing import Callable
from typing import Optional

# Local
from .exceptions import InvalidVariableNameCharFilterError
from .exceptions import InvalidVariableNameError


class VariableName:
    """A Python variable name.

    Valid characters are all letters, underscores, and numbers (except as the
    first characters). All other characters must either be converted to one of
    the former, or removed altogether. By default, they are all converted to
    underscores.

    """

    def __init__(self, s: str) -> None:
        """Create an instance of ``VariableName``.

        No validity check is performed during instantiation; use the method
        ``is_valid`` to perform one.

        Args:
            s: String to be represented as a valid variable name.

        """
        self._s = str(s)

    def is_valid(self) -> bool:
        """Return whether instance is valid variable name."""
        try:
            self.check_valid(self._s)
        except InvalidVariableNameError:
            return False
        else:
            return True

    def convert(self, **kwargs: Any) -> None:
        """Convert in-place to a valid variable name.

        See docstring of ``VariableName.format`` for info on ``kwargs``.

        """
        self._s = self.format(**kwargs)

    def format(
        self, *, lower: bool = False, c_filter: Optional[Callable[[str], str]] = None
    ) -> str:
        """Format to a valid variable name, leaving the instance as is.

        Args:
            lower (optional): Convert the variable to lowercase.

            c_filter (optional): Filter function applied to an invalid variable
                name character; by default, all invalid characters are replaced
                by an underscore ('_').

        """
        rx_valid_first = re.compile("[a-zA-Z_]")
        rx_valid_rest = re.compile("[a-zA-Z0-9_]")
        var = ""
        try:
            for i, c in enumerate(self._s):
                rx_valid = rx_valid_first if i == 0 else rx_valid_rest
                if not rx_valid.match(c):
                    c = self._filter_c(c, c_filter)
                var += c
            self.check_valid(var)
        except (InvalidVariableNameError, InvalidVariableNameCharFilterError) as e:
            raise e if not c_filter else ValueError(
                f"invalid character filter {c_filter}: {e}"
            ) from e
        if lower:
            var = var.lower()
        return var

    def _filter_c(self, c: str, c_filter: Optional[Callable[[str], str]] = None) -> str:
        """Filter an invalid variable name character."""
        if len(c) != 1:
            raise ValueError(
                f"ch has invalid value '{c}': expected single character, got {len(c)}"
            )
        if c_filter is None:
            c_filter = self._default_c_filter
        try:
            c = c_filter(c)
        except Exception as e:
            raise InvalidVariableNameCharFilterError(
                f"raises {type(e).__name__}('{str(e)}') for '{c}'"
            ) from e
        if not isinstance(c, str):
            raise InvalidVariableNameCharFilterError(
                f"returns a {type(c).__name__}, not a str"
            )
        return c

    def __str__(self) -> str:
        return self._s

    @staticmethod
    def check_valid(s: str) -> None:
        """Check whether a string is valid variable name.

        Raises:
            InvalidVariableNameError: If instance is not a valid variable name.

        """
        if re.match(r"^[0-9]", s):
            raise InvalidVariableNameError(f"starts with number: {s}")
        if not re.match(r"^[a-zA-Z0-9_]*$", s):
            raise InvalidVariableNameError(f"contains invalid characters: {s}")

    @staticmethod
    def _default_c_filter(ch: str):
        """Replace an invalid variable name character by an underscore ('_')."""
        if len(ch) != 1:
            raise ValueError(
                f"ch has invalid value '{ch}': expected single character, got {len(ch)}"
            )
        return "_"
