"""Exceptions."""

# Derived from Exception


class InvalidParameterError(Exception):
    """Parameter is invalid."""


class InvalidVariableNameCharFilterError(Exception):
    """Function to filter invalid variable name characters is invalid."""


class InvalidVariableNameError(Exception):
    """A string is not a valid variable name."""


class TypeCastError(Exception):
    """Error casting a value to a type."""


class UnexpandableValueError(Exception):
    """Value is not expandable."""


# Derived from other standard exceptions


class KeyConflictError(KeyError):
    """Key conflict."""


# Derived from custom exceptions


class InvalidParameterNameError(InvalidParameterError):
    """Parameter has invalid name."""


class InvalidParameterValueError(InvalidParameterError):
    """Parameter has invalid value."""
