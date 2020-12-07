"""Exceptions."""

# Derived from Exception


class InvalidParameterError(Exception):
    """Parameter is invalid."""


class TypeCastError(Exception):
    """Error casting a value to a type."""


# Derived from other standard exceptions


class KeyConflictError(KeyError):
    """Key conflict."""


# Derived from custom exceptions


class InvalidParameterNameError(InvalidParameterError):
    """Parameter has invalid name."""


class InvalidParameterValueError(InvalidParameterError):
    """Parameter has invalid value."""
