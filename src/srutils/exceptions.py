"""Exceptions."""

# Derived from Exception


class InvalidParameterError(Exception):
    """Parameter is invalid."""


class InvalidVariableNameCharFilterError(Exception):
    """Function to filter invalid variable name characters is invalid."""


class InvalidVariableNameError(Exception):
    """A string is not a valid variable name."""


class UnexpandableValueError(Exception):
    """Value is not expandable."""


# Derived from other standard exceptions


class IncompatibleTypesError(TypeError):
    """Types are incompatible."""


class KeyConflictError(KeyError):
    """Key conflict."""


class UnsupportedTypeError(TypeError):
    """Type is unsupported."""


# Derived from custom exceptions


class InvalidParameterNameError(InvalidParameterError):
    """Parameter has invalid name."""


class InvalidParameterValueError(InvalidParameterError):
    """Parameter has invalid value."""
