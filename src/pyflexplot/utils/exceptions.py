"""Exceptions."""


# Primary


class ArrayDimensionError(Exception):
    """Array has wrong dimensions."""


class AttributeConflictError(Exception):
    """Conflicting object attributes."""


class FieldAllNaNError(Exception):
    """Field contains only NaN values."""


class InconsistentArrayShapesError(Exception):
    """Arrays have inconsistent shapes."""


class InvalidParameterError(Exception):
    """Parameter is invalid."""


class KeyConflictError(Exception):
    """Conflicting dictionary keys."""


class MaxIterationError(Exception):
    """Maximum number of iterations of a loop exceeded."""


class MinFontSizeReachedError(Exception):
    """Font size cannot be reduced further."""


class MinStrLenReachedError(Exception):
    """String cannot be further truncated."""


class MissingCacheEntryError(Exception):
    """Entry missing in cache."""


class NoPresetFileFoundError(Exception):
    """No preset file found in directory/ies."""


class NotSummarizableError(Exception):
    """Object could not be summarized."""


class UnequalSetupParamValuesError(Exception):
    """Values of a param differs between multiple setups."""


# Derived


class InvalidParameterNameError(InvalidParameterError):
    """Parameter has invalid name."""


class InvalidParameterValueError(InvalidParameterError):
    """Parameter has invalid value."""
