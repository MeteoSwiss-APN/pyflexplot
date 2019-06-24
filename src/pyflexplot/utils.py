# -*- coding: utf-8 -*-
"""
Utils for the command line tool.
"""
import logging


class MaxIterationError(Exception):
    """Maximum number of iterations of a loop exceeded."""
    pass


class KeyConflictError(Exception):
    """Conflicting dictionary keys."""
    pass


def count_to_log_level(count: int) -> int:
    """Map the occurence of the command line option verbose to the log level"""
    if count == 0:
        return logging.ERROR
    elif count == 1:
        return logging.WARNING
    elif count == 2:
        return logging.INFO
    else:
        return logging.DEBUG


def merge_dicts(dicts, unique_keys=True):
    """Merge multiple dictionaries with or without shared keys.

    Args:
        dicts (list[dict]) Dictionaries to be merged.

        unique_keys (bool, optional): Whether keys must be unique.
            If True, duplicate keys raise a ``KeyConflictError``
            exception. If False, dicts take precedence in reverse order
            of ``dicts``, i.e., keys occurring in multiple dicts will
            have the value of the last dict containing that key.

    Raises:
        KeyConflictError: If ``unique_keys=True`` when a key occurs
            in multiple dicts.

    Returns:
        dict: Merged dictionary.

    """
    merged = {}
    for dict_ in dicts:
        if not unique_keys:
            merged.update(dict_)
        else:
            for key, val in dict_.items():
                if key in merged:
                    raise KeyConflictError(key)
                merged[key] = val
    return merged
