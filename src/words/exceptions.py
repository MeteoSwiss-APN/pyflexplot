# -*- coding: utf-8 -*-
"""
Exceptions.
"""


# Primary


class WordsError(Exception):
    """Error related to words."""


# Secondary


class MissingLanguageError(WordsError):
    """Language is missing."""


class MissingWordError(WordsError):
    """Word is missing."""
