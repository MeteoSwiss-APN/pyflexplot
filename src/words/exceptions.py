# -*- coding: utf-8 -*-
"""
Exceptions.
"""


class WordsError(Exception):
    pass


class MissingWordError(WordsError):
    pass


class MissingLanguageError(WordsError):
    pass
