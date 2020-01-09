# -*- coding: utf-8 -*-
"""Top-level package for words."""

__author__ = """Stefan Ruedisuehli"""
__email__ = "stefan.ruedisuehli@env.ethz.ch"
__version__ = "0.1.0"

from .word import Word
from .word import TranslatedWord
from .words import Words
from .words import TranslatedWords

__all__ = [Word, TranslatedWord, Words, TranslatedWords]
