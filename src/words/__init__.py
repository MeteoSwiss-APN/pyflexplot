"""Top-level package for words."""

__author__ = """Stefan Ruedisuehli"""
__email__ = "stefan.ruedisuehli@env.ethz.ch"
__version__ = "0.1.0"

# Standard library
from typing import List

# Local
from .word import TranslatedWord
from .word import Word
from .word import WordT
from .words import TranslatedWords
from .words import Words

__all__: List[str] = [
    "TranslatedWord",
    "TranslatedWords",
    "Word",
    "WordT",  # SR_TMP TODO Subclass ContextWord from Word and eliminate WordT
    "Words",
]
