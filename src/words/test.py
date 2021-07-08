"""Wrappers for ``TranslatedWord[s]`` with testing-friendly interface."""
from __future__ import annotations

# Standard library
from typing import Sequence
from typing import Type

# Local
from .word import TranslatedWord
from .words import TranslatedWords


class TranslatedTestWord(TranslatedWord):
    """Testing wrapper for ``TranslatedWord``."""


class TranslatedTestWords(TranslatedWords):
    """Testing wrapper for ``TranslatedWords``."""

    cls_word: Type[TranslatedWord] = TranslatedTestWord

    def __init__(self, raw_words: Sequence[str], langs: Sequence[str]) -> None:
        """Create an instance of ``TranslatedTestWords``."""
        words_langs: dict[str, dict[str, str]] = {
            word: {lang: word for lang in langs} for word in raw_words
        }
        name = None
        super().__init__(name, words_langs)
