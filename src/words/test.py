"""Wrappers for ``TranslatedWord[s]`` with testing-friendly interface."""
# Local
from .word import TranslatedWord
from .words import TranslatedWords


class TranslatedTestWord(TranslatedWord):
    """Testing wrapper for ``TranslatedWord``."""


class TranslatedTestWords(TranslatedWords):
    """Testing wrapper for ``TranslatedWords``."""

    cls_word = TranslatedTestWord

    def __init__(self, raw_words, langs=None):
        """Create an instance of ``TranslatedTestWords``."""
        if langs is None:
            words_langs = {word: word for word in raw_words}
        else:
            words_langs = {word: {lang: word for lang in langs} for word in raw_words}
        super().__init__(**words_langs)
