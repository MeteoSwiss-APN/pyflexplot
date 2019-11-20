# -*- coding: utf-8 -*-
"""
Wrappers for ``TranslatedWord`` and ``TranslatedWords`` with testing-friendly interface.
"""
from .word import TranslatedWord
from .words import TranslatedWords

class TranslatedTestWord(TranslatedWord):
    pass

class TranslatedTestWords(TranslatedWords):

    cls_word = TranslatedTestWord

    def __init__(self, raw_words, langs=None):
        if langs is None:
            words_langs = {w: w for w in raw_words}
        else:
            words_langs = {w: {l: w for l in langs} for w in raw_words}
        super().__init__(**words_langs)
