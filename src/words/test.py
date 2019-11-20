# -*- coding: utf-8 -*-
"""
Wrappers for ``Word`` and ``Words`` with testing-friendly interface.
"""
from .word import Word
from .words import Words

class TestWord(Word):
    pass

class TestWords(Words):

    cls_word = TestWord

    def __init__(self, raw_words, langs=None):
        if langs is None:
            words_langs = {w: w for w in raw_words}
        else:
            words_langs = {w: {l: w for l in langs} for w in raw_words}
        super().__init__(**words_langs)
