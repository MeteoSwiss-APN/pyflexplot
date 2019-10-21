# -*- coding: utf-8 -*-
"""
Words.
"""

__version__ = '0.1.0'

from .word import Word


class Words:
    """A collection of words in different languages."""

    def __init__(self, name_=None, *, default_=None, **words):
        """Create an instance of ``Words``.

        Args:
            name_ (str, optional): Name of the words collection.
                Defaults to None.

            default_ (str, optional): Default language. Defaults to
                the first language in which the first word is defined.

            **words (dict of str: dict): Words. The keys constitute the
                names of the words, and each value the definition of a
                word in one or more language. All words must be defined
                in the same languages.

        """
        self.name_ = name_
        self.default_ = default_

        self.langs_ = None
        self._words = {}
        for name, word in words.items():
            try:
                langs = list(word.keys())
            except AttributeError:
                raise ValueError(f"word {name} not a dict: {word}")
            if self.langs_ is None:
                self.langs_ = langs
            elif set(langs) != set(self.langs_):
                raise ValueError(
                    f"word '{name}' not defined in all necessary languages: "
                    f"set({langs}) != set({self.langs_})")
            if self.default_ is None:
                self.default_ = next(iter(self.langs_))
            elif self.default_ not in self.langs_:
                raise ValueError(
                    f"invalid default language '{self.default_}': "
                    f"not in {self.langs_}")
            self._words[name] = Word(name, default=self.default_, **word)

    def __repr__(self):
        s_name = f' ({self.name})' if self.name else ''
        return f'{len(self._words)} Words{s_name}: {self._words}'

    def __getattr__(self, name):
        try:
            return self._words[name]
        except KeyError:
            raise ValueError(f"unknown word: {name}")
