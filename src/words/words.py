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

        langs = None
        self._words = {}
        for name, word in words.items():
            try:
                langs_i = list(word.keys())
            except AttributeError:
                raise ValueError(f"word {name} not a dict: {word}")
            if langs is None:
                langs = langs_i
            elif set(langs) != set(langs):
                raise ValueError(
                    f"word '{name}' not defined in all necessary languages: "
                    f"set({langs}) != set({langs})")
            if default_ is None:
                default_ = next(iter(langs))
            elif default_ not in langs:
                raise ValueError(
                    f"invalid default language '{default_}': "
                    f"not in {langs}")
            self._words[name] = Word(name, default=default_, **word)
        self._langs_ = langs
        self.set_default(default_)

    def set_default(self, lang):
        """Change default language recursively for all words."""
        if lang not in self.langs_:
            raise ValueError(
                f"unknown language '{lang}': not among {self.langs_}")
        self._default_ = lang
        for word in self._words.values():
            word.set_default(lang)

    @property
    def default_(self):
        """Return the default language."""
        return self._default_

    @property
    def langs_(self):
        """Return the languages the words are defined in."""
        return [lang for lang in self._langs_]

    def __repr__(self):
        s_name = f' ({self.name})' if self.name else ''
        return f'{len(self._words)} Words{s_name}: {self._words}'

    def __getattr__(self, name):
        try:
            return self._words[name]
        except KeyError:
            raise ValueError(f"unknown word: {name}")
