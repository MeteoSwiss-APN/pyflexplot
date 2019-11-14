# -*- coding: utf-8 -*-
"""
Words.
"""

from .word import Word


class Words:
    """A collection of words in different languages."""

    cls_word = Word

    def __init__(self, name=None, *, default_lang=None, **words):
        """Create an instance of ``Words``.

        Args:
            name (str, optional): Name of the words collection.
                Defaults to None.

            default_lang (str, optional): Default language. Defaults to
                the first language in which the first word is defined.

            **words (dict of str: dict): Words. The keys constitute the
                names of the words, and each value the definition of a
                word in one or more language. All words must be defined
                in the same languages.

        """
        self.name = name

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
            if default_lang is None:
                default_lang = next(iter(langs))
            elif default_lang not in langs:
                raise ValueError(
                    f"invalid default language '{default_lang}': "
                    f"not in {langs}")
            self._words[name] = self.cls_word(
                name, default_lang_query=lambda: self.default_lang, **word)
        self._langs_ = langs
        self.set_default_lang(default_lang)

    def set_default_lang(self, lang):
        """Change default language recursively for all words."""
        if lang not in self.langs:
            raise ValueError(
                f"unknown language '{lang}': not among {self.langs}")
        self._default_lang = lang

    #------------------------------------------------------------------

    def get(self, name, lang=None, ctx=None):
        try:
            word = self._words[name]
        except KeyError:
            s = f"unknown word: {name}"
            if f'{name}_' in type(self).__dict__:
                s = (
                    f"{s}; are you meaning to call "
                    f"`{type(self).__name__}.{name}_`?")
            raise ValueError(s)
        if lang is not None:
            word = word.get_in(lang)
        if ctx is not None:
            word = word.ctx(ctx)
        return word

    @property
    def default_lang(self):
        """Return the default language."""
        return self._default_lang

    @property
    def langs(self):
        """Return the languages the words are defined in."""
        return [lang for lang in self._langs_]

    #------------------------------------------------------------------

    def __repr__(self):
        s_name = f' ({self.name})' if self.name else ''
        return f'{len(self._words)} Words{s_name}: {self._words}'

    def __getitem__(self, key):

        name, lang, ctx = None, None, None

        if isinstance(key, str):
            # Single str element
            name = key

        elif not isinstance(key, tuple):
            # Error: Key neither str nor tuple
            raise ValueError(
                f"'{type(self).__name__}' object only subscriptable by one "
                f"or more objects of type 'str', not '{type(key).__name__}'")

        elif any(not isinstance(i, (str, type(None))) for i in key):
            # Error: Not all key tuple elements are str
            types = []
            for element in key:
                if not isinstance(element, str) and type(element) not in types:
                    types.append(type(element))
            if len(types) == 1:
                s_types = f"'{next(iter(types)).__name__}'"
            else:
                s_types = "[{}]".format(
                    ', '.join([f"'{t.__name__}'" for t in types]))
            raise ValueError(
                f"'{type(self).__name__}' object only subscriptable by one "
                f"or more objects of type 'str' or None, not {s_types}")

        elif len(key) == 2:
            # Tuple with two elements
            name, lang = key

        elif len(key) == 3:
            # Tuple with three elements
            name, lang, ctx = key

        else:
            # Error: Tuple with unexpected number of elements
            raise NotImplementedError(f"{len(key)} elements")

        return self.get(name, lang, ctx)
