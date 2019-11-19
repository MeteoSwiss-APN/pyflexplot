# -*- coding: utf-8 -*-
"""
Words.
"""
from .word import Word
from .exceptions import MissingWordError
from .exceptions import MissingLanguageError


#======================================================================


class Words:
    """A collection of words in different languages."""

    cls_word = Word

    def __init__(self, name=None, *, default_lang=None, **words_langs):
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

        self._langs = None
        self._words = {}
        for word_name, word_langs in words_langs.items():
            try:
                {**word_langs}
            except TypeError:
                raise ValueError(
                    f"word '{word_name}' not a dict: {word_langs}")
            self.add(word_name, **word_langs)

            if default_lang is None:
                default_lang = next(iter(self._langs))

        self.set_default_lang(default_lang)

    #------------------------------------------------------------------

    def add(self, name=None, **word_langs):
        """Add a word."""

        if name is None:
            name = next(iter(word_langs.values()))
            if not isinstance(name, str):
                try:
                    name = next(iter(name.values()))
                except Exception:
                    raise ValueError(
                        f"cannot derive name of {word_langs} from {name} "
                        f"of type {type(name).__name__}: neither a string "
                        f"nor a non-empty dict")

        if not isinstance(name, str):
            raise ValueError(
                f"argument `name`: expect type str, got {type(name).__name__}")

        langs = list(word_langs.keys())

        if self._langs is None:
            self._langs = langs
        elif set(langs) != set(self._langs):
            raise ValueError(
                f"word '{name}' not defined in all necessary languages: "
                f"set({langs}) != set({self._langs})")

        self._words[name] = self.cls_word(
            name, default_lang_query=lambda: self.default_lang, **word_langs)

    def set_default_lang(self, lang):
        """Change default language recursively for all words."""
        if lang and self.langs and lang not in self.langs:
            raise MissingLanguageError(f"{lang} not in {self.langs}")
        self._default_lang = lang

    #------------------------------------------------------------------

    def get(self, name, lang=None, ctx=None):
        try:
            word = self._words[name]
        except KeyError:
            raise MissingWordError(name)
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
        if not self._langs:
            return []
        return [lang for lang in self._langs]

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
