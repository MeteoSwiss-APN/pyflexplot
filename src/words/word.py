# -*- coding: utf-8 -*-
"""
Word.
"""
from .utils import to_varname


class Word:
    """A word in one or more languages."""

    def __init__(
            self, name=None, *, default=None, default_query=None, **langs):
        """Create an instance of ``Word``.

        Args:
            name (str, optional): Key to refer to the object in the
                code. Defaults to the first word defined in ``**langs``.
                If necessary, it is transformed into a valid Python
                variable name by replacing all spaces and dashes by
                underscores and dropping all other special characters.

            default (str, optional): Default language.  Overridden by
                ``default_query`` if the latter is not None. Defaults
                to the first key in ``langs``.

            default_query (callable, optional): Function to query the
                default language. Overrides ``default``. Defaults to
                None.

            **langs (dict of str: str or (dict of str: str)): The word
                in different languages. The values are either strings
                for simple words, or dicts with context-specific
                variants of the word. In the latter case, the first
                entry is the default context.

        Example:
            >>> w = Word(en='high school', de='Mittelschule')
            >>> w.name, str(w), w.de
            ('high_school', 'high school', 'Mittelschule')

        """
        self._check_langs(langs)

        # Set name of word (valid Python variable name)
        if name is None:
            name = next(iter(langs.values()))
            if isinstance(name, dict):
                name = next(iter(name.values()))
        self.name = to_varname(name)

        ctxs = self._collect_contexts(langs)

        # Define words with consistent contexts
        self._langs = {}
        for lang, word in langs.items():
            if isinstance(word, str):
                word = {'*': word}
            elif set(word.keys()) != set(ctxs) and '*' not in word:
                raise ValueError(
                    f"word 'self.name' in '{lang}' must either be defined "
                    f"for all contexts {ctxs}, or for default context '*'")
            self._langs[lang] = WordVariants(lang, **word)

        self.set_default(default, default_query)

    def _check_langs(self, langs):
        """Check validity of languages and words."""

        if not langs:
            raise ValueError('must pass the word in at least one language')

        for lang, word in langs.items():

            if lang in self.__dict__:
                # Name clash between language and class attribute
                raise ValueError(f"invalid language specifier: {lang}")

            if not isinstance(word, (str, dict)):
                # Wrong word type
                raise ValueError(
                    f"word must be of type str or dict, not "
                    f"{type(word).__name__}: {word}")

            if not word:
                # Undefined word
                raise ValueError(
                    f"empty word passed in language '{lang}': {word}")

    def _collect_contexts(self, langs):
        ctxs = []
        for lang, word in langs.items():
            if not isinstance(word, str):
                ctxs += [ctx for ctx in word.keys() if ctx not in ctxs]
        return ctxs

    def set_default(self, lang=None, query=None):
        """Set the default language, either hard-coded or queriable.

        Args:
            lang (str, None): Default language. Overridden by ``query``
                if the latter is not None. Defaults to the first key in
                ``Word.langs``.

            query (callable, None): Function to query the default
                language. Overrides ``default``. Defaults to None.

        """
        if lang is None:
            lang = next(iter(self.langs))
        elif lang not in self.langs:
            raise ValueError(
                f"invalid default language: {lang} (not among {self.langs})")
        self._default = lang

        if query is not None and not callable(query):
            raise ValueError(
                f"query of type {type(query).__name__} not callable")
        self._default_query = query

    @property
    def default(self):
        """Get the default language."""
        if self._default_query is not None:
            return self._default_query()
        return self._default

    def in_(self, lang):
        """Get word in a certain language."""
        try:
            return self._langs[lang]
        except KeyError:
            raise ValueError(
                f"word '{self.name}' not defined in language '{lang}', "
                f"only in {self.langs}")

    def ctx(self, *args, **kwargs):
        return self.in_(self.default).ctx(*args, **kwargs)

    def __getattr__(self, name):
        return self.in_(name)

    def __str__(self):
        return str(self.in_(self.default))

    def __repr__(self):
        s_langs = ', '.join([f"{k}={repr(v)}" for k, v in self._langs.items()])
        return (
            f"{type(self).__name__}({self.name}, {s_langs}, "
            f"default='{self.default}')")

    @property
    def langs(self):
        """List of languages the word is defined in."""
        return list(self._langs.keys())


class WordVariants:
    """One or more variants of a word in a specific language."""

    def __init__(self, lang_=None, *, default_=None, **variants):
        """Create an instance of ``WordVariants``.

        Args:
            lang_ (str, optional): Language. Defaults to None.
                Argument name ends with underscore to avoid conflict
                with possible context specifier 'lang'.

            default_ (str, optional): Default context specifier.
                Defaults to the first key in ``variants``.

            **variants (dict of str: str): One or more variants of a
                word. The keys constitute context specifiers, the
                values the respective word.

        """
        self.lang = lang_

        self._variants = {}
        for ctx, word in variants.items():
            if not isinstance(word, str):
                raise ValueError(f"invalid word: {word}")
            self._variants[ctx] = word

        self.default = next(iter(self._variants))

    def ctx(self, name):
        """Return the variant of the word in a specific context.

        Args:
            name (str): Name of the context (one of ``self.ctxs``).

        """
        if name is None:
            name = self.default
        try:
            return self._variants[name]
        except KeyError:
            return self._variants['*']

    def ctxs(self):
        """List of contexts."""
        return list(self._variants)

    def s(self):
        return str(self)

    def __str__(self):
        return self.ctx(self.default)

    def __repr__(self):
        s_variants = ', '.join(
            [f"{k}={repr(v)}" for k, v in self._variants.items()])
        return f"{type(self).__name__}(lang_='{self.lang}', {s_variants})"
