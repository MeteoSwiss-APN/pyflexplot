# -*- coding: utf-8 -*-
"""
Word.
"""

__version__ = '0.1.0'


class Word:
    """A word in one or more languages."""

    def __init__(self, key=None, *, default=None, **langs):
        """Create an instance of ``Word``.

        Args:
            key (str, optional): Key to refer to the object in the code.
                Must be a valid Python variable name, i.e., not contain
                any spaces, dashes, etc. Defaults to the value of the
                first element in ``**langs``, unless that contains
                invalid characters, in which case a ``ValueError`` is
                raised.

            default (str, optional): Default language. Defaults to the
                first key in ``langs``.

            **langs (dict of str: str or (dict of str: str)): The word
                in different languages. The values are either strings
                for simple words, or dicts with context-specific variants
                of the word. In the latter case, the first entry is the
                default context.
        
        Example:
            >>> w = Word('high_school', en='high school', de='Mittelschule')
            >>> w.key
            'high_school'
            >>> str(w)
            'high school'
            >>> w.de
            'Mittelschule'

        """
        if not langs:
            raise ValueError('must pass the word in at least one language')

        self.key = key or next(iter(langs.values()))
        if not isinstance(self.key, str) or not self.key.isidentifier():
            raise ValueError(f"invalid key: {key}")

        self._langs = {}
        for lang, word in langs.items():
            if lang in self.__dict__:
                raise ValueError(f"invalid language specifier: {lang}")
            if isinstance(word, str):
                word = dict(default=word)
            self._langs[lang] = WordVariants(lang, **word)

        if default is None:
            default = next(iter(langs))
        self.set_default(default)

    def set_default(self, lang):
        """Set the default language."""
        if lang not in self.langs:
            raise ValueError(
                f"invalid default language: {lang} (not among {self.langs})")
        self.default = lang

    def in_(self, lang):
        """Get word in a certain language."""
        try:
            return self._langs[lang]
        except KeyError:
            raise ValueError(
                f"word '{self.key}' not defined in language '{lang}', "
                f"only in {self.langs}")

    def ctx(self, *args, **kwargs):
        return self.in_(self.default).ctx(*args, **kwargs)

    def __getattr__(self, key):
        return self.in_(key)

    def __str__(self):
        return str(self.in_(self.default))

    def __repr__(self):
        s_langs = ', '.join([f"{k}={repr(v)}" for k, v in self._langs.items()])
        return (
            f"{type(self).__name__}({self.key}, {s_langs}, "
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
        try:
            return self._variants[name]
        except KeyError:
            raise ValueError(f"invalid context specifier: {name}")

    def ctxs(self):
        """List of contexts."""
        return list(self._variants)

    def __str__(self):
        return self.ctx(self.default)

    def __repr__(self):
        s_variants = ', '.join(
            [f"{k}={repr(v)}" for k, v in self._variants.items()])
        return f"{type(self).__name__}(lang_='{self.lang}', {s_variants})"
