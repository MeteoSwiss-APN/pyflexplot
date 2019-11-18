# -*- coding: utf-8 -*-
"""
Word.
"""
from srutils.str import capitalize
from srutils.str import titlecase
from srutils.str import to_varname

class Word:
    """A word in one or more languages."""

    def __init__(
            self, name=None, *, default_lang=None, default_lang_query=None,
            **langs):
        """Create an instance of ``Word``.

        Args:
            name (str, optional): Key to refer to the object in the
                code. Defaults to the first word defined in ``**langs``.
                If necessary, it is transformed into a valid Python
                variable name by replacing all spaces and dashes by
                underscores and dropping all other special characters.

            default_lang (str, optional): Default language.  Overridden
                by ``default_lang_query`` if the latter is not None.
                Defaults to the first key in ``langs``.

            default_lang_query (callable, optional): Function to query
                the default language. Overrides ``default_lang``.
                Defaults to None.

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
            self._langs[lang] = ContextWord(lang, **word)

        self.set_default_lang(lang=default_lang, query=default_lang_query)

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

    #------------------------------------------------------------------

    def set_default_lang(self, lang=None, query=None):
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
        self._default_lang = lang

        if query is not None and not callable(query):
            raise ValueError(
                f"query of type {type(query).__name__} not callable")
        self._default_lang_query = query

    @property
    def default_lang(self):
        """Get the default language."""
        if self._default_lang_query is not None:
            return self._default_lang_query()
        return self._default_lang

    def get_in(self, lang):
        """Get word in a certain language."""
        if lang is None:
            lang = self.default_lang
        try:
            return self._langs[lang]
        except KeyError:
            raise ValueError(
                f"word '{self.name}' not defined in language '{lang}', "
                f"only in {self.langs}")

    def ctx(self, *args, **kwargs):
        return self.get_in(self.default_lang).ctx(*args, **kwargs)

    @property
    def langs(self):
        """List of languages the word is defined in."""
        return list(self._langs.keys())

    #------------------------------------------------------------------

    def capital(self, all=False, preserve=True):
        """Capitalize the first letter of the first or of each word.

        Already capitalized words are retained as such, in contrast to
        ``str.capitalize()``.

        Args:
            all (bool, optional): Whether to capitalize all words.
                Defaults to False.

            preserve (bool, optional): Whether to preserve capitalized
                letters. Defaults to True.

        """
        s = str(self)
        if not preserve:
            s = s.lower()
        if all:
            return ' '.join([capitalize(w) for w in s.split(' ')])
        start, rest = s.split(' ', 1)
        return f'{capitalize(start)} {rest}'

    def title(self, preserve=True):
        if self.lang == 'de':
            return self.capital(all=False, preserve=preserve)
        elif self.lang == 'en':
            return titlecase(self, preserve=preserve)

    #------------------------------------------------------------------

    @property
    def s(self):
        return str(self)

    @property
    def c(self):
        """Shorthand to capitalize the first word."""
        return self.capital(all=False, preserve=True)

    @property
    def C(self):
        return self.capital(all=True, preserve=True)

    @property
    def t(self):
        return self.title(preserve=True)

    #------------------------------------------------------------------

    def __str__(self):
        return str(self.get_in(self.default_lang))

    def __repr__(self):
        s_langs = ', '.join([f"{k}={repr(v)}" for k, v in self._langs.items()])
        return (
            f"{type(self).__name__}({self.name}, {s_langs}, "
            f"default_lang='{self.default_lang}')")

    def __eq__(self, other):
        return str(self) == str(other)

    def __getitem__(self, key):
        lang = key
        return self.get_in(lang)


class ContextWord(Word):
    """One or more variants of a word in a specific language."""

    def __init__(self, lang=None, *, default_context=None, **variants):
        """Create an instance of ``ContextWord``.

        Args:
            lang (str, optional): Language. Defaults to None.
                Argument name ends with underscore to avoid conflict
                with possible context specifier 'lang'.

            default_context (str, optional): Default context specifier.
                Defaults to the first key in ``variants``.

            **variants (dict of str: str): One or more variants of a
                word. The keys constitute context specifiers, the
                values the respective word.

        """
        self.lang = lang

        self._variants = {}
        for ctx, word in variants.items():
            if not isinstance(word, str):
                raise ValueError(f"invalid word: {word}")
            self._variants[ctx] = word
        self.default_context = next(iter(self._variants))

    def ctx(self, name):
        """Return the variant of the word in a specific context.

        Args:
            name (str): Name of the context (one of ``self.ctxs``).

        """
        if name is None:
            name = self.default_context
        try:
            return self._variants[name]
        except KeyError:
            return self._variants['*']

    def ctxs(self):
        """List of contexts."""
        return list(self._variants)

    def __str__(self):
        return self.ctx(self.default_context)

    def __repr__(self):
        s_variants = ', '.join(
            [f"{k}={repr(v)}" for k, v in self._variants.items()])
        return f"{type(self).__name__}(lang='{self.lang}', {s_variants})"

    def __eq__(self, other):
        return str(self) == str(other)
