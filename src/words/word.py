"""TranslatedWord."""
# First-party
from srutils.str import capitalize
from srutils.str import check_is_valid_varname
from srutils.str import titlecase
from srutils.str import to_varname


class Word:
    """An individual word."""

    def __init__(self, s, lang, ctx=None):
        """Create an instance of ``Word``.

        Args:
            s (str): The word as a string.

            lang (str): Language in which the word is given.

            ctx (str, optional): Context in which the word is valid.

        """
        if not isinstance(s, str):
            raise ValueError(f"type of s is {type(s).__name__}, not str")
        self._s = s
        self.lang = lang
        self.ctx = ctx

    def __eq__(self, other):
        return str(self) == str(other)

    def __str__(self):
        return self._s

    def __hash__(self):
        return hash(str(self))

    def capital(self, *, all_=False, preserve=True):
        """Capitalize the first letter of the first or of each word.

        Already capitalized words are retained as such, in contrast to
        ``str.capitalize()``.

        Args:
            all_ (bool, optional): Whether to capitalize all words. Defaults to
                False.

            preserve (bool, optional): Whether to preserve capitalized letters.
                Defaults to True.

        """
        s = str(self)
        if not preserve:
            s = s.lower()
        if all_:
            return " ".join([capitalize(w) for w in s.split(" ")])
        words = s.split(" ")
        return " ".join([capitalize(words[0])] + words[1:])

    def title(self, *, preserve=True):
        if self.lang == "en":
            return titlecase(str(self), preserve=preserve)
        elif self.lang == "de":
            return self.capital(all_=False, preserve=preserve)
        else:
            raise NotImplementedError(f"{type(self).__name__}.title", self.lang)

    @property
    def s(self):
        return str(self)

    @property
    def c(self):
        """Shorthand to capitalize the first word."""
        return self.capital(all_=False, preserve=True)

    @property
    def C(self):  # pylint: disable=C0103  # invalid-name
        return self.capital(all_=True, preserve=True)

    @property
    def t(self):
        return self.title(preserve=True)


class TranslatedWord:
    """A word in one or more languages."""

    cls_word = Word

    def __init__(
        self, name=None, *, active_lang=None, active_lang_query=None, **translations
    ):
        """Create an instance of ``TranslatedWord``.

        Args:
            name (str, optional): Key to refer to the object in the code.
                Defaults to the first word defined in ``**langs``. If
                necessary, it is transformed into a valid Python variable name
                by replacing all spaces and dashes by underscores and dropping
                all other special characters.

            active_lang (str, optional): Active language. Overridden by
                ``active_lang_query`` if the latter is not None. Defaults to the
                first key in ``langs``.

            active_lang_query (callable, optional): Function to query the active
                language. Overrides ``active_lang``. Defaults to None.

            **translations (dict of str: str or (dict of str: str)): The word
                in different languages. The values are either strings for
                simple words, or dicts with context-specific variants of the
                word. In the latter case, the first entry is the default
                context.

        Example:
            >>> w = TranslatedWord(en='high school', de='Mittelschule')
            >>> w.name, str(w), w.de
            ('high_school', 'high school', 'Mittelschule')

        """
        self.name = None
        self._translations = {}

        self._check_langs(translations)

        # Set name of word (valid Python variable name)
        if name is None:
            name = next(iter(translations.values()))
            if isinstance(name, dict):
                name = next(iter(name.values()))
            name = to_varname(name).lower()
        else:
            check_is_valid_varname(name)
        self.name = name

        ctxs = self._collect_contexts(translations)

        # Define words with consistent contexts
        for lang, word in translations.items():
            if isinstance(word, str):
                word = self.cls_word(word, lang=lang)
            if isinstance(word, Word):
                word = {"*": word}
            elif isinstance(word, dict):
                if set(word.keys()) != set(ctxs) and "*" not in word:
                    raise ValueError(
                        f"word '{self.name}' in language '{lang}' must either "
                        f"be defined for all contexts {ctxs}, or for default "
                        f"context '*'"
                    )
            else:
                raise ValueError(
                    f"word '{self.name}' in language '{lang}' has "
                    f"unexpected type {type(word).__name__}"
                )
            self._translations[lang] = ContextWord(lang, **word)

        self.set_active_lang(lang=active_lang, query=active_lang_query)

    def _check_langs(self, translations):
        """Check validity of languages and words."""
        if not translations:
            raise ValueError("must pass the word in at least one language")
        for lang, word in translations.items():
            if lang in self.__dict__:
                # Name clash between language and class attribute
                raise ValueError("invalid language specifier", lang)
            if not isinstance(word, (Word, str, dict)):
                # Wrong word type
                raise ValueError("word must be of type str or dict", type(word), word)
            if not word:
                # Undefined word
                raise ValueError("empty word passed in language", word, lang)

    @staticmethod
    def _collect_contexts(translations):
        ctxs = []
        for word in translations.values():
            try:
                {**word}
            except TypeError:
                pass
            else:
                ctxs += [ctx for ctx in word.keys() if ctx not in ctxs]
        return ctxs

    def set_active_lang(self, lang=None, query=None):
        """Set the active language, either hard-coded or queryable.

        Args:
            lang (str, None): Default language. Overridden by ``query`` if the
                latter is not None. Defaults to the first key in
                ``TranslatedWord.langs``.

            query (callable, None): Function to query the active language.
                Overrides ``default``. Defaults to None.

        """
        if lang is None:
            lang = next(iter(self.langs))
        elif lang not in self.langs:
            raise ValueError(f"invalid language: {lang} (not among {self.langs})")
        self._active_lang = lang

        if query is not None and not callable(query):
            raise ValueError(f"query of type {type(query).__name__} not callable")
        self._active_lang_query = query

    @property
    def active_lang(self):
        """Get the active language."""
        if self._active_lang_query is not None:
            return self._active_lang_query()
        return self._active_lang

    def get(self):
        """Get word in the active language."""
        return self.get_in(None)

    def get_in(self, lang):
        """Get word in a certain language."""
        if lang is None:
            lang = self.active_lang
        try:
            return self._translations[lang]
        except KeyError as e:
            raise ValueError(
                f"word '{self.name}' not defined in language '{lang}', only in "
                f"{self.langs}"
            ) from e

    def ctx(self, *args, **kwargs):
        return self.get_in(self.active_lang).ctx(*args, **kwargs)

    @property
    def langs(self):
        """List of languages the word is defined in."""
        return list(self._translations.keys())

    def __str__(self):
        w = self.get().get()
        assert isinstance(w, Word)  # SR_TMP
        return w.s

    def __repr__(self):
        s_langs = ", ".join(
            [f"{lang}={repr(word)}" for lang, word in self._translations.items()]
        )
        return (
            f"{type(self).__name__}({self.name}, {s_langs}, lang='{self.active_lang}')"
        )

    def __eq__(self, other):
        return str(self) == str(other)

    def __getitem__(self, key):
        lang = key
        if lang is None:
            lang = self.active_lang
        return self.get_in(lang)


class ContextWord:
    """One or more variants of a word in a specific language."""

    cls_word = Word

    def __init__(self, lang, *, default_context=None, **variants):
        """Create an instance of ``ContextWord``.

        Args:
            lang (str): Language. Argument name ends with underscore to avoid
                conflict with possible context specifier 'lang'.

            default_context (str, optional): Default context specifier.
                Defaults to the first key in ``variants``.

            **variants (dict of str: str): One or more variants of a word. The
                keys constitute context specifiers, the values the respective
                word.

        """
        self.lang = lang
        self._variants = {}
        for ctx, word in variants.items():
            if not isinstance(word, (Word, str)):
                raise ValueError(f"invalid word: {word}")
            if not isinstance(word, Word):
                word = self.cls_word(word, lang=lang, ctx=ctx)
            self._variants[ctx] = word
        self.default_context = default_context or next(iter(self._variants))

    def get(self):
        """Return word in default context."""
        return self.ctx(None)

    def ctx(self, name, as_str=False):
        """Return the variant of the word in a specific context.

        Args:
            name (str): Name of the context (one of ``self.ctxs``).

            as_str (bool, optional): Return string. Defaults to False.

        """
        if name is None:
            name = self.default_context
        elif name not in self._variants:
            name = "*"
        word = self._variants[name]
        if as_str:
            assert isinstance(word, Word)  # SR_TMP
            return word
        return word

    def ctxs(self):
        """List of contexts."""
        return list(self._variants.keys())

    def __str__(self):
        w = self.ctx(self.default_context, as_str=True)
        assert isinstance(w, Word)  # SR_TMP
        return w.s

    def __repr__(self):
        s_variants = ", ".join([f"{k}={repr(v)}" for k, v in self._variants.items()])
        return f"{type(self).__name__}(lang='{self.lang}', {s_variants})"

    def __eq__(self, other):
        return str(self) == str(other)

    @property
    def s(self):
        return str(self)

    def capital(self, **kwargs):
        return self.get().capital(**kwargs)

    def title(self, **kwargs):
        return self.get().title(**kwargs)

    @property
    def c(self):
        return self.get().c

    @property
    def C(self):  # pylint: disable=C0103  # invalid-name
        return self.get().C

    @property
    def t(self):
        return self.get().t
