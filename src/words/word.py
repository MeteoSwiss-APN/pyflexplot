"""TranslatedWord."""
from __future__ import annotations

# Standard library
from typing import Any
from typing import Callable
from typing import cast
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Type
from typing import Union

# Third-party
from typing_extensions import Literal

# First-party
from srutils.exceptions import InvalidVariableNameError
from srutils.format import capitalize
from srutils.format import titlecase
from srutils.varname import VariableName

# SR_TODO Subclass ContextWord from Word and eliminate WordT
WordT = Union["Word", "ContextWord"]


class Word:
    """An individual word."""

    def __init__(self, s: str, lang: str, ctx: Optional[str] = None) -> None:
        """Create an instance of ``Word``.

        Args:
            s: The word as a string.

            lang: Language in which the word is given.

            ctx (optional): Context in which the word is valid.

        """
        if not isinstance(s, str):
            raise ValueError(f"type of s is {type(s).__name__}, not str")
        self._s: str = s
        self.lang: str = lang
        self.ctx: Optional[str] = ctx

    def __eq__(self, other: Any) -> bool:
        return str(self) == str(other)

    def __str__(self) -> str:
        return self._s

    def __hash__(self) -> int:
        return hash(str(self))

    def capital(self, *, all_: bool = False, preserve: bool = True) -> str:
        """Capitalize the first letter of the first or of each word.

        Already capitalized words are retained as such, in contrast to
        ``str.capitalize()``.

        Args:
            all_ (optional): Whether to capitalize all words.

            preserve (optional): Whether to preserve capitalized letters.

        """
        s = str(self)
        if not preserve:
            s = s.lower()
        if all_:
            return " ".join([capitalize(w) for w in s.split(" ")])
        words = s.split(" ")
        return " ".join([capitalize(words[0])] + words[1:])

    def title(self, *, preserve: bool = True) -> str:
        if self.lang == "en":
            return titlecase(str(self), preserve=preserve)
        elif self.lang == "de":
            return self.capital(all_=False, preserve=preserve)
        else:
            raise NotImplementedError(f"{type(self).__name__}.title", self.lang)

    @property
    def s(self) -> str:
        return str(self)

    @property
    def c(self) -> str:
        """Shorthand to capitalize the first word."""
        return self.capital(all_=False, preserve=True)

    @property
    # pylint: disable=C0103  # invalid-name
    def C(self) -> str:
        return self.capital(all_=True, preserve=True)

    @property
    def t(self) -> str:
        return self.title(preserve=True)


class TranslatedWord:
    """A word in one or more languages."""

    cls_word: Type[Word] = Word

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        active_lang: Optional[str] = None,
        active_lang_query: Optional[Callable[[], str]] = None,
        **translations: Union[WordT, str, Mapping[str, Union[WordT, str]]],
    ) -> None:
        """Create an instance of ``TranslatedWord``.

        Args:
            name (optional): Key to refer to the object in the code. Defaults to
                the first word defined in ``**langs``. If necessary, it is
                transformed into a valid Python variable name by replacing all
                spaces and dashes by underscores and dropping all other special
                characters.

            active_lang (optional): Active language. Overridden by
                ``active_lang_query`` if the latter is not None. Defaults to the
                first key in ``langs``.

            active_lang_query (optional): Function to query the active language.
                Overrides ``active_lang``.

            **translations: The word in different languages. The values are
                either strings for simple words, or dicts with context-specific
                variants of the word. In the latter case, the first entry is the
                default context.

        Example:
            >>> w = TranslatedWord(en='high school', de='Mittelschule')
            >>> w.name, str(w), w.de
            ('high_school', 'high school', 'Mittelschule')

        """
        orig_name = name
        self.name = None
        self._translations: dict[str, ContextWord] = {}

        self._check_langs(translations)

        # Set name of word (valid Python variable name)
        try:
            if name is None:
                tmp_name: Union[WordT, str, Mapping[str, Union[WordT, str]]] = next(
                    iter(translations.values())
                )
                if isinstance(tmp_name, Mapping):
                    tmp_name = next(iter(tmp_name.values()))
                name = VariableName(str(tmp_name)).format(lower=True)
            else:
                VariableName.check_valid(name)
        except InvalidVariableNameError as e:
            raise ValueError(orig_name) from e
        self.name = name

        ctxs = self._collect_contexts(translations)

        # Define words with consistent contexts
        for lang, word in translations.items():
            if isinstance(word, str):
                word = self.cls_word(word, lang=lang)
            if isinstance(word, ContextWord):
                # Note [2022-03-09]: Not sure what should happen in this case,
                # but it does not happen in the testsuite, anyway...
                raise NotImplementedError(f"word is of type ContextWord: '{word}'")
            elif isinstance(word, Word):
                word = {"*": word}
            elif isinstance(word, Mapping):
                if any(isinstance(v, ContextWord) for v in word.values()):
                    # Note [2022-03-09]: Not sure what should happen in this case,
                    # but it does not happen in the testsuite, anyway...
                    raise NotImplementedError(
                        f"word variant is of type ContextWord: '{word}'"
                    )
                word = cast(Mapping[str, Union[Word, str]], word)  # mypy
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

    def _check_langs(
        self,
        translations: Mapping[str, Union[WordT, str, Mapping[str, Union[WordT, str]]]],
    ) -> None:
        """Check validity of languages and words."""
        if not translations:
            raise ValueError("must pass the word in at least one language")
        for lang, word in translations.items():
            if lang in self.__dict__:
                # Name clash between language and class attribute
                raise ValueError("invalid language specifier", lang)
            if not isinstance(word, (Word, ContextWord, str, Mapping)):
                # Wrong word type
                raise ValueError(
                    f"word not str or dict-like: {word} {type(word).__name__}"
                )
            if not word:
                # Undefined word
                raise ValueError(f"empty word passed in language '{lang}'")

    @staticmethod
    def _collect_contexts(
        translations: Mapping[str, Union[WordT, str, Mapping[str, Union[WordT, str]]]]
    ) -> list[str]:
        ctxs: list[str] = []
        for word in translations.values():
            if isinstance(word, Mapping):
                ctxs += [ctx for ctx in word.keys() if ctx not in ctxs]
        return ctxs

    def set_active_lang(
        self, lang: Optional[str] = None, query: Callable[[], str] = None
    ) -> None:
        """Set the active language, either hard-coded or queryable.

        Args:
            lang (optional): Default language. Overridden by ``query`` if the
                latter is not None. Defaults to the first key in
                ``TranslatedWord.langs``.

            query (None): Function to query the active language. Overrides
                ``default``.

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
    def active_lang(self) -> str:
        """Get the active language."""
        if self._active_lang_query is not None:
            return self._active_lang_query()
        return self._active_lang

    def get(self) -> ContextWord:
        """Get word in the active language."""
        return self.get_in(None)

    def get_in(self, lang: Optional[str]) -> ContextWord:
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

    @overload
    def ctx(self, name: Optional[str], as_str: Literal[False] = False) -> Word:
        ...

    @overload
    def ctx(self, name: Optional[str], as_str: Literal[True]) -> str:
        ...

    def ctx(self, name, as_str=False):
        return self.get_in(self.active_lang).ctx(name, as_str)

    @property
    def langs(self) -> list[str]:
        """List of languages the word is defined in."""
        return list(self._translations.keys())

    def __str__(self) -> str:
        w = self.get().get()
        assert isinstance(w, Word)  # SR_TMP
        return w.s

    def __repr__(self) -> str:
        s_langs = ", ".join(
            [f"{lang}={repr(word)}" for lang, word in self._translations.items()]
        )
        return (
            f"{type(self).__name__}({self.name}, {s_langs}, lang='{self.active_lang}')"
        )

    def __eq__(self, other: Any) -> bool:
        return str(self) == str(other)

    def __getitem__(self, key: Optional[str]) -> ContextWord:
        lang = key
        if lang is None:
            lang = self.active_lang
        return self.get_in(lang)


# SR_TODO Make a subclass of Word (requires same interface)
class ContextWord:
    """One or more variants of a word in a specific language."""

    cls_word: Type[Word] = Word

    def __init__(
        self,
        lang: str,
        *,
        default_context: Optional[str] = None,
        **variants: Union[Word, str],
    ) -> None:
        """Create an instance of ``ContextWord``.

        Args:
            lang: Language. Argument name ends with underscore to avoid
                conflict with possible context specifier 'lang'.

            default_context (optional): Default context specifier. Defaults to
                the first key in ``variants``.

            **variants: One or more variants of a word. The keys constitute
                context specifiers, the values the respective word.

        """
        self.lang: str = lang
        self._variants: dict[str, Word] = {}
        for ctx, word in variants.items():
            if not isinstance(word, (Word, str)):
                raise ValueError(f"invalid word: {word}")
            if not isinstance(word, Word):
                word = self.cls_word(word, lang=lang, ctx=ctx)
            self._variants[ctx] = word
        self.default_context = default_context or next(iter(self._variants))

    def get(self) -> Word:
        """Return word in default context."""
        return self.ctx(None)

    def ctx(self, name: Optional[str], as_str: bool = False) -> Word:
        """Return the variant of the word in a specific context.

        Args:
            name: Name of the context (one of ``self.ctxs``).

            as_str (optional): Return string.

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

    def ctxs(self) -> list[str]:
        """List of contexts."""
        return list(self._variants.keys())

    def __str__(self) -> str:
        w = self.ctx(self.default_context, as_str=True)
        assert isinstance(w, Word)  # SR_TMP
        return w.s

    def __repr__(self) -> str:
        s_variants = ", ".join([f"{k}={repr(v)}" for k, v in self._variants.items()])
        return f"{type(self).__name__}(lang='{self.lang}', {s_variants})"

    def __eq__(self, other: Any) -> bool:
        return str(self) == str(other)

    @property
    def s(self) -> str:
        return str(self)

    def capital(self, **kwargs: Any) -> str:
        return self.get().capital(**kwargs)

    def title(self, **kwargs: Any) -> str:
        return self.get().title(**kwargs)

    @property
    def c(self) -> str:
        return self.get().c

    @property
    # pylint: disable=C0103  # invalid-name
    def C(self) -> str:
        return self.get().C

    @property
    def t(self) -> str:
        return self.get().t
