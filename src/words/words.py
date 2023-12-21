"""TranslatedWords."""
from __future__ import annotations

# Standard library
from pprint import pformat
from typing import Any
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import Union

# Third-party
from typing_extensions import Literal

# First-party
from srutils.varname import VariableName

# Local
from .exceptions import MissingLanguageError
from .exceptions import MissingWordError
from .word import ContextWord
from .word import TranslatedWord
from .word import Word
from .word import WordT


class TranslatedWords:
    """A collection of words in different languages."""

    cls_word: Type[TranslatedWord] = TranslatedWord

    def __init__(
        self,
        name: Optional[str],
        words_langs: Optional[
            Mapping[str, Mapping[str, Union[TranslatedWord, WordT, str]]]
        ] = None,
        *,
        active_lang: Optional[str] = None,
    ) -> None:
        """Create an instance of ``TranslatedWords``.

        Args:
            name (optional): Name of the words collection.

            words_langs (optional): Words in one or more languages. The dict keys
                constitute the names of the words, and each value the definition
                of a word in one or more language. All words must be defined in
                the same languages.

            active_lang (optional): Active language. Defaults to the first
                language in which the first word is defined.

        """
        self.name: Optional[str] = name
        if words_langs is None:
            words_langs = {}
        self._langs: list[str] = []
        self.words: dict[str, TranslatedWord] = {}
        for word_name, word_langs in words_langs.items():
            try:
                self.add(word_name, **word_langs)
            except TypeError as e:
                raise ValueError(f"word '{word_name}' not a dict: {word_langs}") from e
            if active_lang is None:
                active_lang = next(iter(self._langs))
        self._active_lang: str
        if active_lang is not None:
            self.set_active_lang(active_lang)

    def add(self, name: Optional[str] = None, **word_langs) -> TranslatedWord:
        """Add a word."""
        name = self._prepare_word_name(name, word_langs)
        langs: list[str] = list(word_langs.keys())
        if not self._langs:
            self._langs = langs
        elif set(langs) != set(self._langs):
            raise ValueError(
                f"word '{name}' not defined in all necessary languages: set({langs}) "
                f"!= set({self._langs})"
            )
        word: TranslatedWord = self.cls_word(
            name, active_lang_query=lambda: self.active_lang, **word_langs
        )
        self.words[name] = word
        return word

    @staticmethod
    def _prepare_word_name(
        name: Optional[str], word_langs: Mapping[str, Mapping[str, Union[Word, str]]]
    ) -> str:
        if name is None:
            tmp_name: Union[str, Mapping[str, Union[Word, str]]] = next(
                iter(word_langs.values())
            )
            if isinstance(tmp_name, Mapping):
                try:
                    tmp_name = str(next(iter(tmp_name.values())))
                except Exception as e:
                    raise ValueError(
                        f"cannot derive name of {word_langs} from {tmp_name} of type "
                        f"{type(tmp_name).__name__}: not a str or non-empty dict-like"
                    ) from e
            name = VariableName(tmp_name).format(
                lower=True, c_filter=lambda c: "_" if c in "- " else ""
            )
        if not isinstance(name, str):
            raise ValueError(
                f"argument `name`: expect type str, got {type(name).__name__}"
            )
        return name

    def set_active_lang(self, lang: str) -> None:
        """Change active language recursively for all words."""
        if lang and self.langs and lang not in self.langs:
            raise MissingLanguageError(f"{lang} not in {self.langs}")
        self._active_lang = lang

    @overload
    def get(
        self,
        name: str,
        ctx: Literal[None] = ...,
        lang: Literal[None] = ...,
        *,
        chain: Literal[True] = True,
    ) -> ContextWord:
        ...

    @overload
    def get(
        self,
        name: str,
        ctx: Optional[str] = ...,
        lang: Optional[str] = ...,
        *,
        chain: bool = ...,
    ) -> WordT:
        ...

    def get(self, name: str, ctx=None, lang=None, *, chain=True):
        try:
            word = self.words[name]
        except KeyError as e:
            raise MissingWordError(name) from e
        if chain:
            if ctx is None:
                if lang is None:
                    return word
                return word.get_in(lang)
        return word.get_in(lang).ctx(ctx)

    @property
    def active_lang(self) -> str:
        """Return the active language."""
        try:
            return self._active_lang
        except NameError as e:
            raise Exception("active language is not set") from e

    @property
    def langs(self) -> list[Optional[str]]:
        """Return the languages the words are defined in."""
        return list(self._langs)

    def __repr__(self) -> str:
        s_name = f" ({self.name})" if self.name else ""
        return f"{len(self.words)} TranslatedWords{s_name}: {self.words}"

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False
        return self.words == other.words

    def __hash__(self) -> int:
        return hash(pformat(self.words))

    def __getitem__(
        self,
        key: Union[
            str,
            Tuple[str],
            Tuple[str, Optional[str]],
            Tuple[str, Optional[str], Optional[str]],
        ],
    ) -> WordT:
        if not isinstance(key, tuple):
            key = (key,)
        name: str
        ctx: Optional[str]
        lang: Optional[str]
        if len(key) == 1:
            name, ctx, lang = key[0], None, None
        elif len(key) == 2:
            name, ctx, lang = key[0], key[1], None  # type: ignore
        elif len(key) == 3:
            name, ctx, lang = key[0], key[1], key[2]  # type: ignore
        else:
            raise KeyError(f"wrong number of key elements: {len(key)} not in [1, 2, 3]")
        return self.get(name, ctx=ctx, lang=lang, chain=False)


class Words(TranslatedWords):  # SR_TMP TODO Remove parent or invert inheritance order
    def __init__(self, name: str, words: Mapping[str, Union[str, Word]]) -> None:
        """Create an instance of ``Words``."""
        # SR_TMP <
        lang = "None"
        words_langs = {name: {lang: word} for name, word in words.items()}
        super().__init__(name, words_langs, active_lang=lang)
        # SR_TMP >
