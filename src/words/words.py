# -*- coding: utf-8 -*-
"""
TranslatedWords.
"""
# Standard library
from pprint import pformat

# First-party
from srutils.str import to_varname

# Local
from .exceptions import MissingLanguageError
from .exceptions import MissingWordError
from .word import TranslatedWord


class TranslatedWords:
    """A collection of words in different languages."""

    cls_word = TranslatedWord

    def __init__(self, name, words_langs=None, *, active_lang=None):
        """Create an instance of ``TranslatedWords``.

        Args:
            name (str, optional): Name of the words collection. Defaults to
                None.

            words_langs (dict of str: dict, optional): Words in one or more
                languages. The dict keys constitute the names of the words, and
                each value the definition of a word in one or more language.
                All words must be defined in the same languages. Defaults to
                None.

            active_lang (str, optional): Active language. Defaults to the first
                language in which the first word is defined.

        """
        self.name = name

        if words_langs is None:
            words_langs = {}

        self._langs = None
        self.words = {}
        for word_name, word_langs in words_langs.items():
            try:
                {**word_langs}
            except TypeError:
                raise ValueError(f"word '{word_name}' not a dict: {word_langs}")
            self.add(word_name, **word_langs)

            if active_lang is None:
                active_lang = next(iter(self._langs))

        self.set_active_lang(active_lang)

    def add(self, name=None, **word_langs):
        """Add a word."""
        name = self._prepare_word_name(name, word_langs)
        langs = list(word_langs.keys())
        if self._langs is None:
            self._langs = langs
        elif set(langs) != set(self._langs):
            raise ValueError(
                f"word '{name}' not defined in all necessary languages: set({langs}) "
                f"!= set({self._langs})"
            )
        word = self.cls_word(
            name, active_lang_query=lambda: self.active_lang, **word_langs
        )
        self.words[name] = word
        return word

    def _prepare_word_name(self, name, word_langs):

        if name is None:
            name = next(iter(word_langs.values()))
            if isinstance(name, dict):
                try:
                    name = str(next(iter(name.values())))
                except Exception:
                    raise ValueError(
                        f"cannot derive name of {word_langs} from {name} of type "
                        f"{type(name).__name__}: neither a string nor a non-empty dict"
                    )
            name = to_varname(
                name, filter_invalid=lambda c: "_" if c in "- " else ""
            ).lower()

        if not isinstance(name, str):
            raise ValueError(
                f"argument `name`: expect type str, got {type(name).__name__}"
            )

        return name

    def set_active_lang(self, lang):
        """Change active language recursively for all words."""
        if lang and self.langs and lang not in self.langs:
            raise MissingLanguageError(f"{lang} not in {self.langs}")
        self._active_lang = lang

    def get(self, name, lang=None, ctx=None, *, chainable=True):

        try:
            word = self.words[name]
        except KeyError:
            raise MissingWordError(name)

        if chainable:
            if lang is None and ctx is None:
                return word
            elif ctx is None:
                return word.get_in(lang)
        return word.get_in(lang).ctx(ctx)

    @property
    def active_lang(self):
        """Return the active language."""
        return self._active_lang

    @property
    def langs(self):
        """Return the languages the words are defined in."""
        if not self._langs:
            return []
        return list(self._langs)

    def __repr__(self):
        s_name = f" ({self.name})" if self.name else ""
        return f"{len(self.words)} TranslatedWords{s_name}: {self.words}"

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return self.words == other.words

    def __hash__(self):
        return hash(pformat(self.words))

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        elif len(key) not in [1, 2, 3]:
            raise KeyError(f"wrong number of key elements: {len(key)} not in [1, 2, 3]")
        return self.get(*key, chainable=False)


class Words(TranslatedWords):  # SR_TMP TODO remove parent
    def __init__(self, name, words):
        # SR_TMP <
        lang = "None"
        words_langs = {name: {lang: word} for name, word in words.items()}
        super().__init__(name, words_langs, active_lang=lang)
        # SR_TMP >
