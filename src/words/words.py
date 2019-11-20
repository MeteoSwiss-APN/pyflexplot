# -*- coding: utf-8 -*-
"""
TranslatedWords.
"""
from srutils.str import to_varname

from .word import TranslatedWord
from .exceptions import MissingWordError
from .exceptions import MissingLanguageError

#======================================================================


class TranslatedWords:
    """A collection of words in different languages."""

    cls_word = TranslatedWord

    def __init__(self, name, words_langs=None, *, default_lang=None):
        """Create an instance of ``TranslatedWords``.

        Args:
            name (str, optional): Name of the words collection.
                Defaults to None.

            words_langs (dict of str: dict, optional): Words in one or
                more languages. The dict keys constitute the names of
                the words, and each value the definition of a word in
                one or more language. All words must be defined in the
                same languages. Defaults to None.

            default_lang (str, optional): Default language. Defaults to
                the first language in which the first word is defined.

        """
        self.name = name

        if words_langs is None:
            words_langs = {}

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
        name = self._prepare_word_name(name, word_langs)

        langs = list(word_langs.keys())

        if self._langs is None:
            self._langs = langs
        elif set(langs) != set(self._langs):
            raise ValueError(
                f"word '{name}' not defined in all necessary languages: "
                f"set({langs}) != set({self._langs})")

        self._words[name] = self.cls_word(
            name, default_lang_query=lambda: self.default_lang, **word_langs)

    def _prepare_word_name(self, name, word_langs):

        if name is None:
            name = next(iter(word_langs.values()))
            if isinstance(name, dict):
                try:
                    name = str(next(iter(name.values())))
                except Exception:
                    raise ValueError(
                        f"cannot derive name of {word_langs} from {name} "
                        f"of type {type(name).__name__}: neither a string "
                        f"nor a non-empty dict")
            name = to_varname(name)

        if not isinstance(name, str):
            raise ValueError(
                f"argument `name`: expect type str, got {type(name).__name__}")

        return name

    def set_default_lang(self, lang):
        """Change default language recursively for all words."""
        if lang and self.langs and lang not in self.langs:
            raise MissingLanguageError(f"{lang} not in {self.langs}")
        self._default_lang = lang

    #------------------------------------------------------------------

    def get(self, name, lang=None, ctx=None, *, chainable=True):

        try:
            word = self._words[name]
        except KeyError:
            raise MissingWordError(name)

        if chainable:
            if lang is None and ctx is None:
                return word
            elif ctx is None:
                return word.get_in(lang)
        return word.get_in(lang).ctx(ctx)

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
        return f'{len(self._words)} TranslatedWords{s_name}: {self._words}'

    def __getitem__(self, key):
        name, lang, ctx = None, None, None
        if not isinstance(key, tuple):
            key = (key,)
        elif len(key) not in [1, 2, 3]:
            raise KeyError(
                f"wrong number of key elements: {len(key)} not in [1, 2, 3]")
        return self.get(*key, chainable=False)


class Words(TranslatedWords):  #SR_TMP TODO remove parent

    def __init__(self, name, words):
        #SR_TMP<
        lang = 'None'
        words_langs = {name: {lang: word} for name, word in words.items()}
        super().__init__(name, words_langs, default_lang=lang)
        #SR_TMP>
