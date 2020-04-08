# -*- coding: utf-8 -*-
"""
Tests for module ``words.word``.
"""
# Standard library
import functools

# Third-party
import pytest  # type: ignore

# First-party
from srutils.testing import property_obj
from words import TranslatedWords
from words.exceptions import MissingWordError

p_TranslatedWords = functools.partial(property_obj, TranslatedWords)


class Test_Basic:
    """Test for simple words."""

    ws = p_TranslatedWords(
        "words",
        {
            "train": {"en": "train", "de": "Zug"},
            "high_school": {"en": "high school", "de": "Mittelschule"},
        },
    )

    def test_default(self):
        assert self.ws.get("train") == "train"
        assert self.ws.get("high_school") == "high school"

    def test_langs(self):
        assert self.ws.get("train").get_in("en") == "train"
        assert self.ws.get("high_school").get_in("en") == "high school"
        assert self.ws.get("train").get_in("de") == "Zug"
        assert self.ws.get("high_school").get_in("de") == "Mittelschule"

    def test_format(self):
        assert self.ws.get("high_school").get_in("en").s == "high school"
        assert self.ws.get("high_school").get_in("en").c == "High school"
        assert self.ws.get("high_school").get_in("en").C == "High School"
        assert self.ws.get("high_school").get_in("en").t == "High School"
        assert self.ws.get("high_school").get_in("de").s == "Mittelschule"
        assert self.ws.get("high_school").get_in("de").c == "Mittelschule"
        assert self.ws.get("high_school").get_in("de").C == "Mittelschule"
        assert self.ws.get("high_school").get_in("de").t == "Mittelschule"


class Test_Basic_BracketsInterface:
    """Test simple words with brackets interface."""

    ws = Test_Basic.ws

    def test_default(self):
        assert self.ws["train"] == "train"
        assert self.ws["high_school"] == "high school"

    def test_default_lang_none_bracket(self):
        assert self.ws["train", None] == "train"
        assert self.ws["high_school", None] == "high school"

    def test_lang_bracket(self):
        assert self.ws["train", "en"] == "train"
        assert self.ws["train", "de"] == "Zug"
        assert self.ws["high_school", "en"] == "high school"
        assert self.ws["high_school", "de"] == "Mittelschule"

    def test_format(self):
        assert self.ws["high_school", "en"].s == "high school"
        assert self.ws["high_school", "de"].s == "Mittelschule"
        assert self.ws["high_school", "en"].c == "High school"
        assert self.ws["high_school", "de"].c == "Mittelschule"
        assert self.ws["high_school", "en"].C == "High School"
        assert self.ws["high_school", "de"].C == "Mittelschule"
        assert self.ws["high_school", "en"].t == "High School"
        assert self.ws["high_school", "de"].t == "Mittelschule"


class Test_ContextDependent_OneToMany:
    """Test words with context-dependency in one language."""

    ws = p_TranslatedWords(
        "words",
        {"at": {"en": "at", "de": {"place": "bei", "level": "auf"}}},
        default_lang="de",
    )

    def test_change_default_lang(self):
        ws = self.ws
        assert ws.default_lang == "de"
        assert ws.get("at") == "bei"
        ws.set_default_lang("en")
        assert ws.default_lang == "en"
        assert ws.get("at") == "at"

    def test_default_context(self):
        assert self.ws.get("at") == "bei"
        assert self.ws.get("at").get_in("de") == "bei"
        assert self.ws.get("at").get_in("en") == "at"
        assert self.ws.get("at", "de") == "bei"
        assert self.ws.get("at", "en") == "at"

    def test_explicit_context__arg(self):
        assert self.ws.get("at").ctx("place") == "bei"
        assert self.ws.get("at").ctx("level") == "auf"
        assert self.ws.get("at").get_in("de").ctx("place") == "bei"
        assert self.ws.get("at").get_in("de").ctx("level") == "auf"
        assert self.ws.get("at").get_in("en").ctx("place") == "at"
        assert self.ws.get("at").get_in("en").ctx("level") == "at"

    def test_explicit_context_one_args(self):
        assert self.ws.get("at", None).ctx("place") == "bei"
        assert self.ws.get("at", None).ctx("level") == "auf"
        assert self.ws.get("at", "de").ctx("place") == "bei"
        assert self.ws.get("at", "de").ctx("level") == "auf"
        assert self.ws.get("at", "en").ctx("place") == "at"
        assert self.ws.get("at", "en").ctx("level") == "at"

    def test_explicit_context_three_args(self):
        assert self.ws.get("at", None, "place") == "bei"
        assert self.ws.get("at", None, "level") == "auf"
        assert self.ws.get("at", "de", "place") == "bei"
        assert self.ws.get("at", "de", "level") == "auf"
        assert self.ws.get("at", "en", "place") == "at"
        assert self.ws.get("at", "en", "level") == "at"


class Test_ContextDependent_OneToMany_BracketInterface:
    """Test bracket interface for one-to-many context-dep. words."""

    ws = Test_ContextDependent_OneToMany.ws

    def test_default_lang_default_context(self):
        assert self.ws["at"] == "bei"
        assert self.ws["at", None] == "bei"
        assert self.ws["at", None, None] == "bei"

    def test_explicit_lang_default_context(self):
        assert self.ws["at", "en"] == "at"
        assert self.ws["at", "en", None] == "at"

    def test_default_lang_explicit_context(self):
        assert self.ws["at", None, "level"] == "auf"

    def test_explicit_lang_explicit_context(self):
        assert self.ws["at", "en", "level"] == "at"


class Test_ContextDependent_ManyToMany:
    """Test words with context-dependency in both languages."""

    ws = p_TranslatedWords(
        "words",
        {
            "integrated": {
                "en": {"*": "integrated", "abbr": "int."},
                "de": {"f": "integrierte", "*": "integriert", "abbr": "int."},
            }
        },
        default_lang="de",
    )

    def test_change_default_lang(self):
        ws = self.ws
        assert ws.default_lang == "de"
        assert ws.get("integrated") == "integrierte"
        ws.set_default_lang("en")
        assert ws.default_lang == "en"
        assert ws.get("integrated") == "integrated"

    def test_default_context(self):
        assert self.ws.get("integrated") == "integrierte"
        assert self.ws.get("integrated").get_in("de") == "integrierte"
        assert self.ws.get("integrated").get_in("en") == "integrated"
        assert self.ws.get("integrated", "de") == "integrierte"
        assert self.ws.get("integrated", "en") == "integrated"

    def test_explicit_context_one_arg(self):
        assert self.ws.get("integrated").ctx("*") == "integriert"
        assert self.ws.get("integrated").ctx("f") == "integrierte"
        assert self.ws.get("integrated").ctx("abbr") == "int."
        assert self.ws.get("integrated").get_in("de").ctx("*") == "integriert"
        assert self.ws.get("integrated").get_in("de").ctx("f") == "integrierte"
        assert self.ws.get("integrated").get_in("de").ctx("abbr") == "int."
        assert self.ws.get("integrated").get_in("en").ctx("*") == "integrated"
        assert self.ws.get("integrated").get_in("en").ctx("f") == "integrated"
        assert self.ws.get("integrated").get_in("en").ctx("abbr") == "int."

    def test_explicit_context_two_args(self):
        assert self.ws.get("integrated", None).ctx("*") == "integriert"
        assert self.ws.get("integrated", None).ctx("f") == "integrierte"
        assert self.ws.get("integrated", None).ctx("abbr") == "int."
        assert self.ws.get("integrated", "de").ctx("*") == "integriert"
        assert self.ws.get("integrated", "de").ctx("f") == "integrierte"
        assert self.ws.get("integrated", "de").ctx("abbr") == "int."
        assert self.ws.get("integrated", "en").ctx("*") == "integrated"
        assert self.ws.get("integrated", "en").ctx("f") == "integrated"
        assert self.ws.get("integrated", "en").ctx("abbr") == "int."

    def test_explicit_context_three_args(self):
        assert self.ws.get("integrated", None, "*") == "integriert"
        assert self.ws.get("integrated", None, "f") == "integrierte"
        assert self.ws.get("integrated", None, "abbr") == "int."
        assert self.ws.get("integrated", "de", "*") == "integriert"
        assert self.ws.get("integrated", "de", "f") == "integrierte"
        assert self.ws.get("integrated", "de", "abbr") == "int."
        assert self.ws.get("integrated", "en", "*") == "integrated"
        assert self.ws.get("integrated", "en", "f") == "integrated"
        assert self.ws.get("integrated", "en", "abbr") == "int."


class Test_ContextDependent_ManyToMany_BracketInterface:
    """Test bracket interface for many-to-many context-dep. words."""

    ws = Test_ContextDependent_ManyToMany.ws

    def test_default_lang_default_context(self):
        assert self.ws["integrated"] == "integrierte"
        assert self.ws["integrated", None] == "integrierte"
        assert self.ws["integrated", None, None] == "integrierte"

    def test_explicit_lang_default_context(self):
        assert self.ws["integrated", "en"] == "integrated"
        assert self.ws["integrated", "en", None] == "integrated"

    def test_default_lang_explicit_context(self):
        assert self.ws["integrated", None, "*"] == "integriert"
        assert self.ws["integrated", None, "f"] == "integrierte"

    def test_explicit_lang_explicit_context(self):
        assert self.ws["integrated", "en", "*"] == "integrated"
        assert self.ws["integrated", "en", "f"] == "integrated"

    def test_format(self):
        assert self.ws["integrated", "en", "*"].s == "integrated"
        assert self.ws["integrated", "en", "f"].s == "integrated"
        assert self.ws["integrated", "en", "*"].c == "Integrated"
        assert self.ws["integrated", "en", "f"].c == "Integrated"
        assert self.ws["integrated", "en", "*"].C == "Integrated"
        assert self.ws["integrated", "en", "f"].C == "Integrated"
        assert self.ws["integrated", "en", "*"].t == "Integrated"
        assert self.ws["integrated", "en", "f"].t == "Integrated"


class Test_Interface_AddWords:

    sol = {
        ("train", "en", None): "train",
        ("train", "de", None): "Zug",
        ("at", "en", "place"): "at",
        ("at", "de", "place"): "bei",
        ("at", "en", "time"): "at",
        ("at", "de", "time"): "um",
        ("at", "en", "level"): "at",
        ("at", "de", "level"): "auf",
    }

    def check_ws(self, ws):
        for (name, lang, ctx), s in self.sol.items():
            assert ws[name, lang, ctx].s == s

    def test_fail(self):
        """Ensure that ``_test_ws`` fails with wrong input."""
        ws = TranslatedWords("words")
        with pytest.raises(MissingWordError):
            self.check_ws(ws)

    def test_init(self):
        ws = TranslatedWords(
            "words",
            {
                "train": {"en": "train", "de": "Zug"},
                "at": {
                    "en": "at",
                    "de": {"place": "bei", "time": "um", "level": "auf"},
                },
            },
        )
        self.check_ws(ws)

    def test_add_explicit_name(self):
        """Use ``TranslatedWords.add`` with  names specified explicitly."""
        ws = TranslatedWords("words")
        ws.add("train", en="train", de="Zug")
        ws.add("at", en="at", de={"place": "bei", "time": "um", "level": "auf"})
        self.check_ws(ws)

    def test_add_implicit_name(self):
        """Use ``TranslatedWords.add`` with names derived from first variant."""
        ws = TranslatedWords("name")
        ws.add(en="train", de="Zug")
        ws.add(en="at", de={"place": "bei", "time": "um", "level": "auf"})
        self.check_ws(ws)


class Test_Interface_Various:
    def test_init_empty_with_default_lang(self):
        TranslatedWords("name", default_lang="en")
