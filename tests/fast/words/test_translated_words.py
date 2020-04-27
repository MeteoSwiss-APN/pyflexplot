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


class Test_ContextDependent_OneToMany:
    """Test words with context-dependency in one language."""

    ws = p_TranslatedWords(
        "words",
        {"at": {"en": "at", "de": {"place": "bei", "level": "auf"}}},
        active_lang="de",
    )

    def test_change_active_lang(self):
        ws = self.ws
        assert ws.active_lang == "de"
        assert ws.get("at") == "bei"
        ws.set_active_lang("en")
        assert ws.active_lang == "en"
        assert ws.get("at") == "at"

    def test_default_context(self):
        assert self.ws.get("at") == "bei"
        assert self.ws.get("at").get_in("de") == "bei"
        assert self.ws.get("at").get_in("en") == "at"
        assert self.ws.get("at", lang="de") == "bei"
        assert self.ws.get("at", lang="en") == "at"

    def test_explicit_context__arg(self):
        assert self.ws.get("at").ctx("place") == "bei"
        assert self.ws.get("at").ctx("level") == "auf"
        assert self.ws.get("at").get_in("de").ctx("place") == "bei"
        assert self.ws.get("at").get_in("de").ctx("level") == "auf"
        assert self.ws.get("at").get_in("en").ctx("place") == "at"
        assert self.ws.get("at").get_in("en").ctx("level") == "at"

    def test_explicit_context_one_args(self):
        assert self.ws.get("at", lang=None).ctx("place") == "bei"
        assert self.ws.get("at", lang=None).ctx("level") == "auf"
        assert self.ws.get("at", lang="de").ctx("place") == "bei"
        assert self.ws.get("at", lang="de").ctx("level") == "auf"
        assert self.ws.get("at", lang="en").ctx("place") == "at"
        assert self.ws.get("at", lang="en").ctx("level") == "at"

    def test_explicit_context_three_args(self):
        assert self.ws.get("at", ctx="place", lang=None) == "bei"
        assert self.ws.get("at", ctx="level", lang=None) == "auf"
        assert self.ws.get("at", ctx="place", lang="de") == "bei"
        assert self.ws.get("at", ctx="level", lang="de") == "auf"
        assert self.ws.get("at", ctx="place", lang="en") == "at"
        assert self.ws.get("at", ctx="level", lang="en") == "at"


class Test_ContextDependent_OneToMany_BracketInterface:
    """Test bracket interface for one-to-many context-dep. words."""

    ws = Test_ContextDependent_OneToMany.ws

    def test_active_lang_default_context(self):
        assert self.ws["at"] == "bei"
        assert self.ws["at", None] == "bei"
        assert self.ws["at", None, None] == "bei"

    def test_explicit_lang_default_context(self):
        assert self.ws["at", None, "en"] == "at"
        assert self.ws["at", None, "en"] == "at"

    def test_active_lang_explicit_context(self):
        assert self.ws["at", "level", None] == "auf"

    def test_explicit_lang_explicit_context(self):
        assert self.ws["at", "level", "en"] == "at"


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
        active_lang="de",
    )

    def test_change_active_lang(self):
        ws = self.ws
        assert ws.active_lang == "de"
        assert ws.get("integrated") == "integrierte"
        ws.set_active_lang("en")
        assert ws.active_lang == "en"
        assert ws.get("integrated") == "integrated"

    def test_default_context(self):
        assert self.ws.get("integrated") == "integrierte"
        assert self.ws.get("integrated").get_in("de") == "integrierte"
        assert self.ws.get("integrated").get_in("en") == "integrated"
        assert self.ws.get("integrated", lang="de") == "integrierte"
        assert self.ws.get("integrated", lang="en") == "integrated"

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
        assert self.ws.get("integrated", lang=None).ctx("*") == "integriert"
        assert self.ws.get("integrated", lang=None).ctx("f") == "integrierte"
        assert self.ws.get("integrated", lang=None).ctx("abbr") == "int."
        assert self.ws.get("integrated", lang="de").ctx("*") == "integriert"
        assert self.ws.get("integrated", lang="de").ctx("f") == "integrierte"
        assert self.ws.get("integrated", lang="de").ctx("abbr") == "int."
        assert self.ws.get("integrated", lang="en").ctx("*") == "integrated"
        assert self.ws.get("integrated", lang="en").ctx("f") == "integrated"
        assert self.ws.get("integrated", lang="en").ctx("abbr") == "int."

    def test_explicit_context_three_args(self):
        assert self.ws.get("integrated", lang=None, ctx="*") == "integriert"
        assert self.ws.get("integrated", lang=None, ctx="f") == "integrierte"
        assert self.ws.get("integrated", lang=None, ctx="abbr") == "int."
        assert self.ws.get("integrated", lang="de", ctx="*") == "integriert"
        assert self.ws.get("integrated", lang="de", ctx="f") == "integrierte"
        assert self.ws.get("integrated", lang="de", ctx="abbr") == "int."
        assert self.ws.get("integrated", lang="en", ctx="*") == "integrated"
        assert self.ws.get("integrated", lang="en", ctx="f") == "integrated"
        assert self.ws.get("integrated", lang="en", ctx="abbr") == "int."


class Test_ContextDependent_ManyToMany_BracketInterface:
    """Test bracket interface for many-to-many context-dep. words."""

    ws = Test_ContextDependent_ManyToMany.ws

    def test_active_lang_default_context(self):
        assert self.ws["integrated"] == "integrierte"
        assert self.ws["integrated", None] == "integrierte"
        assert self.ws["integrated", None, None] == "integrierte"

    def test_explicit_lang_default_context(self):
        assert self.ws["integrated", None, "en"] == "integrated"

    def test_active_lang_explicit_context(self):
        assert self.ws["integrated", "*"] == "integriert"
        assert self.ws["integrated", "f"] == "integrierte"
        assert self.ws["integrated", "*", None] == "integriert"
        assert self.ws["integrated", "f", None] == "integrierte"

    def test_explicit_lang_explicit_context(self):
        assert self.ws["integrated", "*", "en"] == "integrated"
        assert self.ws["integrated", "f", "en"] == "integrated"

    def test_format(self):
        assert self.ws["integrated", "*", "en"].s == "integrated"
        assert self.ws["integrated", "f", "en"].s == "integrated"
        assert self.ws["integrated", "*", "en"].c == "Integrated"
        assert self.ws["integrated", "f", "en"].c == "Integrated"
        assert self.ws["integrated", "*", "en"].C == "Integrated"
        assert self.ws["integrated", "f", "en"].C == "Integrated"
        assert self.ws["integrated", "*", "en"].t == "Integrated"
        assert self.ws["integrated", "f", "en"].t == "Integrated"


class Test_Interface_AddWords:

    sol = {
        ("train", None, "en"): "train",
        ("train", None, "de"): "Zug",
        ("at", "place", "en"): "at",
        ("at", "place", "de"): "bei",
        ("at", "time", "en"): "at",
        ("at", "time", "de"): "um",
        ("at", "level", "en"): "at",
        ("at", "level", "de"): "auf",
    }

    def check_ws(self, ws):
        for (name, ctx, lang), s in self.sol.items():
            assert ws[name, ctx, lang].s == s

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
    def test_init_empty_with_active_lang(self):
        TranslatedWords("name", active_lang="en")
