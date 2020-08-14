# -*- coding: utf-8 -*-
"""
TranslatedWords.
"""
# First-party
from words import TranslatedWords
from words import Words

SYMBOLS = Words(
    "symbols",
    {
        "ae": r"$\mathrm{\"a}$",
        "copyright": "\u00a9",
        "deg": r"$^\circ$",
        "geq": r"$\geq$",
        "oe": r"$\mathrm{\"o}$",
        "t0": r"$\mathrm{T_0}$",
        "short_space": r"$\,$",
        "ue": r"$\mathrm{\"u}$",
    },
)
s = SYMBOLS

WORDS = TranslatedWords("words", {})

# A
WORDS.add(en="accumulated over", de=f'akkumuliert {s["ue"]}ber')
WORDS.add(
    en={"*": "activity concentration", "abbr": "activity concentr."},
    de={
        "*": f'Aktivit{s["ae"]}tskonzentration',
        "abbr": f'Aktivit{s["ae"]}tskonzentr.',
    },
)
WORDS.add(en="arrival", de="Ankunft")
WORDS.add(en="arrival time", de="Ankunftszeit")
WORDS.add(en="affected area", de="Beaufschlagtes Gebiet")
WORDS.add(en="m AGL", de=f'm {s["ue"]}.G.')
WORDS.add(en="at", de={"level": "auf", "place": "in", "time": "um"})
WORDS.add(en="averaged over", de=f'gemittelt {s["ue"]}ber')
# B
WORDS.add(en="based on", de="basierend auf")
# C
WORDS.add(en="cloud", de="Wolke")
WORDS.add(en="cloud arrival probability", de="Wolkenankunftswahrscheinlichkeit")
WORDS.add(en="cloud arrival time", de="Wolkenankunftszeit")
WORDS.add(en="cloud density", de="Wolkendichte")
WORDS.add(en="cloud departure probability", de="Wolkenabzugswahrscheinlichkeit")
WORDS.add(en="cloud departure time", de="Wolkenabzugszeit")
WORDS.add(en="cloud occurrence probability", de="Wolkenauftretenswahrscheinlichkeit")
WORDS.add(
    en={"*": "cloud probability", "abbr": "cloud prob."},
    de={"*": "Wolkenwahrscheinlichkeit", "abbr": "Wolkenwahrsch."},
)
WORDS.add(
    en={"*": "cloud threshold", "abbr": "cloud thresh."},
    de={"*": "Wolkengrenzwert", "abbr": "Wolkengrenzw."},
)
WORDS.add(
    en={"*": "concentration", "abbr": "concentr."},
    de={"*": "Konzentration", "abbr": "Konzentr."},
)
WORDS.add(en="control run", de="Kontrolllauf")
# D
WORDS.add(en="data", de="Daten")
WORDS.add(en="departure", de="Abzug")
WORDS.add(en="departure time", de="Abzugszeit")
deg_ = f"{s['deg']}{s['short_space']}"
WORDS.add("degE", en=f"{deg_}E", de=f"{deg_}O")
WORDS.add("degN", en=f"{deg_}N", de=f"{deg_}N")
WORDS.add("degS", en=f"{deg_}S", de=f"{deg_}S")
WORDS.add("degW", en=f"{deg_}W", de=f"{deg_}W")
WORDS.add(
    en={"*": "deposition velocity", "abbr": "deposit. vel."},
    de={"*": "Depositionsgeschwindigkeit", "abbr": "Deposit.-Geschw."},
)
WORDS.add(en="deposition", de="Deposition")
WORDS.add(en="dry", de={"*": "trocken", "f": "trockene", "g": "trockenen"})
# E
WORDS.add(en={"*": "east", "abbr": "E"}, de={"*": "Ost", "abbr": "O"})
WORDS.add(en="end", de="Ende")
WORDS.add(en="ensemble", de="Ensemble")
WORDS.add(en="ensemble maximum", de="Ensemble-Maximum")
WORDS.add(en="ensemble mean", de="Ensemble-Mittel")
WORDS.add(en="ensemble median", de="Ensemble-Median")
WORDS.add(en="ensemble minimum", de="Ensemble-Minimum")
WORDS.add(
    en={"*": "ensemble variable", "abbr": "ens. variable"},
    de={"*": "Ensemblevariable", "abbr": "Ens.-Variable"},
)
# F
WORDS.add(en="field", de="Feld")
WORDS.add(en="FLEXPART", de="FLEXPART")
WORDS.add(en="from now", de="ab jetzt")
# G
# H
WORDS.add(en="half-life", de="Halbwertszeit")
WORDS.add(en="height", de=f'H{s["oe"]}he')
WORDS.add(
    en={"*": "hour", "pl": "hours", "abbr": "h"},
    de={"*": "Stunde", "pl": "Stunden", "abbr": "h"},
)
# I
WORDS.add(en="in", de="in")
WORDS.add(en="input variable", de="Inputvariable")
WORDS.add(
    en={"*": "integrated", "abbr": "int."},
    de={
        "*": "integriert",
        "abbr": "int.",
        "m": "integrierter",
        "f": "integrierte",
        "n": "integriertes",
        "g": "integrierten",
    },
)
# J
# K
# L
WORDS.add(en="latitude", de="Breite")
WORDS.add(en="lead time", de="Vorhersagezeit")
WORDS.add(en="level", de="Level")
WORDS.add(en="longitude", de=f'L{s["ae"]}nge')
# M
WORDS.add(en="m AGL", de=f'm {s["ue"]}.G.')
WORDS.add(en={"*": "maximum", "abbr": "max."}, de={"*": "Maximum", "abbr": "Max."})
WORDS.add(en="mean", de="Mittel")
WORDS.add(en="median", de="Median")
WORDS.add(en={"*": "member", "pl": "members"}, de={"*": "Member", "pl": "Members"})
WORDS.add(en="MeteoSwiss", de="MeteoSchweiz")
WORDS.add(
    en={"*": "minimum", "abbr": "min."}, de={"*": "Minimum", "abbr": "min."},
)
# N
WORDS.add(en={"*": "north", "abbr": "N"}, de={"*": "Norden", "abbr": "N"})
WORDS.add(en={"*": "number of", "abbr": "no."}, de={"*": "Anzahl", "abbr": "Anz."})
# O
WORDS.add(en="of", de={"*": "von", "fg": "der"})
# P
WORDS.add(en="percent", de="Prozent")
WORDS.add(en="probability", de="Wahrscheinlichkeit")
# Q
# R
WORDS.add(en="rate", de="Rate")
WORDS.add(en="release", de="Freisetzung")
WORDS.add(en="release site", de="Abgabeort")
WORDS.add(en="release start", de="Freisetzungsbeginn")
# S
WORDS.add(
    en={"*": "sedimentation velocity", "abbr": "sediment. vel."},
    de={"*": "Sedimentiergeschwindigkeit", "abbr": "Sediment.-Geschw."},
)
WORDS.add(en="since", de="seit")
WORDS.add(en="site", de="Ort")
WORDS.add(en={"*": "south", "abbr": "S"}, de={"*": "S{s['ue']}den", "abbr": "S"})
WORDS.add(en="start", de="Start")
WORDS.add(en="substance", de="Substanz")
WORDS.add(en="summed over", de=f'aufsummiert {s["ue"]}ber')
WORDS.add(
    en={"*": "surface deposition", "abbr": "surface dep."},
    de={"*": "Bodendeposition", "abbr": "Bodendep."},
)
# T
WORDS.add(en="threshold", de="Grenzwert")
WORDS.add(en="total", de={"*": "total", "m": "totaler", "f": "totale", "g": "totalen"})
WORDS.add(en="total mass", de="Totale Masse")
WORDS.add(en="time window", de="Zeitfenster")
# U
WORDS.add(en="until", de="bis")
# V
WORDS.add(en="variable", de="Variable")
# W
WORDS.add(en="washout coeff.", de="Auswaschkoeff.")
WORDS.add(en="washout exponent", de="Auswaschexponent")
WORDS.add(en={"*": "west", "abbr": "W"}, de={"*": "Westen", "abbr": "W"})
WORDS.add(en="wet", de={"*": "nass", "f": "nasse", "g": "nassen"})
# X
# Y
# Z
