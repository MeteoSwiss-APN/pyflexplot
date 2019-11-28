# -*- coding: utf-8 -*-
"""
TranslatedWords.
"""
from words import Words
from words import TranslatedWords

symbols = Words(
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

s = symbols
words = TranslatedWords("words", {}, default_lang="en")
# A
words.add(en="accumulated over", de=f'akkumuliert {s["ue"]}ber')
words.add(
    en={"*": "concentration", "abbr": "concentr."},
    de={"*": "Konzentration", "abbr": "Konzentr."},
)
words.add(
    en={"*": "activity concentration", "abbr": "activity concentr."},
    de={
        "*": f'Aktivit{s["ae"]}tskonzentration',
        "abbr": f'Aktivit{s["ae"]}tskonzentr.',
    },
)
words.add(en="affected area", de="Beaufschlagtes Gebiet")
words.add(en="at", de={"level": "auf", "place": "bei", "time": "um"})
words.add(en="averaged over", de=f'gemittelt {s["ue"]}ber')
# B
words.add(en="based on", de="basierend auf")
# C
# D
deg_ = f"{s['deg']}{s['short_space']}"
words.add("degE", en=f"{deg_}E", de=f"{deg_}O")
words.add("degN", en=f"{deg_}N", de=f"{deg_}N")
words.add("degS", en=f"{deg_}S", de=f"{deg_}S")
words.add("degW", en=f"{deg_}W", de=f"{deg_}W")
words.add(
    en={"*": "deposition velocity", "abbr": "deposit. vel."},
    de={"*": "Depositionsgeschwindigkeit", "abbr": "Deposit.-Geschw."},
)
words.add(en="deposition", de="Deposition")
words.add(en="dry", de="trocken")
# E
words.add(en={"*": "east", "abbr": "E"}, de={"*": "Ost", "abbr": "O"})
words.add(en="end", de="Ende")
words.add(en="ensemble", de="Ensemble")
words.add(en="ensemble mean", de="Ensemble-Mittel")
# F
words.add(en="FLEXPART", de="FLEXPART")
# G
# H
words.add(en="half-life", de="Halbwertszeit")
words.add(en="height", de=f'H{s["oe"]}he')
# I
words.add(
    en={"*": "integrated", "abbr": "int."},
    de={
        "*": "integriert",
        "abbr": "int.",
        "m": "integrierter",
        "f": "integrierte",
        "n": "integriertes",
    },
)
# J
# K
# L
words.add(en="latitude", de="Breite")
words.add(en="longitude", de=f'L{s["ae"]}nge')
# M
words.add(en="m AGL", de=f'm {s["ue"]}.G.')
words.add(en="max.", de="Max.")
words.add(en={"*": "member", "pl": "members"}, de={"*": "Member", "pl": "Members"})
words.add(en="MeteoSwiss", de="MeteoSchweiz")
# N
words.add(en={"*": "north", "abbr": "N"}, de={"*": "Norden", "abbr": "N"})
words.add(en={"*": "number of", "abbr": "no."}, de={"*": "Anzahl", "abbr": "Anz."})
# O
# P
# Q
# R
words.add(en="rate", de="Rate")
words.add(en="release", de="Freisetzung")
words.add(en="release site", de="Abgabeort")
words.add(en="release start", de="Freisetzungsbeginn")
# S
words.add(
    en={"*": "sedimentation velocity", "abbr": "sediment. vel."},
    de={"*": "Sedimentiergeschwindigkeit", "abbr": "Sediment.-Geschw."},
)
words.add(en="since", de="seit")
words.add(en="site", de="Ort")
words.add(en={"*": "south", "abbr": "S"}, de={"*": "S{s['ue']}den", "abbr": "S"})
words.add(en="start", de="Start")
words.add(en="substance", de="Substanz")
words.add(en="summed up over", de=f'aufsummiert {s["ue"]}ber')
words.add(
    en={"*": "surface deposition", "abbr": "surface dep."},
    de={"*": "Bodendeposition", "abbr": "Bodendep."},
)
# T
words.add(
    en="threshold agreement", de=f"Grenzwert{s['ue']}bereinstimmung",
)
words.add(en="total", de={"*": "total", "m": "totaler", "f": "totale"})
words.add(en="total mass", de="Totale Masse")
# U
# V
# W
words.add(en="washout coeff.", de="Auswaschkoeff.")
words.add(en="washout exponent", de="Auswaschexponent")
words.add(en={"*": "west", "abbr": "W"}, de={"*": "Westen", "abbr": "W"})
words.add(en="wet", de="nass")
# X
# Y
# Z

words.symbols = symbols  # SR_TMP
