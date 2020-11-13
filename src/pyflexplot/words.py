"""TranslatedWords.

TODO:
    Split into primary and derived words/expressions to remove duplication

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
        "ge": r"$>$",
        "geq": r"$\geq$",
        "le": r"$<$",
        "leq": r"$\leq$",
        "oe": r"$\mathrm{\"o}$",
        "short_space": r"$\,$",
        "t0": r"$\mathrm{T_0}$",
        "ue": r"$\mathrm{\"u}$",
    },
)
s = SYMBOLS

WORDS = TranslatedWords("words", {})

# A
WORDS.add(en="accumulated over", de=f'akkumuliert {s["ue"]}ber')
WORDS.add(en="after", de="nach")
WORDS.add(
    en={
        "*": "air activity concentration",
        "abbr": "air activ. concentr.",
        "of": "of air activity concentration",
    },
    de={
        "*": f"Luftaktivit{s['ae']}tskonzentration",
        "abbr": "Luftaktiv.-Konzentr.",
        "of": f"der Luftaktivit{s['ae']}tskonzentration",
    },
)
WORDS.add(en="arrival", de="Ankunft")
WORDS.add(en="arrival time", de="Ankunftszeit")
WORDS.add(
    en={"*": "affected area", "of": "of affected area"},
    de={
        "*": "beaufschlagtes Gebiet",
        "g": "beaufschlagten Gebietes",
        "of": "des beaufschlagten Gebietes",
    },
)
WORDS.add(en="m AGL", de=f'm {s["ue"]}.G.')
WORDS.add(en="at", de={"level": "auf", "place": "in", "time": "um"})
WORDS.add(en="averaged over", de=f'gemittelt {s["ue"]}ber')
# B
WORDS.add(en="based on", de="basierend auf")
# C
WORDS.add(en="cloud", de="Wolke")
WORDS.add(
    en={"*": "cloud arrival time", "of": "of cloud arrival time"},
    de={"*": "Wolkenankunftszeit", "of": "der Wolkenankunftszeit"},
)
WORDS.add(en="cloud density", de="Wolkendichte")
WORDS.add(
    en={"*": "cloud departure time", "of": "of cloud departure time"},
    de={"*": "Wolkenabzugszeit", "of": "der Wolkenabzugszeit"},
)
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
WORDS.add(
    en={
        "*": "dry surface deposition",
        "abbr": "dry surface dep.",
        "of": "of dry surface deposition",
    },
    de={
        "*": "trockene Bodendeposition",
        "abbr": "trockene Bodendep.",
        "of": "der trockenen Bodendeposition",
    },
)
# E
WORDS.add(en={"*": "east", "abbr": "E"}, de={"*": "Ost", "abbr": "O"})
WORDS.add(en="end", de="Ende")
WORDS.add(en="ensemble", de="Ensemble")
WORDS.add(
    en={"*": "ensemble cloud arrival time", "abbr": "ens. cloud arr. t."},
    de={"*": "Ensemble-Wolkenankunftszeit", "abbr": "Ens.-Wolkenankunftsz."},
)
WORDS.add(
    en={"*": "ensemble cloud departure time", "abbr": "ens. cloud dep. t."},
    de={"*": "Ensemble-Wolkenabzugszeit", "abbr": "Ens.-Wolkenabzugsz."},
)
WORDS.add(en="ensemble maximum", de="Ensemble-Maximum")
WORDS.add(en="ensemble mean", de="Ensemble-Mittel")
WORDS.add(en="ensemble median", de="Ensemble-Median")
WORDS.add(
    en={
        "*": "ensemble median absolute deviation",
        "abbr": "ens. median abs. deviation",
    },
    de={
        "*": "Mittlere absolute Abweichung vom Ensemble-Median",
        "abbr": "Mittlere abs. Abw. vom Ens.-Median",
    },
)
WORDS.add(en="ensemble minimum", de="Ensemble-Minimum")
WORDS.add(en="ensemble percentile", de="Ensemble-Perzentil")
WORDS.add(en="ensemble standard deviation", de="Ensemble-Standardabweichung")
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
    en={
        "*": "incremental dry surface deposition",
        "abbr": "incr. dry surface dep.",
        "of": "of incremental dry surface deposition",
    },
    de={
        "*": "inkrementelle trockene Bodendeposition",
        "abbr": "inkr. trockene Bodendep.",
        "of": "der incrementellen trockenen Bodendeposition",
    },
)
WORDS.add(
    en={
        "*": "incremental total surface deposition",
        "abbr": "incr. total surface dep.",
        "of": "of incremental total surface deposition",
    },
    de={
        "*": "inkrementelle totale Bodendeposition",
        "abbr": "inkr. totale Bodendep.",
        "of": "der incrementellen total Bodendeposition",
    },
)
WORDS.add(
    en={
        "*": "incremental wet surface deposition",
        "abbr": "incr. wet surface dep.",
        "of": "of incremental wet surface deposition",
    },
    de={
        "*": "inkrementelle nasse Bodendeposition",
        "abbr": "inkr. nasse Bodendep.",
        "of": "der incrementellen nassen Bodendeposition",
    },
)
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
WORDS.add(
    en={
        "*": "integrated concentration",
        "abbr": "int. concentr.",
        "of": "of integrated concentration",
    },
    de={
        "*": "integrierte Konzentration",
        "abbr": "int. Konzentr.",
        "of": "der integrierten Konzentration",
    },
)
WORDS.add(
    en={
        "*": "integrated air activity concentration",
        "abbr": "int. air activ. concentr.",
        "of": "of integrated air activity concentration",
    },
    de={
        "*": f"integrierte Luftaktivit{s['ae']}tskonzentration",
        "abbr": "int. Luftaktiv.-Konzentr.",
        "of": f"der integrierten Luftaktivit{s['ae']}tskonzentration",
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
WORDS.add(
    en={
        "*": "median absolute deviation",
        "abbr": "median abs. deviation",
    },
    de={
        "*": "Mittlere absolute Abweichung vom Median",
        "abbr": "Mittlere abs. Abw. vom Median",
    },
)
WORDS.add(en={"*": "member", "pl": "members"}, de={"*": "Member", "pl": "Members"})
WORDS.add(en="MeteoSwiss", de="MeteoSchweiz")
WORDS.add(en={"*": "minimum", "abbr": "min."}, de={"*": "Minimum", "abbr": "min."})
# N
WORDS.add(en={"*": "north", "abbr": "N"}, de={"*": "Norden", "abbr": "N"})
WORDS.add(en={"*": "number of", "abbr": "no."}, de={"*": "Anzahl", "abbr": "Anz."})
# O
# WORDS.add(en="of", de={"*": "von", "fg": "der", "ng": "des"})
# P
WORDS.add(en="percent", de="Prozent")
WORDS.add(en="percentile", de="Perzentil")
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
WORDS.add(en="standard deviation", de="Standardabweichung")
WORDS.add(en="start", de="Start")
WORDS.add(en="substance", de="Substanz")
WORDS.add(en="summed over", de=f'aufsummiert {s["ue"]}ber')
WORDS.add(
    en={
        "*": "surface deposition",
        "abbr": "surface dep.",
        "of": "of surface deposition",
    },
    de={"*": "Bodendeposition", "abbr": "Bodendep.", "of": "der Bodendeposition"},
)
# T
WORDS.add(
    "th",
    en={
        "st": r"$^\mathrm{st}$",
        "nd": r"$^\mathrm{nd}$",
        "rd": r"$^\mathrm{rd}$",
        "*": r"$^\mathrm{th}$",
    },
    de="-tes",
)
WORDS.add(en="threshold", de="Grenzwert")
WORDS.add(en="time window", de="Zeitfenster")
WORDS.add(en="total", de={"*": "total", "m": "totaler", "f": "totale", "g": "totalen"})
WORDS.add(en="total mass", de="Totale Masse")
WORDS.add(
    en={
        "*": "total surface deposition",
        "abbr": "total surface dep.",
        "of": "of total surface deposition",
    },
    de={
        "*": "totale Bodendeposition",
        "abbr": "totale Bodendep.",
        "of": "der totalen Bodendeposition",
    },
)
# U
WORDS.add(en="until", de="bis")
# V
WORDS.add(en="variable", de="Variable")
# W
WORDS.add(en="washout coeff.", de="Auswaschkoeff.")
WORDS.add(en="washout exponent", de="Auswaschexponent")
WORDS.add(en={"*": "west", "abbr": "W"}, de={"*": "Westen", "abbr": "W"})
WORDS.add(en="wet", de={"*": "nass", "f": "nasse", "g": "nassen"})
WORDS.add(
    en={
        "*": "wet surface deposition",
        "abbr": "wet surface dep.",
        "of": "of wet surface deposition",
    },
    de={
        "*": "nasse Bodendeposition",
        "abbr": "nasse Bodendep.",
        "of": "der nassen Bodendeposition",
    },
)
# X
# Y
# Z
