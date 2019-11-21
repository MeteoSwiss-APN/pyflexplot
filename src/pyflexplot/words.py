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
        "oe": r"$\mathrm{\"o}$",
        "t0": r"$\mathrm{T_0}$",
        "ue": r"$\mathrm{\"u}$",
    },
)

s = symbols
words = TranslatedWords("words", {}, default_lang="en")
# A
words.add("accumulated_over", en="accumulated over", de=f'akkumuliert {s["ue"]}ber')
words.add(
    "concentration",
    en={"*": "concentration", "abbr": "concentr."},
    de={"*": "Konzentration", "abbr": "Konzentr."},
)
words.add(
    "activity_concentration",
    en="activity concentration",
    de=f'Aktivit{s["ae"]}tskonzentration',
)
words.add(
    "activity_concentr", en="activity concentr.", de=f'Aktivit{s["ae"]}tskonzentr.'
)
words.add("affected_area", en="affected area", de="Beaufschlagtes Gebiet")
words.add("at", en="at", de={"level": "auf", "place": "bei", "time": "um"})
words.add("averaged_over", en="averaged over", de=f'gemittelt {s["ue"]}ber')
# B
words.add("based_on", en="based on", de="basierend auf")
# C
# D
words.add("deposit_vel", en="deposit. vel.", de="Deposit.-Geschw.")
words.add("deposition", en="deposition", de="Deposition")
words.add("dry", en="dry", de="trocken")
# E
words.add("end", en="end", de="Ende")
words.add("ensemble", en="ensemble", de="Ensemble")
words.add("ensemble_mean", en="ensemble mean", de="Ensemble-Mittel")
# F
words.add("flexpart", en="FLEXPART", de="FLEXPART")
# G
# H
words.add("half_life", en="half-life", de="Halbwertszeit")
words.add("height", en="height", de=f'H{s["oe"]}he')
# I
words.add(
    "integrated",
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
words.add("latitude", en="latitude", de="Breite")
words.add("longitude", en="longitude", de=f'L{s["ae"]}nge')
# M
words.add("m_agl", en="m AGL", de=f'm {s["ue"]}.G.')
words.add("max", en="max.", de="Max.")
words.add("mch", en="MeteoSwiss", de="MeteoSchweiz")
# N
# O
# P
# Q
# R
words.add("rate", en="rate", de="Rate")
words.add("release", en="release", de="Freisetzung")
words.add("release_site", en="release site", de="Abgabeort")
# S
words.add(
    "sediment_vel",
    en={"*": "sedimentation velocity", "abbr": "sediment. vel."},
    de={"*": "Sedimentiergeschwindigkeit", "abbr": "Sediment.-Geschw."},
)
words.add("since", en="since", de="seit")
words.add("site", en="site", de="Ort")
words.add("start", en="start", de="Start")
words.add("substance", en="substance", de="Substanz")
words.add("summed_up_over", en="summed up over", de=f'aufsummiert {s["ue"]}ber')
words.add("surface_deposition", en="surface deposition", de="Bodendeposition")
# T
words.add(
    "threshold_agreement",
    en="threshold agreement",
    de=f'Grenzwert{s["ue"]}bereinstimmung',
)
words.add("total", en="total", de={"*": "total", "m": "totaler", "f": "totale"})
words.add("total_mass", en="total mass", de="Totale Masse")
# U
# V
# W
words.add("washout_coeff", en="washout coeff.", de="Auswaschkoeff.")
words.add("washout_exponent", en="washout exponent", de="Auswaschexponent")
words.add("wet", en="wet", de="nass")
# X
# Y
# Z

words.symbols = symbols  # SR_TMP
