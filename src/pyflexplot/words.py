# -*- coding: utf-8 -*-
"""
Words.
"""
from words import Words

# yapf: disable

symbols = Words(
    ae          = {'': r'$\mathrm{\"a}$'},
    copyright   = {'': u'\u00a9'},
    oe          = {'': r'$\mathrm{\"o}$'},
    t0          = {'': r'$\mathrm{T_0}$'},
    ue          = {'': r'$\mathrm{\"u}$'},
)

e, d = 'en', 'de'
s = symbols
words = Words(
# A
accumulated_over        = { e: 'accumulated over',       d: f'akkumuliert {s["ue"]}ber'},
concentration           = { e: {'*': 'concentration', 'abbr': 'concentr.'},
                            d: {'*': 'Konzentration', 'abbr': 'Konzentr.'}},
activity_concentration  = { e: 'activity concentration', d: f'Aktivit{s["ae"]}tskonzentration'},
activity_concentr       = { e: 'activity concentr.',     d: f'Aktivit{s["ae"]}tskonzentr.'},
affected_area           = { e: 'affected area',          d: 'Beaufschlagtes Gebiet'},
at                      = { e: 'at',
                            d: {'level': 'auf', 'place': 'bei', 'time' : 'um'}},
averaged_over           = { e: 'averaged over',          d: f'gemittelt {s["ue"]}ber'},
# B
based_on                = { e: 'based on',               d: 'basierend auf'},
# C
# D
deposit_vel             = { e: 'deposit. vel.',          d: 'Deposit.-Geschw.'},
deposition              = { e: 'deposition',             d: 'Deposition'},
# E
end                     = { e: 'end',                    d: 'Ende'},
# F
flexpart                = { e: 'FLEXPART',               d: 'FLEXPART'},
# G
# H
half_life               = { e: 'half-life',              d: 'Halbwertszeit'},
height                  = { e: 'height',                 d: f'H{s["oe"]}he'},
# I
integrated              = { e: {'*': 'integrated', 'abbr': 'int.'},
                            d: {'*': 'integriert', 'abbr': 'int.',
                                'm': 'integrierter', 'f': 'integrierte',
                                'n': 'integriertes'}},
# J
# K
# L
latitude                = {e: 'latitude',               d: 'Breite'},
longitude               = {e: 'longitude',              d: f'L{s["ae"]}nge'},
# M
max                     = {e: 'max.',                   d: 'Max.'},
mch                     = {e: 'MeteoSwiss',             d: 'MeteoSchweiz'},
# N
# O
# P
# Q
# R
rate                    = {e: 'rate',                   d: 'Rate'},
release                 = {e: 'release',                d: 'Freisetzung'},
release_site            = {e: 'release site',           d: 'Abgabeort'},
# S
sediment_vel            = {e: 'sediment. vel.',         d: 'Sediment.-Geschw.'},
since                   = {e: 'since',                  d: 'seit'},
site                    = {e: 'site',                   d: 'Ort'},
start                   = {e: 'start',                  d: 'Start'},
substance               = {e: 'substance',              d: 'Substanz'},
summed_up_over          = {e: 'summed up over',         d: f'aufsummiert {s["ue"]}ber'},
surface_deposition      = {e: 'surface deposition',     d: 'Bodendeposition'},
# T
total_mass              = {e: 'total mass',             d: 'Totale Masse'},
# U
# V
# W
washout_coeff           = {e: 'washout coeff.',         d: 'Auswaschkoeff.'},
washout_exponent        = {e: 'washout exponent',       d: 'Auswaschexponent'},
# X
# Y
# Z
)

words.symbols = symbols  #SR_TMP

# yapf: enable
