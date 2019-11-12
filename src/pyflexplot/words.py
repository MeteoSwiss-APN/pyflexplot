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
#
at                = {e: 'at',                 d: {'level' : 'auf',
                                                  'place' : 'bei',
                                                  'time'  : 'um'}},
accumulated_over  = {e: 'accumulated_over',   d: f'akkumuliert {s.ue}ber'},
averaged_over     = {e: 'averaged over',      d: f'gemittelt {s.ue}ber'},
based_on          = {e: 'based on',           d: 'basierend auf'},
deposit_vel       = {e: 'deposit. vel.',      d: 'Deposit.-Geschw.'},
end               = {e: 'end',                d: 'Ende'},
flexpart          = {e: 'FLEXPART',           d: 'FLEXPART'},
half_life         = {e: 'half-life',          d: 'Halbwertszeit'},
height            = {e: 'height',             d: f'H{s.oe}he'},
latitude          = {e: 'latitude',           d: 'Breite'},
longitude         = {e: 'longitude',          d: f'L{s.ae}nge'},
max               = {e: 'max.',               d: 'Max.'},
mch               = {e: 'MeteoSwiss',         d: 'MeteoSchweiz'},
rate              = {e: 'rate',               d: 'Rate'},
release           = {e: 'release',            d: 'Freisetzung'},
release_site      = {e: 'release site',       d: 'Abgabeort'},
sediment_vel      = {e: 'sediment. vel.',     d: 'Sediment.-Geschw.'},
since             = {e: 'since',              d: 'seit'},
site              = {e: 'site',               d: 'Ort'},
start             = {e: 'start',              d: 'Start'},
substance         = {e: 'substance',          d: 'Substanz'},
summed_up_over    = {e: 'summed up over',     d: f'aufsummiert {s.ue}ber'},
total_mass        = {e: 'total mass',         d: 'Totale Masse'},
washout_coeff     = {e: 'washout coeff.',     d: 'Auswaschkoeff.'},
washout_exponent  = {e: 'washout exponent',   d: 'Auswaschexponent'},
)

words.symbols = symbols  #SR_TMP

# yapf: enable
