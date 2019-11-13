#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some testing utils.
"""

def property_obj(cls, *args, **kwargs):
    """Define a class property creating a given object on-the-fly.

    The purpose of creating the object on-the-fly in a property method
    is to isolate any errors during instatiation in the test methods
    using the property. This prevents the whole test suite from being
    aborted, as it would if the object were defined as a simple class
    attribute and it's instatiation failed -- instead, only the tests
    attempting to use the object will fail.

    And yes, this is indeed merely a convenience function to save two
    lines of code wherever it is used. :-)

    Usage:
        The following class definitions are equivalent:

        >>> class C1:
        ...     @property
        ...     def w(self):
        ...         return Word(en='train', de='Zug')

        >>> class C2:
        ...     w = property_word(en='train', de='Zug')

    """
    def create_obj(self):
        return cls(*args, **kwargs)
    return property(create_obj)
