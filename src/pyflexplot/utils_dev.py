# -*- coding: utf-8 -*-
"""
Utils for development.
"""
import logging as log
import sys
import warnings
import IPython


def ipython(
        _globals_, _locals_, _msg_=None, _err_=66, _log_lvl_=log.INFO,
        _colors_=True, _bg_='light', _edit_mode_='vi'):
    """Drop into an iPython shell with all global and local variables.

    To pass the global and local variables, call with globals() and locals()
    as arguments, respectively. Notice how both functions must called in the
    call to ipython.

    To exit the program after leaving the iPython shell, pass an integer,
    which will be returned as the error code.

    To display a message when dropping into the iPython shell, pass a
    string after the error code.

    Examples
    --------
    >>> ipython(globals(), locals(), "I'm here!")

    """
    log.getLogger().setLevel(_log_lvl_)

    _kwargs_ = {'display_banner': None, 'editing_mode': _edit_mode_}

    if _colors_:
        if _bg_ == 'light':
            _kwargs_['colors'] = 'LightBG'
        elif _bg_ == 'dark':
            _kwargs_['colors'] = 'Linux'
        else:
            _kwargs_['colors'] = _colors_

    globals().update(_globals_)
    locals().update(_locals_)

    print('\n----------\n')
    if _msg_ is not None:
        print("\n{l}\n{m}\n{l}\n".format(l="*" * 60, m=_msg_))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        import IPython
        IPython.embed(**_kwargs_)

    if _err_ is not None:
        sys.exit(_err_)
