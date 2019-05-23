# -*- coding: utf-8 -*-
"""
Utils for development.
"""
import sys
import warnings
import IPython


def ipython(_globals_, _locals_, _msg_=None, _err_=66):
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        import IPython
        print('\n----------\n')
        globals().update(_globals_)
        locals().update(_locals_)
        if _msg_ is not None:
            print("\n{l}\n{m}\n{l}\n".format(l="*" * 60, m=_msg_))
        IPython.embed(display_banner=None)
        if _err_ is not None:
            sys.exit(_err_)
