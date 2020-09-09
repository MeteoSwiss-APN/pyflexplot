# pylint: disable=E1101  # no-member (logging.verbose, logging.VERBOSE)
"""
Preset setup files.
"""
# Standard library
import logging
from typing import Optional


def get_log_level() -> int:
    return logging.getLogger().level


def set_log_level(verbosity: int) -> None:
    if verbosity <= 0:
        logging.getLogger().setLevel(logging.INFO)
    elif verbosity == 1:
        logging.getLogger().setLevel(logging.VERBOSE)  # type: ignore
    elif verbosity >= 2:
        logging.getLogger().setLevel(logging.DEBUG)


def log(
    *,
    err: Optional[str] = None,
    wrn: Optional[str] = None,
    inf: Optional[str] = None,
    vbs: Optional[str] = None,
    dbg: Optional[str] = None,
) -> None:
    """Log the message at the lowest available level.

    If case of multiple messages, only that at the lowest level is logged,
    whereby ``inf`` (info) > ``vbs`` (verbose) > ``dbg`` (debug), i.e., if all
    are passed, ``dbg`` wins.

    """
    if err is not None:
        err = f"error: {err}"
        logging.error(err)
    if wrn is not None:
        wrn = f"warning: {wrn}"
        logging.error(wrn)
    level = get_log_level()
    if level <= logging.DEBUG and dbg is not None:
        logging.debug(dbg)
        return
    if level <= logging.VERBOSE and vbs is not None:  # type: ignore
        logging.verbose(vbs)  # type: ignore
        return
    if level <= logging.INFO and inf is not None:
        logging.info(inf)
        return


def add_logging_level(name: str, level: int) -> None:
    """Add additional logging level.

    src: https://stackoverflow.com/a/55049399/4419816

    """

    def new_log(msg, *args, **kwargs):
        if logging.getLogger().isEnabledFor(level):
            logging.log(level, msg, *args, **kwargs)

    logging.addLevelName(level, "VERBOSE")
    setattr(logging, name.upper(), level)
    setattr(logging.Logger, name, new_log)
    setattr(logging, name, new_log)
