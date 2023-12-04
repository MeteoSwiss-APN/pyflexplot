"""Functions to convert from rotated to unrotated coordinates."""
# Standard library
from typing import Optional

# Third-party
import numpy as np


def latrot2lat(
    phirot: np.ndarray,
    rlarot: np.ndarray,
    polphi: float,
    polgam: Optional[float] = None,
) -> np.ndarray:
    zrpi18 = 57.2957795
    zpir18 = 0.0174532925

    zsinpol = np.sin(zpir18 * polphi)
    zcospol = np.cos(zpir18 * polphi)

    zphis = zpir18 * phirot
    zrlas = np.where(rlarot > 180.0, rlarot - 360.0, rlarot)
    zrlas = zpir18 * zrlas

    if polgam is not None:
        zgam = zpir18 * polgam
        zarg = zsinpol * np.sin(zphis) + zcospol * np.cos(zphis) * (
            np.cos(zrlas) * np.cos(zgam) - np.sin(zgam) * np.sin(zrlas)
        )
    else:
        zarg = zcospol * np.cos(zphis) * np.cos(zrlas) + zsinpol * np.sin(
            zphis
        )  # noqa: E501

    return zrpi18 * np.arcsin(zarg)


def lonrot2lon(
    phirot: np.ndarray,
    rlarot: np.ndarray,
    polphi: float,
    pollam: float,
    polgam: Optional[float] = None,
) -> np.ndarray:
    zrpi18 = 57.2957795
    zpir18 = 0.0174532925

    zsinpol = np.sin(zpir18 * polphi)
    zcospol = np.cos(zpir18 * polphi)
    zlampol = zpir18 * pollam
    zphis = zpir18 * phirot

    zrlas = np.where(rlarot > 180.0, rlarot - 360.0, rlarot)
    zrlas = zpir18 * zrlas

    if polgam is not None:
        zgam = zpir18 * polgam
        zarg1 = np.sin(zlampol) * (
            -zsinpol
            * np.cos(zphis)
            * (np.cos(zrlas) * np.cos(zgam) - np.sin(zrlas) * np.sin(zgam))
            + zcospol * np.sin(zphis)
        ) - np.cos(zlampol) * np.cos(zphis) * (
            np.sin(zrlas) * np.cos(zgam) + np.cos(zrlas) * np.sin(zgam)
        )

        zarg2 = np.cos(zlampol) * (
            -zsinpol
            * np.cos(zphis)
            * (np.cos(zrlas) * np.cos(zgam) - np.sin(zrlas) * np.sin(zgam))
            + zcospol * np.sin(zphis)
        ) + np.sin(zlampol) * np.cos(zphis) * (
            np.sin(zrlas) * np.cos(zgam) + np.cos(zrlas) * np.sin(zgam)
        )
    else:
        zarg1 = np.sin(zlampol) * (
            -zsinpol * np.cos(zrlas) * np.cos(zphis)
            + zcospol * np.sin(zphis)  # noqa: E501
        ) - np.cos(zlampol) * np.sin(zrlas) * np.cos(zphis)

        zarg2 = np.cos(zlampol) * (
            -zsinpol * np.cos(zrlas) * np.cos(zphis)
            + zcospol * np.sin(zphis)  # noqa: E501
        ) + np.sin(zlampol) * np.sin(zrlas) * np.cos(zphis)

    zarg2 = np.where(zarg2 == 0.0, 1.0e-20, zarg2)

    return zrpi18 * np.arctan2(zarg1, zarg2)
