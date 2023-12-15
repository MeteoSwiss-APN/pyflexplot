"""Functions to convert from rotated to unrotated coordinates."""
# Third-party
import numpy as np


def latrot2lat(
    phirot: np.ndarray,
    rlarot: np.ndarray,
    polphi: float,
    polgam: float = 0,
) -> np.ndarray:
    rad_to_deg = 180 / np.pi
    deg_to_rad = np.pi / 180

    zsinpol = np.sin(deg_to_rad * polphi)
    zcospol = np.cos(deg_to_rad * polphi)

    zphis = deg_to_rad * phirot
    zrlas = np.where(rlarot > 180.0, rlarot - 360.0, rlarot)
    zrlas = deg_to_rad * zrlas

    zgam = deg_to_rad * polgam
    zarg = zsinpol * np.sin(zphis) + zcospol * np.cos(zphis) * (
        np.cos(zrlas) * np.cos(zgam) - np.sin(zgam) * np.sin(zrlas)
    )

    return rad_to_deg * np.arcsin(zarg)


def lonrot2lon(
    phirot: np.ndarray,
    rlarot: np.ndarray,
    polphi: float,
    pollam: float,
    polgam: float = 0,
) -> np.ndarray:
    rad_to_deg = 57.2957795
    deg_to_rad = 0.0174532925

    zsinpol = np.sin(deg_to_rad * polphi)
    zcospol = np.cos(deg_to_rad * polphi)
    zlampol = deg_to_rad * pollam
    zphis = deg_to_rad * phirot

    zrlas = np.where(rlarot > 180.0, rlarot - 360.0, rlarot)
    zrlas = deg_to_rad * zrlas

    zgam = deg_to_rad * polgam
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
    zarg2 = np.where(zarg2 == 0.0, 1.0e-20, zarg2)

    return rad_to_deg * np.arctan2(zarg1, zarg2)
