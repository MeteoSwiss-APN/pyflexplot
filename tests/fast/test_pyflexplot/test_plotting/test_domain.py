"""Tests for module ``pyflexplot.plotting.domain``."""
# Standard library
import dataclasses as dc
from typing import List
from typing import Tuple

# Third-party
import numpy as np
import pytest

# First-party
from pyflexplot.plotting.domain import CloudDomain
from pyflexplot.plotting.domain import find_gaps


class Test_GlobalCloudDomain:
    """Test cloud domain on global grid."""

    class General:
        """General cloud that may be comprised of multiple sub-clouds."""

        lat = np.arange(-85, 85.1, 10)
        lon = np.arange(-175, 175.1, 10)

        def get_mask(self, bboxes):
            mask = np.zeros([self.lat.size, self.lon.size], np.bool_)
            for idx_lllon, idx_urlon, idx_lllat, idx_urlat in bboxes:
                if idx_lllon < 0:
                    mask[idx_lllat:idx_urlat, idx_lllon:] = True
                    mask[idx_lllat:idx_urlat, :idx_urlon] = True
                else:
                    mask[idx_lllat:idx_urlat, idx_lllon:idx_urlon] = True
            assert mask.any(), f"invalid bboxes (empty mask): {bboxes}"
            return mask

        def get_domain(self, *mask_args, **config):
            mask = self.get_mask(*mask_args)
            return CloudDomain(self.lat, self.lon, mask=mask, config=config)

        def get_bbox_corners(self, *mask_args, **config):
            domain = self.get_domain(*mask_args, **config)
            return domain.get_bbox_extent()

    class Continuous(General):
        """Continuous cloud."""

        def get_mask(self, *bbox_idcs):
            return super().get_mask([bbox_idcs])

    class Test_ZonalNearPole(Continuous):
        """Zonally extended cloud near a pole."""

        def test_north_no_aspect(self):
            """Near north pole, w/o specified aspect ratio."""
            bbox = self.get_bbox_corners(10, 29, 15, 17)
            # dlon, dlat = 180, 10; aspect = 18
            assert bbox == (-75, 105, 65, 75)

        def test_north_aspect(self):
            """Near north pole, with specified aspect ratio."""
            bbox = self.get_bbox_corners(10, 29, 16, 18, aspect=2)
            # dlat = dlon / aspect = 180 / 2 = 90
            # ddlat = 90 - 10 = 80
            assert bbox == (-75, 105, -5, 85)

        def test_south_no_aspect(self):
            """Near south pole, w/o specified aspect ratio."""
            bbox = self.get_bbox_corners(15, 24, 1, 4)
            # dlon, dlat = 80, 20; aspect = 4
            assert bbox == (-25, 55, -75, -55)

        def test_south_aspect(self):
            """Near south pole, with specified aspect ratio."""
            bbox = self.get_bbox_corners(15, 24, 1, 4, aspect=1)
            # dlat = dlon / aspect = 80 / 1 = 80
            # ddlat = 80 - 20 = 60
            assert bbox == (-25, 55, -85, -5)

    class Test_AcrossDateline(Continuous):
        """Cloud stretching across dateline."""

        def test_non_periodic_wrong(self):
            """Results are wrong if domain is not specified as periodic."""
            bbox = self.get_bbox_corners(-3, 2, 10, 17, periodic_lon=False)
            assert bbox == (-175, 175, 15, 75)

        def test_periodic_ok(self):
            """Results are correct if domain is specified as periodic."""
            bbox = self.get_bbox_corners(-3, 2, 10, 17, periodic_lon=True)
            assert bbox == (155, -165, 15, 75)

        def test_zonal_no_aspect(self):
            """Zonally extended cloud, w/o specified aspect ratio."""
            bbox = self.get_bbox_corners(-5, 4, 13, 15, periodic_lon=True)
            # dlon, dlat = 80, 10; aspect = 8
            assert bbox == (135, -145, 45, 55)

        def test_zonal_aspect(self):
            """Zonally extended cloud, with specified aspect ratio."""
            bbox = self.get_bbox_corners(-5, 4, 13, 15, periodic_lon=True, aspect=1.5)
            # dlat = dlon / aspect = 80 / 1.5 = 53.333
            # ddlat = 53.333 - 10 = 43.333
            ddlat2 = (43 + 1 / 3) / 2
            assert bbox == (135, -145, 45 - ddlat2, 55 + ddlat2)

        def test_meridional_no_aspect(self):
            """Meridionally extended cloud, w/o specified aspect ratio."""
            bbox = self.get_bbox_corners(-1, 2, 10, 17, periodic_lon=True)
            # dlon, dlat = 20, 60; aspect = 0.333
            assert bbox == (175, -165, 15, 75)

        def test_meridional_aspect(self):
            """Meridionally extended cloud, with specified aspect ratio."""
            bbox = self.get_bbox_corners(-1, 2, 10, 17, periodic_lon=True, aspect=1.5)
            # dlon = aspect * dlat = 1.5 * 60 = 90
            # ddlon = 90 - 20 = 70
            ddlon2 = 70 / 2
            assert bbox == (175 - ddlon2, -165 + ddlon2, 15, 75)

    class Test_NonContinuousNearDateline(General):
        """Non-continuous cloud around the dateline."""

        def test_continuous(self):
            """W/o specified aspect ratio but with periodic zonal boundaries."""
            bbox = self.get_bbox_corners([(-5, 4, 10, 17)], periodic_lon=True)
            assert bbox == (135, -145, 15, 75)

        def test_non_continuous_non_period(self):
            """W/o specified aspect ratio and w/o periodic zonal boundaries."""
            bbox = self.get_bbox_corners([(30, 33, 10, 17), (1, 4, 10, 17)])
            assert bbox == (-165, 145, 15, 75)

        def test_non_continuous_periodic(self):
            """W/o specified aspect ratio but with periodic zonal boundaries."""
            bbox = self.get_bbox_corners(
                [(31, 33, 10, 17), (1, 4, 10, 17)], periodic_lon=True
            )
            assert bbox == (135, -145, 15, 75)

    class Test_NonContinuousAcrossPoleAndDateline(General):
        """Non-continuous cloud stretching across North Pole and dateline."""

        bbox_dateline = (-4, 2, 15, 17)
        bbox_pole_east = (3, 12, 16, 18)
        bbox_pole_west = (21, 30, 16, 18)

        def test_dateline(self):
            """Subcloud away from the pole stretching across dateline."""
            bbox = self.get_bbox_corners([self.bbox_dateline], periodic_lon=True)
            assert bbox == (145, -165, 65, 75)

        def test_pole_east(self):
            """Part east of the dateline of the cloud over the pole."""
            bbox = self.get_bbox_corners([self.bbox_pole_east], periodic_lon=True)
            assert bbox == (-145, -65, 75, 85)

        def test_pole_west(self):
            """Part west of the dateline of the cloud over the pole."""
            bbox = self.get_bbox_corners([self.bbox_pole_west], periodic_lon=True)
            assert bbox == (35, 115, 75, 85)

        def test_combined(self):
            """Combine all subclouds."""
            bbox = self.get_bbox_corners(
                [self.bbox_dateline, self.bbox_pole_west, self.bbox_pole_east],
                periodic_lon=True,
            )
            assert bbox == (35, -65, 65, 85)

        def test_combined_aspect(self):
            """Combine all subclouds and fix aspect ratio."""
            bbox = self.get_bbox_corners(
                [self.bbox_dateline, self.bbox_pole_west, self.bbox_pole_east],
                periodic_lon=True,
                aspect=2,
            )
            assert bbox == (35, -65, -45, 85)


class Test_CloudDomain_RealCase:
    """Test cloud domain based on a real case that yielded wrong results."""

    # fmt: off
    lat = np.array([
        -89.875, -87.375, -84.875, -82.375, -79.875, -77.375, -74.875, -72.375, -69.875,  # noqa
        -67.375, -64.875, -62.375, -59.875, -57.375, -54.875, -52.375, -49.875, -47.375,  # noqa
        -44.875, -42.375, -39.875, -37.375, -34.875, -32.375, -29.875, -27.375, -24.875,  # noqa
        -22.375, -19.875, -17.375, -14.875, -12.375,  -9.875,  -7.375,  -4.875,  -2.375,  # noqa
          0.125,   2.625,   5.125,   7.625,  10.125,  12.625,  15.125,  17.625,  20.125,  # noqa
         22.625,  25.125,  27.625,  30.125,  32.625,  35.125,  37.625,  40.125,  42.625,  # noqa
         45.125,  47.625,  50.125,  52.625,  55.125,  57.625,  60.125,  62.625,  65.125,  # noqa
         67.625,  70.125,  72.625,  75.125,  77.625,  80.125,  82.625,  85.125,  87.625,  # noqa
    ])
    lon = np.array([
        -179.875, -177.375, -174.875, -172.375, -169.875, -167.375, -164.875, -162.375,  # noqa
        -159.875, -157.375, -154.875, -152.375, -149.875, -147.375, -144.875, -142.375,  # noqa
        -139.875, -137.375, -134.875, -132.375, -129.875, -127.375, -124.875, -122.375,  # noqa
        -119.875, -117.375, -114.875, -112.375, -109.875, -107.375, -104.875, -102.375,  # noqa
         -99.875,  -97.375,  -94.875,  -92.375,  -89.875,  -87.375,  -84.875,  -82.375,  # noqa
         -79.875,  -77.375,  -74.875,  -72.375,  -69.875,  -67.375,  -64.875,  -62.375,  # noqa
         -59.875,  -57.375,  -54.875,  -52.375,  -49.875,  -47.375,  -44.875,  -42.375,  # noqa
         -39.875,  -37.375,  -34.875,  -32.375,  -29.875,  -27.375,  -24.875,  -22.375,  # noqa
         -19.875,  -17.375,  -14.875,  -12.375,   -9.875,   -7.375,   -4.875,   -2.375,  # noqa
           0.125,    2.625,    5.125,    7.625,   10.125,   12.625,   15.125,   17.625,  # noqa
          20.125,   22.625,   25.125,   27.625,   30.125,   32.625,   35.125,   37.625,  # noqa
          40.125,   42.625,   45.125,   47.625,   50.125,   52.625,   55.125,   57.625,  # noqa
          60.125,   62.625,   65.125,   67.625,   70.125,   72.625,   75.125,   77.625,  # noqa
          80.125,   82.625,   85.125,   87.625,   90.125,   92.625,   95.125,   97.625,  # noqa
         100.125,  102.625,  105.125,  107.625,  110.125,  112.625,  115.125,  117.625,  # noqa
         120.125,  122.625,  125.125,  127.625,  130.125,  132.625,  135.125,  137.625,  # noqa
         140.125,  142.625,  145.125,  147.625,  150.125,  152.625,  155.125,  157.625,  # noqa
         160.125,  162.625,  165.125,  167.625,  170.125,  172.625,  175.125,  177.625,  # noqa
    ])
    raw_pts = np.array([
        [
            55, 55, 55, 55, 56, 56, 56, 56, 57, 57, 57, 57, 57, 58, 58, 58, 58,  # noqa
            58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59,  # noqa
            59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60,  # noqa
            60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,  # noqa
            60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,  # noqa
            60, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,  # noqa
            61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,  # noqa
            61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,  # noqa
            61, 61, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62,  # noqa
            62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62,  # noqa
            62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62,  # noqa
            62, 62, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,  # noqa
            63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,  # noqa
            63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,  # noqa
            63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64,  # noqa
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,  # noqa
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,  # noqa
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,  # noqa
            64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,  # noqa
            65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,  # noqa
            65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,  # noqa
            65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66,  # noqa
            66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,  # noqa
            66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,  # noqa
            66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,  # noqa
            66, 66, 66, 66, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67, 67, 67,  # noqa
            67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67,  # noqa
            67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67,  # noqa
            67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67,  # noqa
            67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67,  # noqa
            67, 67, 67, 67, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,  # noqa
            68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,  # noqa
            68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,  # noqa
            68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,  # noqa
            68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68,  # noqa
            68, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,  # noqa
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,  # noqa
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,  # noqa
            69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69,  # noqa
            69, 69, 69, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70,  # noqa
            70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70,  # noqa
            70, 70, 70, 70, 70, 70, 70, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71,  # noqa
            71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71,  # noqa
            71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71,  # noqa
            71, 71, 71, 71, 71, 71, 71, 71, 71,  # noqa
        ],
        [
            73,  74,  75,  76,  72,  73,  74,  75,  71,  72,  73,  74,  75,  # noqa
            71,  72,  73,  74,  75,  57,  58,  59,  60,  61,  62,  63,  64,  # noqa
            70,  71,  72,  73,  74,  75,  76,  87,  88,  89,  90,  91,  92,  # noqa
            93,  94,  95,  96,  97,  98,  99,  54,  55,  56,  57,  58,  59,  # noqa
            60,  61,  62,  63,  64,  65,  70,  71,  72,  73,  74,  75,  76,  # noqa
            86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  # noqa
            99, 100, 101, 102, 103, 104, 105, 106,  53,  54,  55,  56,  57,  # noqa
            58,  59,  60,  61,  62,  63,  64,  65,  66,  70,  71,  72,  73,  # noqa
            74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  # noqa
            87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  # noqa
           100, 101, 102, 103, 104, 105, 106, 107,  53,  54,  55,  56,  57,  # noqa
            58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  69,  70,  71,  # noqa
            72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  # noqa
            85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  # noqa
            98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108,  51,  52,  # noqa
            53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  # noqa
            66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  # noqa
            79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  # noqa
            92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104,  # noqa
           105, 106, 107, 108,  27,  49,  50,  51,  52,  53,  54,  55,  56,  # noqa
            57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  # noqa
            70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  # noqa
            83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  # noqa
            96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107,  48,  # noqa
            49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  # noqa
            62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  # noqa
            75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  # noqa
            88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100,  # noqa
           101, 102, 103, 104, 105,  28,  30,  44,  45,  47,  48,  49,  50,  # noqa
            51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  # noqa
            64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  # noqa
            77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  # noqa
            90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102,  # noqa
           103, 104, 105, 106, 107,  25,  26,  27,  28,  29,  30,  31,  42,  # noqa
            43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  # noqa
            56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  # noqa
            69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  # noqa
            82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  # noqa
            95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107,  # noqa
           108, 109, 110, 111, 112, 113, 114,  26,  27,  28,  29,  30,  31,  # noqa
            32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  # noqa
            45,  46,  47,  48,  49,  50,  51,  53,  54,  55,  56,  57,  58,  # noqa
            59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  # noqa
            72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  # noqa
            85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  # noqa
            98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,  # noqa
           111, 112, 113, 114,  30,  31,  32,  33,  34,  36,  47,  48,  49,  # noqa
            50,  51,  52,  59,  61,  62,  63,  64,  65,  66,  67,  68,  69,  # noqa
            70,  71,  72,  73,  74,  75,  76,  80,  81,  82,  85,  86,  87,  # noqa
            88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100,  # noqa
           101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,  # noqa
           114, 115, 116,  39,  40,  41,  42,  43,  45,  50,  88,  89,  90,  # noqa
            91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,  # noqa
           104, 105, 106, 107, 108, 109, 110, 111, 112, 119, 120, 123, 124,  # noqa
           125, 126,   6,   7,   8,   9,  10,  87,  88,  89,  90,  91,  92,  # noqa
            93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105,  # noqa
           106, 107, 108, 109, 110, 111, 112, 113, 114, 118, 119, 120, 121,  # noqa
           122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,  # noqa
           135, 136, 138,  # noqa
        ]
    ])
    # fmt: on
    config = {
        "rel_offset": (0.0, 0.0),
        "zoom_fact": 0.9,
        "aspect": 1.447,
        "min_size_lat": None,
        "min_size_lon": 20.0,
        "periodic_lon": True,
        "release_lat": 47.55,
        "release_lon": 8.23,
    }

    def get_mask(self):
        lat, lon = self.lat, self.lon
        pts = [[y, x] for y, x in self.raw_pts.T if y < lat.size and x < lon.size]
        idcs = tuple(map(np.array, np.array(list(map(tuple, pts))).T))
        mask = np.full((lat.size, lon.size), False)
        mask[idcs] = True
        assert mask.sum() == len(idcs[0]) > 0
        return mask

    def get_domain(self, **config):
        mask = self.get_mask()
        config = {**self.config, **config}
        return CloudDomain(self.lat, self.lon, mask=mask, config=config)

    def test_no_aspect(self):
        """Don't impose aspect ratio."""
        domain = self.get_domain(aspect=None)
        clon, clat = domain.get_center()
        assert 65 < clat < 70 and 40 < clon < 45

    def test_aspect(self):
        """Impose aspect ratio."""
        domain = self.get_domain()
        clon, clat = domain.get_center()
        assert -2 < clat < -1 and 40 < clon < 45


@dc.dataclass
class FindGapsConfig:
    mask: List[int]
    gaps: List[Tuple[int, int, int]]
    periodic: bool = True


@pytest.mark.parametrize(
    "cfg",
    [
        FindGapsConfig(  # [cfg0]
            mask=[0, 0, 0, 0],
            gaps=[],
        ),
        FindGapsConfig(  # [cfg1]
            mask=[0, 0, 0, 0],
            gaps=[(4, 0, 3)],
            periodic=False,
        ),
        FindGapsConfig(  # [cfg2]
            mask=[1, 0, 0, 1],
            gaps=[(2, 1, 2)],
        ),
        FindGapsConfig(  # [cfg3]
            mask=[1, 0, 0, 1],
            gaps=[(2, 1, 2)],
            periodic=False,
        ),
        FindGapsConfig(  # [cfg4]
            mask=[0, 1, 1, 0],
            gaps=[(2, 3, 0)],
        ),
        FindGapsConfig(  # [cfg5]
            mask=[0, 1, 1, 0],
            gaps=[(1, 0, 0), (1, 3, 3)],
            periodic=False,
        ),
        FindGapsConfig(  # [cfg6]
            mask=[0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            gaps=[(1, 3, 3), (3, 5, 7), (4, 9, 0)],
        ),
        FindGapsConfig(  # [cfg7]
            mask=[0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            gaps=[(1, 0, 0), (1, 3, 3), (3, 5, 7), (3, 9, 11)],
            periodic=False,
        ),
    ],
)
def test_find_gaps(cfg):
    gaps = find_gaps(cfg.mask, periodic=cfg.periodic)
    assert gaps == cfg.gaps
