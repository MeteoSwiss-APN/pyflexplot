"""Test feature to generate shape files."""
# Standard library
import os
import zipfile

# Third-party
import numpy as np
import shapefile

# First-party
from pyflexplot.data_transformation.rotated_pole import latrot2lat
from pyflexplot.data_transformation.rotated_pole import lonrot2lon
from pyflexplot.save_data import ShapeFileSaver

# Local
from .shared import _TestBase
from .shared import _TestCreatePlot  # noqa:F401
from .shared import _TestCreateReference  # noqa:F401
from .shared import datadir  # noqa:F401  # required by _TestBase.test

INFILE_1 = "flexpart_cosmo-1_2019093012.nc"

# Uncomment to create plots for all tests
# _TestBase = _TestCreatePlot


# Uncomment to references for all tests
# _TestBase = _TestCreateReference
class Test_ShapeFileGeneration(_TestBase):
    reference = "ref_cosmo-1_concentration"
    setup_dct = {
        "files": {
            "input": INFILE_1,
            "output": f"{reference}.shp",
        },
        "model": {
            "name": "COSMO-1",
        },
        "panels": [
            {
                "plot_variable": "concentration",
                "integrate": False,
                "lang": "de",
                "domain": "full",
                "dimensions": {
                    "species_id": 1,
                    "time": 5,
                    "level": 0,
                    "multiplier": 100,
                },
            }
        ],
    }

    file_name = ""

    def test(self, datadir):
        shape_file_saver = ShapeFileSaver()
        field_group = self.get_field_group(datadir)
        plot = self.get_plot(field_group)
        file_name = datadir + self.setup_dct["files"]["output"]
        shape_file_saver.save(filename=file_name, plot=plot, data=field_group)
        zip_shape_file = f"{file_name}.zip"
        assert os.path.exists(
            zip_shape_file
        ), "Zip shape file {zip_shape_file} was not found."
        zip_file = zipfile.ZipFile(zip_shape_file, "r")
        for ext in [".shp", ".shx", ".dbf", ".shp.xml", ".prj"]:
            assert np.any(
                [file.endswith(ext) for file in zip_file.namelist()]
            ), f"File in shape file with {ext} not found."
        zip_file.close()
        with shapefile.Reader(zip_shape_file) as sf:
            records = sf.records()
            shapes = sf.shapes()

        for n, field in enumerate(field_group):
            # Recreate the transformation and filtering logic from your function
            fld = field.fld.ravel()
            relevant_indices = np.where(fld > 0)
            fld = np.log10(fld[relevant_indices])

            coordinates = np.array(
                [[lon, lat] for lat in field.lat for lon in field.lon]
            )[relevant_indices]

            true_lat = latrot2lat(
                coordinates[:, 1],
                coordinates[:, 0],
                field.mdata.simulation.grid_north_pole_lat,
            )
            true_lon = lonrot2lon(
                coordinates[:, 1],
                coordinates[:, 0],
                field.mdata.simulation.grid_north_pole_lat,
                field.mdata.simulation.grid_north_pole_lon,
            )
            transformed_coordinates = np.array(
                [[lon, lat] for lon, lat in zip(true_lon, true_lat)]
            )

            # Compare shapefile data with original data
            for (shape, record), (coord, conc) in zip(
                zip(shapes, records), zip(transformed_coordinates, fld)
            ):
                assert np.all(
                    np.isclose(shape.points[0], [coord[0], coord[1]])
                ), "ERROR: Coordinates differ"
                assert np.isclose(
                    record[0], conc
                ), "ERROR: Input and output fields differ"

            os.remove(zip_shape_file)
