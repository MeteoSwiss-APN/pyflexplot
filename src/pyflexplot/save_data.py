"""
This module contains the implementation for data saving strategies, 
including shapefile creation.
"""
# Standard library
import os
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

# Third-party
import numpy as np
import shapefile

# Local
from pyflexplot.plot_layouts import BoxedPlotLayout
from pyflexplot.plotting.boxed_plot import BoxedPlot, BoxedPlotConfig
from pyflexplot.plotting.text_box_axes import TextBoxAxes
from .input.field import FieldGroup
from .utils.logging import log


class DataSaverFactory:
    @staticmethod
    def create_saver(filename: str) -> "DataSaver":
        _, extension = os.path.splitext(filename)
        if extension == ".pdf":
            return DataSaver(GeneralDataSaver())
        elif extension == ".png":
            return DataSaver(GeneralDataSaver())
        elif extension == ".shp":
            return DataSaver(ShapeSaver())
        else:
            raise ValueError(f"Unsupported file extension: {extension}")


class DataSaver:
    def __init__(self, strategy) -> None:
        self.strategy = strategy

    def save(self, filename: str, plot: BoxedPlot, data: FieldGroup) -> None:
        return self.strategy.save(filename, plot, data)


class GeneralDataSaver:
    def save(self, filename: str, plot: BoxedPlot, _data: FieldGroup) -> None:
        plot.write(filename)
        # SR_TMP < TODO clean this up; add optional setup param for file name
        if "standalone_release_info" in plot.config.labels:
            self.write_standalone_release_info(
                filename,
                plot.config,
            )

    def write_standalone_release_info(
        self, plot_path: str, plot_config: BoxedPlotConfig
    ) -> str:
        path = Path(plot_path).with_suffix(f".release_info{Path(plot_path).suffix}")
        log(inf=f"write standalone release info to {path}")
        layout = BoxedPlotLayout(
            plot_config.setup.layout.derive({"type": "standalone_release_info"}),
            aspects={"tot": 1.5},
            rects={"tot": (0, 0, 1, 1)},
        )
        species_id = plot_config.setup.panels.collect_equal("dimensions.species_id")
        n_species = 1 if isinstance(species_id, int) else len(species_id)
        width = 1.67 + 0.67 * n_species
        config = BoxedPlotConfig(
            fig_size=(width, 2.5),
            layout=layout,
            labels=plot_config.labels,
            panels=[],
            setup=plot_config.setup,
        )

        def fill_box(box: TextBoxAxes, plot: BoxedPlot) -> None:
            labels = plot.config.labels["standalone_release_info"]

            # Box title
            box.text(
                s=labels["title"],
                loc="tc",
                fontname=plot.config.font.name,
                size=plot.config.font.sizes.title_small,
            )

            # Add lines bottom-up (to take advantage of baseline alignment)
            box.text_blocks_hfill(
                labels["lines_str"],
                dy_unit=-10.0,
                dy_line=8.0,
                fontname=plot.config.font.name,
                size=plot.config.font.sizes.content_small,
            )

        plot = BoxedPlot(config)
        plot.add_text_box("standalone_release_info", (0, 0, 1, 1), fill=fill_box)
        plot.write(path)
        return str(path)


class ShapeSaver:
    def save(self, filename: str, _plot: BoxedPlot, data: FieldGroup) -> None:
        base_file_dir, _ = os.path.splitext(filename)
        with zipfile.ZipFile(f"{filename}.zip", "w") as zip_file:
            lat_values = []
            lon_values = []
            shapefile_writer = shapefile.Writer(filename, shapeType=shapefile.POINT)
            for n, field in enumerate(data):
                grid_north_pole_lat = field.mdata.simulation.grid_north_pole_lat
                grid_north_pole_lon = field.mdata.simulation.grid_north_pole_lon
                fld = field.fld.ravel()
                relevant_indices = np.where(fld > 0)
                fld = np.log10(fld[relevant_indices])
                field_name = f"{field.mdata.species.name}"
                shapefile_writer.field(field_name, "F", 8, 15)
                if len(fld) == 0:
                    shapefile_writer.point(field.mdata.release.lon, field.mdata.release.lat)
                    shapefile_writer.record(field_name=[-100.0])
                    lat_values.extend([field.mdata.release.lat])
                    lon_values.extend([field.mdata.release.lon])
                    continue

                coordinates = np.array(
                    [[lon, lat] for lat in field.lat for lon in field.lon]
                )[relevant_indices]

                true_lat = latrot2lat(
                    coordinates[:, 1], coordinates[:, 0], grid_north_pole_lat
                )
                true_lon = lonrot2lon(
                    coordinates[:, 1],
                    coordinates[:, 0],
                    grid_north_pole_lat,
                    grid_north_pole_lon,
                )
                coordinates = [[lon, lat] for lon, lat in zip(true_lon, true_lat)]
                lat_values.extend([np.max(true_lat), np.min(true_lat)])
                lon_values.extend([np.max(true_lon), np.min(true_lon)])
                for coord, conc in zip(coordinates, fld):
                    shapefile_writer.point(*coord)
                    shapefile_writer.record(conc)

            shapefile_writer.close()

            min_lon, max_lon = np.min(lon_values), np.max(lon_values)
            min_lat, max_lat = np.min(lat_values), np.max(lat_values)
            domain_metadata = { 
                'DomainBoundaries': [min_lon, max_lon, min_lat, max_lat]
            }
            #print("KKKK ", _plot.config.labels)
            self._write_metadata_file(base_file_dir, zip_file, {**_plot.config.labels, **domain_metadata})
            self._move_shape_file_to_zip(base_file_dir, zip_file)
            # Some GIS Software require information about the projection
            self._write_projection_file_to_zip(
                os.path.basename(base_file_dir), zip_file
            )
            zip_file.close()

    def _move_shape_file_to_zip(self, base_file_dir: str, zip_file: zipfile.ZipFile):
        extensions = [".shp", ".shx", ".dbf"]
        if "_domain" not in base_file_dir:
            extensions += [".shp.xml"]
        for ext in extensions:
            file_to_copy = f"{base_file_dir}{ext}"
            with open(file_to_copy, "rb") as file_in_zip:
                zip_file.writestr(
                    f"{os.path.basename(base_file_dir)}{ext}", file_in_zip.read()
                )
            file_in_zip.close()
            os.remove(file_to_copy)

    def _write_metadata_file(
        self, filename: str, zip_file: zipfile.ZipFile, metadata: dict = None
    ):
        title_string = list(metadata["title"].values())[0]
        title_latex_string = "<BR>".join(list(metadata["title"].values()))
        release_latex_string = metadata["release_info"]["lines_str"].replace(
            "\n", "<BR>"
        )
        model_info_string = "<BR>".join(list(metadata["bottom"].values()))
        replacements = {
            "\\\\": "",  # Remove double backslashes
            "\\mathrm": "",  # Remove the \mathrm command
            "\\circ": "°",  # Degree symbol
            '\\,"': "",  # Remove small space
            '\\"o': "ö",  # ö character
            '\\"a': "ä",  # ä character
            '\\"u': "ü",  # ü character
            "$": "",  # Remove dollar signs used for math mode
            "^": "",  # Superscript (might not be needed in plain text)
            "s^{-1}": "s-1",  # Inverse seconds
            "\\t": " ",  # Tab character
            "\\n": " ",
            "{": "",
            "}": "",
            "\\,": " ",
        }

        for key, value in replacements.items():
            release_latex_string = release_latex_string.replace(key, value)
            title_latex_string = title_latex_string.replace(key, value)
            title_string = title_string.replace(key, value)
            model_info_string = model_info_string.replace(key, value)

        total_content = f"{title_latex_string}<BR><BR>{release_latex_string}<BR><BR>{model_info_string}"
        domain_boundaries = metadata['DomainBoundaries']
        xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<metadata>
    <dataIdInfo>
    <dataExt>
        <geoEle>
            <GeoBndBox>
                <westBL>{domain_boundaries[0]}</westBL>
                <eastBL>{domain_boundaries[1]}</eastBL>
                <northBL>{domain_boundaries[2]}</northBL>
                <southBL>{domain_boundaries[3]}</southBL>
            </GeoBndBox>
        </geoEle>
    </dataExt>
    <idCitation><resTitle>{title_string}</resTitle></idCitation>
        <idAbs>
            <![CDATA[{total_content}]]>
        </idAbs>
    </dataIdInfo>
</metadata>"""

        with open(f"{filename}.shp.xml", "w", encoding="utf-8") as f:
            f.write(xml_content)

    def _write_projection_file_to_zip(self, base_name: str, zip_file: zipfile.ZipFile):
        proj_file_content = (
            'GEOGCS["WGS 84",DATUM["WGS_1984",'
            'SPHEROID["WGS 84",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433]]'
        )
        zip_file.writestr(f"{base_name}.prj", proj_file_content)


def latrot2lat(
    phirot: np.ndarray, rlarot: np.ndarray, polphi: float, polgam: float = None
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
        zarg = zcospol * np.cos(zphis) * np.cos(zrlas) + zsinpol * np.sin(zphis)

    return zrpi18 * np.arcsin(zarg)


def lonrot2lon(
    phirot: np.ndarray,
    rlarot: np.ndarray,
    polphi: float,
    pollam: float,
    polgam: float = None,
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
            -zsinpol * np.cos(zrlas) * np.cos(zphis) + zcospol * np.sin(zphis)
        ) - np.cos(zlampol) * np.sin(zrlas) * np.cos(zphis)

        zarg2 = np.cos(zlampol) * (
            -zsinpol * np.cos(zrlas) * np.cos(zphis) + zcospol * np.sin(zphis)
        ) + np.sin(zlampol) * np.sin(zrlas) * np.cos(zphis)

    zarg2 = np.where(zarg2 == 0.0, 1.0e-20, zarg2)

    return zrpi18 * np.arctan2(zarg1, zarg2)
