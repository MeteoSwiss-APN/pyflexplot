# Changelog

## Version 2.7.1

### Features

- Rasterize additional layer (country borders) to optimize PDF file size. Reduzes PDF by 30% - 40%.

### Bug Fixes

- Handle `zorder` argument correctly in `_ax_add_cities`.

### Breaking Changes

- None

## Version 2.0.1 (2024-06-13)

- Integrate poetry dependency management substituting conda development environments (following MCH python templates)
- Automate distributable package generation using poetry
- Configure CI/CD pipeline to run in MCH Jenkins server
- Containerize pyflexplot tool to be able to run it within a container
- Integrate AWS deployment

## Version 1.1.1 (2024-02-28)

- Add presets and adapt code to plot FLEXPART-ICON output. by @pirmink in [#37](https://github.com/MeteoSwiss-APN/pyflexplot/pull/37)

## Version 1.1.0 (2024-01-16)

- PyFlexPlot produces PDF or PNG plots from FLEXPART output in NetCDF format. This release adds the capability to write out shapefiles.
