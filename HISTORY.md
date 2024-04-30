# History

## 2.0.x (2024-xx-xx)

- Integrate poetry dependency management substituting conda development environments (following MCH python templates)
- Automate distributable package generation using poetry
- Configure CI/CD pipeline to run in MCH Jenkins server
- Containerize pyflexplot tool to be able to run it within a container
- Integrate AWS deployment

## 1.1.1 (2024-02-28)

- Add presets and adapt code to plot FLEXPART-ICON output. by @pirmink in [#37](https://github.com/MeteoSwiss-APN/pyflexplot/pull/37)

## 1.1.0 (2024-01-16)

- PyFlexPlot produces PDF or PNG plots from FLEXPART output in NetCDF format. This release adds the capability to write out shapefiles.