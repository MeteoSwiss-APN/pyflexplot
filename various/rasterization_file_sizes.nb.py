#
# %% Imports
# Standard library
import re

# Third-party
import matplotlib.pyplot as plt
import numpy as np

# mpl.rc("text", usetex=True)


# %% Data
# fmt: off
file_names = [
    "20200216T0000_Goesgen_FLEXPART-COSMO-1E_Dispersion_Deposition.pdf",  # noqa
    "20200216T0000_Goesgen_FLEXPART-COSMO-1E_Dispersion_Deposition_zoom.pdf",  # noqa
    "20200216T0000_Goesgen_FLEXPART-COSMO-1E_Dispersion_Konzentration.pdf",  # noqa
    "20200216T0000_Goesgen_FLEXPART-COSMO-1E_Dispersion_Konzentration_zoom.pdf",  # noqa
    "20200216T0000_Goesgen_FLEXPART-COSMO-1E_Dispersion_beaufschl_Gebiet.pdf",  # noqa
    "20200216T0000_Goesgen_FLEXPART-COSMO-1E_Dispersion_beaufschl_Gebiet_zoom.pdf",  # noqa
    "20200216T0000_Goesgen_FLEXPART-COSMO-1E_Dispersion_integrierte_Luftaktivitaet.pdf",  # noqa
    "20200216T0000_Goesgen_FLEXPART-COSMO-1E_Dispersion_integrierte_Luftaktivitaet_zoom.pdf",  # noqa
    "20200818T0000_Bushehr_FLEXPART-IFS-HRES_Dispersion_Deposition.pdf",  # noqa
    "20200818T0000_Bushehr_FLEXPART-IFS-HRES_Dispersion_Deposition_zoom.pdf",  # noqa
    "20200818T0000_Bushehr_FLEXPART-IFS-HRES_Dispersion_Konzentration.pdf",  # noqa
    "20200818T0000_Bushehr_FLEXPART-IFS-HRES_Dispersion_Konzentration_zoom.pdf",  # noqa
    "20200818T0000_Bushehr_FLEXPART-IFS-HRES_Dispersion_beaufschl_Gebiet.pdf",  # noqa
    "20200818T0000_Bushehr_FLEXPART-IFS-HRES_Dispersion_beaufschl_Gebiet_zoom.pdf",  # noqa
    "20200818T0000_Bushehr_FLEXPART-IFS-HRES_Dispersion_integrierte_Luftaktivitaet.pdf",  # noqa
    "20200818T0000_Bushehr_FLEXPART-IFS-HRES_Dispersion_integrierte_Luftaktivitaet_zoom.pdf",  # noqa
    "20200818T0000_Goesgen_FLEXPART-IFS-HRES-EU_Dispersion_Deposition.pdf",  # noqa
    "20200818T0000_Goesgen_FLEXPART-IFS-HRES-EU_Dispersion_Deposition_zoom.pdf",  # noqa
    "20200818T0000_Goesgen_FLEXPART-IFS-HRES-EU_Dispersion_Konzentration.pdf",  # noqa
    "20200818T0000_Goesgen_FLEXPART-IFS-HRES-EU_Dispersion_Konzentration_zoom.pdf",  # noqa
    "20200818T0000_Goesgen_FLEXPART-IFS-HRES-EU_Dispersion_beaufschl_Gebiet.pdf",  # noqa
    "20200818T0000_Goesgen_FLEXPART-IFS-HRES-EU_Dispersion_beaufschl_Gebiet_zoom.pdf",  # noqa
    "20200818T0000_Goesgen_FLEXPART-IFS-HRES-EU_Dispersion_integrierte_Luftaktivitaet.pdf",  # noqa
    "20200818T0000_Goesgen_FLEXPART-IFS-HRES-EU_Dispersion_integrierte_Luftaktivitaet_zoom.pdf",  # noqa
]
file_sizes_by_case = {
    "orig": [22700, 16964, 19612, 13712, 664, 408, 1772, 1292, 17940, 17404, 18052, 17420, 524, 508, 1076, 1036, 63356, 71436, 61372, 69316, 732, 828, 3700, 4172],  # noqa
    "all_300dpi": [23064, 15476, 22820, 15608, 940, 644, 1932, 1328, 16856, 16592, 17604, 16648, 476, 468, 1080, 1008, 90384, 73784, 99696, 75664, 1184, 828, 6012, 4400],  # noqa
    "all_150dpi": [11696, 7880, 11596, 7928, 452, 312, 928, 640, 9164, 8956, 9492, 9000, 248, 240, 548, 516, 52860, 45568, 49904, 37836, 572, 392, 2904, 2076],  # noqa
    "all_30dpi": [3140, 2356, 3140, 2364, 84, 72, 168, 144, 3088, 3068, 3156, 3080, 60, 64, 136, 132, 23836, 25196, 11508, 10672, 100, 80, 500, 416],  # noqa
    "contourf_30dpi": [15752, 9560, 15632, 9448, 648, 388, 1316, 800, 18084, 17548, 18148, 17524, 524, 508, 1084, 1044, 61112, 69420, 60924, 69208, 716, 812, 3604, 4116],  # noqa
    "zorder-frames_150dpi": [11684, 7864, 11584, 7908, 452, 312, 928, 640, 9132, 8912, 9460, 8960, 244, 240, 548, 512, 52804, 45528, 49828, 37776, 572, 392, 2900, 2076],  # noqa
    "zorder-marker_150dpi": [11684, 7860, 11580, 7908, 452, 312, 928, 640, 9128, 8908, 9456, 8956, 244, 236, 548, 512, 52804, 45520, 49824, 37760, 572, 392, 2900, 2072],  # noqa
    "zorder-grid_150dpi": [9824, 7196, 9720, 7252, 368, 280, 756, 580, 7884, 7784, 8196, 7804, 204, 200, 452, 428, 49352, 44056, 45452, 35984, 512, 368, 2608, 1928],  # noqa
    "zorder-geo-upper_150dpi": [14964, 10464, 14900, 10604, 600, 428, 1220, 876, 15496, 15208, 15796, 15240, 436, 424, 912, 876, 64844, 62464, 66444, 60980, 756, 644, 3880, 3336],  # noqa
    "zorder-fld_150dpi": [22052, 18312, 18964, 15060, 636, 468, 1716, 1412, 15732, 15492, 16024, 15508, 456, 448, 948, 920, 67944, 66712, 67496, 63020, 808, 748, 4084, 3776],  # noqa
    "zorder-geo-lower_150dpi": [21488, 17412, 18400, 14164, 608, 428, 1664, 1332, 15924, 15772, 16080, 15788, 460, 456, 952, 936, 64824, 64336, 63328, 59848, 756, 708, 3820, 3576],  # noqa
    "zorder-lowest_150dpi": [22700, 16964, 19612, 13712, 664, 408, 1772, 1292, 17940, 17404, 18052, 17420, 524, 508, 1076, 1036, 63356, 71436, 61372, 69316, 732, 828, 3700, 4172],  # noqa
    "custom_150dpi": [9280, 9324, 8480, 8744, 328, 352, 764, 792, 13724, 13320, 13720, 13264, 376, 364, 896, 860, 57864, 33720, 56620, 32552, 676, 368, 3420, 2192],  # noqa
    "custom_100dpi": [

    ],  # noqa
}
# fmt: on


# %%
def group_file_names_by_model():
    rx = re.compile(
        r"(?P<base>[0-9]{8}T[0-9]{4})_(?P<site>\w+)_FLEXPART-(?P<model>[\w-]+)"
        r"_Dispersion_([\w_]+)(_zoom)?.pdf",
    )
    file_names_by_model = {}
    for file_name in file_names:
        match = rx.match(file_name)
        if match:
            model = match.group("model")
            if model not in file_names_by_model:
                file_names_by_model[model] = []
            file_names_by_model[model].append(file_name)
        else:
            raise Exception(
                "file name did not match pattern"
                f"\nfile name: {file_name}\npattern: {rx}"
            )
    return file_names_by_model


# %%
def plot_raw_files(path, cases=None):
    """Plot the sizes of individual files for selected cases as a line plot."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.grid(axis="y")

    yticks = list(range(len(file_names)))
    ax.set_yticks(yticks)
    ax.set_yticklabels(file_names)
    ax.yaxis.tick_right()
    ax.set_xlabel(r"file size (512B)")

    for case in cases or file_sizes_by_case:
        sizes = file_sizes_by_case[case]
        ax.plot(sizes, yticks, "o-", label=case)
    ax.legend()

    ax.set_aspect(1.5 / ax.get_data_ratio())

    fig.savefig(path, bbox_inches="tight")


plot_raw_files("rasterization_file_sizes_all.pdf")
# plot_raw_files(["orig", "all_300dpi", "all_150dpi", "contourf_30dpi"])
zorder_cases = ["orig"] + list(
    filter(lambda case: case.startswith("zorder"), file_sizes_by_case)
)
plot_raw_files("rasterization_file_sizes_zorder.pdf", cases=zorder_cases)


# %%
def plot_grouped_files(path):
    """Plot the sizes of individual files grouped by model as a scatter plot."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(axis="y")

    cases = list(file_sizes_by_case)
    # cases = [case.replace("_", r"\_") for case in file_sizes_by_case]
    yticks = np.arange(len(cases))[::-1] * 2
    ax.set_yticks(yticks)
    ax.set_yticklabels(cases)
    # ax.set_xlabel(r"file size ($\times 512$ bytes)")
    ax.set_xlabel(r"file size (512B)")

    file_names_by_model = group_file_names_by_model()

    models = list(file_names_by_model)
    colors = ["red", "green", "blue"]
    colors_by_model = dict(zip(models, colors))

    for ytick, sizes in zip(yticks, file_sizes_by_case.values()):
        for model, file_names_i in list(file_names_by_model.items())[::-1]:
            idcs = list(map(file_names.index, file_names_i))
            sizes_model = [sizes[idx] for idx in idcs]
            color = colors_by_model[model]
            label = model if ytick == yticks[0] else None
            y = ytick + (models.index(model) - 1) * 0.25
            ys = [y] * len(sizes_model)
            ax.plot(sizes_model, ys, "x", color=color, label=label)
    ax.legend()

    fig.savefig(path, bbox_inches="tight")


plot_grouped_files("rasterization_file_sizes_by_model.pdf")

# %%
